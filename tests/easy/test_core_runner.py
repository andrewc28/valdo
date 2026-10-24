"""Runner building blocks: hints, log capture, datasets.csv contributions, fingerprints,
safe cleaning, the lock."""

import json
import os
import socket
import subprocess
import sys

import pytest

from valdo.easy import hints, runner
from valdo.easy.stages import BY_NAME, NAMES, STAGES, Result


# ---------------------------------------------------------------------------
# hints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage, text, expected", [
    ("train", "Traceback ...\nRuntimeError: CUDA out of memory. Tried to allocate", "cuda_oom"),
    ("scale", "numpy.linalg.LinAlgError: Singular matrix", "singular_matrix"),
    ("preprocess", "numpy.linalg.LinAlgError: Singular matrix", "unknown"),        # scale-only hint
    ("rescale", "OSError: [Errno 28] No space left on device", "disk_full"),
    ("standardize", "/bin/sh: line 1: 1234 Killed python", "memory"),
    ("filter", "KeyError: 'F-obs-scaled'", "missing_column"),
    ("reconstruct", "RuntimeError: Attempting to deserialize object on a CUDA device but ...", "gpu_pickle"),
    ("tag_blobs", "AttributeError: 'gemmi.UnitCell' object has no attribute 'fractionalization_matrix'", "gemmi_api"),
    ("preprocess", "ValueError: No objects to concatenate", "nothing_left"),
    ("train", "ModuleNotFoundError: No module named 'torch'", "no_torch"),
    ("scale", "something nobody has seen before", "unknown"),
])
def test_hints(stage, text, expected):
    h = hints.explain(stage, text)
    assert h["id"] == expected and h["what"] and h["fix"]


def test_hint_last_match_wins():
    text = "FileNotFoundError: x\n... later ...\nMemoryError"
    assert hints.explain("scale", text)["id"] == "memory"


def test_tail(tmp_path):
    p = tmp_path / "log"
    p.write_text("\n".join(f"line {i}" for i in range(100)))
    assert hints.tail(str(p), 3) == "line 97\nline 98\nline 99"
    assert hints.tail(str(tmp_path / "missing")) == ""


# ---------------------------------------------------------------------------
# log capture
# ---------------------------------------------------------------------------

def test_capture_output_gets_python_c_and_child_output(tmp_path):
    log = str(tmp_path / "stage.log")
    with runner.capture_output(log):
        print("from python print")
        sys.stderr.write("from sys.stderr\n")
        os.write(1, b"from fd 1\n")
        subprocess.run([sys.executable, "-c", "import sys; print('from a child'); sys.stderr.write('child err\\n')"])
    text = open(log).read()
    for piece in ("from python print", "from sys.stderr", "from fd 1", "from a child", "child err"):
        assert piece in text
    print("after capture")   # stdout is restored (pytest captures this)


def test_capture_output_echo(tmp_path):
    import io
    log = str(tmp_path / "stage.log")
    echo = io.StringIO()
    with runner.capture_output(log, echo=echo):
        print("both places")
    assert "both places" in open(log).read() and "both places" in echo.getvalue()


# ---------------------------------------------------------------------------
# datasets.csv view and fingerprints
# ---------------------------------------------------------------------------

BASE = [
    {"id": "0001", "original_name": "a", "status": "ok", "flags": "reference", "variant": "", "dropped_stage": "",
     "dropped_reason": ""},
    {"id": "0002", "original_name": "b", "status": "ok", "flags": "", "variant": "", "dropped_stage": "",
     "dropped_reason": ""},
    {"id": "0003", "original_name": "c", "status": "dropped", "flags": "", "variant": "", "dropped_stage": "input",
     "dropped_reason": "unreadable"},
]


def test_merged_rows_applies_done_stages_in_order():
    state = {"stages": {
        "reindex": {"status": "done", "contrib": {"flags": {"0002": ["reindexed"]}, "dropped": {}, "values": {}}},
        "filter": {"status": "done", "contrib": {"dropped": {"0002": "low CC"},
                                                  "values": {"0001": {"variant": "0001_0", "reindex_op": "h,k,l"}},
                                                  "flags": {}}},
        "preprocess": {"status": "done", "contrib": {"dropped": {"0002": "should not override"}}},
        "train": {"status": "failed", "contrib": {"dropped": {"0001": "ignored: stage not done"}}},
    }}
    rows = {r["id"]: r for r in runner.merged_rows(BASE, state)}
    assert rows["0001"]["status"] == "ok" and rows["0001"]["variant"] == "0001_0"
    assert rows["0002"]["status"] == "dropped" and rows["0002"]["dropped_stage"] == "filter"
    assert rows["0002"]["dropped_reason"] == "low CC" and rows["0002"]["flags"] == "reindexed"
    assert rows["0003"]["dropped_stage"] == "input"
    assert BASE[1]["status"] == "ok"   # base rows are not modified


def test_merged_rows_before_a_stage():
    """What a stage sees (and fingerprints) must not change when a later stage drops datasets."""
    state = {"stages": {"filter": {"status": "done", "contrib": {"dropped": {"0002": "low CC"}}}}}
    assert {r["id"]: r for r in runner.merged_rows(BASE, state, before="filter")}["0002"]["status"] == "ok"
    assert {r["id"]: r for r in runner.merged_rows(BASE, state, before="scale")}["0002"]["status"] == "ok"
    assert {r["id"]: r for r in runner.merged_rows(BASE, state, before="train")}["0002"]["status"] == "dropped"
    assert {r["id"]: r for r in runner.merged_rows(BASE, state)}["0002"]["status"] == "dropped"


def test_fingerprint_ignores_ncpu_and_tracks_everything_else():
    stage = BY_NAME["scale"]
    cfg = {"file_list": "/w/configs/scale_files.txt", "when_opt": "all", "ncpu": 4}
    fp = runner.fingerprint(stage, cfg, {"x": 1}, "tok")
    assert fp == runner.fingerprint(stage, dict(cfg, ncpu=32), {"x": 1}, "tok")
    assert fp != runner.fingerprint(stage, dict(cfg, when_opt="never"), {"x": 1}, "tok")
    assert fp != runner.fingerprint(stage, cfg, {"x": 2}, "tok")
    assert fp != runner.fingerprint(stage, cfg, {"x": 1}, "other token")


def test_stage_table():
    assert NAMES == ["standardize", "reindex", "scale", "refine", "filter", "preprocess", "train", "reconstruct",
                     "rescale", "add_phases_and_blobs", "tag_blobs", "report"]
    assert [s.number for s in STAGES] == list(range(1, 13))
    assert all(s.describe()["description"] for s in STAGES)
    from valdo.commandline.pipeline import STAGE_REGISTRY
    upstream = [s.name for s in STAGES if s.upstream == s.name]
    assert set(upstream) == set(STAGE_REGISTRY)


def test_result_contrib():
    res = Result()
    res.drop("0001", "first")
    res.drop("0001", "second reason is ignored")
    res.flag("0002", "x")
    res.flag("0002", "x")
    res.set("0002", variant="0002_1")
    assert res.contrib() == {"dropped": {"0001": "first"}, "flags": {"0002": ["x"]},
                             "values": {"0002": {"variant": "0002_1"}}}


# ---------------------------------------------------------------------------
# cleaning and lock
# ---------------------------------------------------------------------------

class _FakeRunner(runner.Runner):
    def __init__(self, workdir):  # only what _reset_dir needs
        self.workdir = workdir


def test_reset_dir_never_follows_symlinks(tmp_path):
    work = tmp_path / "run"
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("user data")
    r = _FakeRunner(str(work))
    r._reset_dir("03_scale")                                # creates folder + marker
    stage_dir = work / "03_scale"
    assert (stage_dir / runner.MARKER).exists()
    (stage_dir / "old.mtz").write_text("stale")
    os.symlink(str(outside), str(stage_dir / "link_to_dir"))
    os.symlink(str(outside / "keep.txt"), str(stage_dir / "link_to_file"))
    (stage_dir / "sub").mkdir()
    os.symlink(str(outside), str(stage_dir / "sub" / "nested_link"))
    r._reset_dir("03_scale")
    assert sorted(os.listdir(stage_dir)) == [runner.MARKER]
    assert (outside / "keep.txt").read_text() == "user data"


def test_reset_dir_refuses_a_symlinked_stage_folder(tmp_path):
    """Review F5: a stage folder that is itself a symbolic link (e.g. moved to another disk by hand)
    was emptied through the link."""
    work = tmp_path / "run"
    work.mkdir()
    elsewhere = tmp_path / "other_disk" / "03_scale"
    elsewhere.mkdir(parents=True)
    (elsewhere / runner.MARKER).write_text("created by valdo-easy\n")   # looks like valdo's own folder
    (elsewhere / "keep.mtz").write_text("data")
    os.symlink(str(elsewhere), str(work / "03_scale"))
    r = _FakeRunner(str(work))
    with pytest.raises(runner.RunError) as err:
        r._reset_dir("03_scale")
    assert err.value.code == "unsafe_path" and "symbolic link" in err.value.what and err.value.fix
    assert (elsewhere / "keep.mtz").read_text() == "data"
    # a link further down (a folder inside the stage folder) is refused too
    os.unlink(str(work / "03_scale"))
    (work / "04_refine").mkdir()
    (work / "04_refine" / runner.MARKER).write_text("x\n")
    os.symlink(str(elsewhere), str(work / "04_refine" / "runs"))
    with pytest.raises(runner.RunError):
        r._reset_dir("04_refine/runs")
    assert (elsewhere / "keep.mtz").exists()
    # the work directory itself may be a link
    real = tmp_path / "real_work"
    real.mkdir()
    os.symlink(str(real), str(tmp_path / "linked_work"))
    _FakeRunner(str(tmp_path / "linked_work"))._reset_dir("05_filter")
    assert (real / "05_filter" / runner.MARKER).exists()


def test_reset_dir_refuses_foreign_folders(tmp_path):
    work = tmp_path / "run"
    (work / "05_filter").mkdir(parents=True)
    (work / "05_filter" / "mine.txt").write_text("not valdo's")
    r = _FakeRunner(str(work))
    with pytest.raises(runner.RunError) as err:
        r._reset_dir("05_filter")
    assert err.value.code == "foreign_folder"
    assert (work / "05_filter" / "mine.txt").exists()
    with pytest.raises(runner.RunError):
        r._reset_dir("../escape")
    (work / "06_preprocess").mkdir()                        # an empty folder is adopted
    r._reset_dir("06_preprocess")
    assert (work / "06_preprocess" / runner.MARKER).exists()


def test_lock(tmp_path):
    lock = runner.Lock(str(tmp_path))
    lock.acquire()
    other = runner.Lock(str(tmp_path))
    with pytest.raises(runner.RunError) as err:
        other.acquire()
    assert err.value.code == "locked" and ".lock" in err.value.fix
    lock.release()
    assert not os.path.exists(lock.path)
    # a lock left by a dead process on this host is removed with a note
    with open(lock.path, "w") as handle:
        json.dump({"pid": 2 ** 22 + 12345, "host": socket.gethostname(), "started": "then"}, handle)
    lock2 = runner.Lock(str(tmp_path))
    lock2.acquire()
    assert lock2.note and "no longer runs" in lock2.note
    lock2.release()
    # a lock from another host without heartbeat for a long time is stale too
    with open(lock.path, "w") as handle:
        json.dump({"pid": 1, "host": "elsewhere", "started": "then"}, handle)
    old = os.stat(lock.path).st_mtime - runner.STALE_LOCK_S - 60
    os.utime(lock.path, (old, old))
    lock3 = runner.Lock(str(tmp_path))
    lock3.acquire()
    assert "no sign of life" in lock3.note
    lock3.release()


def test_lock_being_written_is_not_stale(tmp_path):
    """Review F4: a lock file that another run has just created (empty, not yet written) used to be
    judged stale and removed, so both runs went ahead."""
    lock = runner.Lock(str(tmp_path))
    with open(lock.path, "w"):
        pass                                                  # created, not written yet
    with pytest.raises(runner.RunError) as err:
        runner.Lock(str(tmp_path)).acquire()
    assert err.value.code == "locked" and "starting" in err.value.what
    assert "Do not delete" in err.value.fix and "expires 5 minutes" in err.value.fix
    old = os.stat(lock.path).st_mtime - 60                    # an unreadable lock nobody writes: stale
    os.utime(lock.path, (old, old))
    lock2 = runner.Lock(str(tmp_path))
    lock2.acquire()
    assert "unreadable" in lock2.note
    lock2.release()


def test_stale_lock_removal_does_not_remove_a_fresh_lock(tmp_path):
    """Review F4: between judging a lock stale and removing it, another run may have replaced it by its
    own fresh lock; that one must survive."""
    lock = runner.Lock(str(tmp_path))
    with open(lock.path, "w") as handle:
        json.dump({"pid": 2 ** 22 + 12345, "host": socket.gethostname(), "started": "then"}, handle)
    stale_info = lock.read()
    assert lock._stale(stale_info)[0]
    fresh = {"pid": 4242, "host": "another-node", "started": "now"}
    with open(lock.path, "w") as handle:               # the other run won the race
        json.dump(fresh, handle)
    assert lock._remove_stale(stale_info) is False
    assert {k: lock.read()[k] for k in fresh} == fresh   # put back
    assert [n for n in os.listdir(tmp_path)] == [runner.LOCK_FILE]   # nothing left aside
    with open(lock.path, "w") as handle:                 # the stale lock itself is removed
        json.dump({"pid": 2 ** 22 + 12345, "host": socket.gethostname(), "started": "then"}, handle)
    assert lock._remove_stale(lock.read()) is True and not os.listdir(tmp_path)
    assert lock._remove_stale(stale_info) is False       # already gone: nothing to do


def test_duration_format():
    assert runner.duration(0.25) == "0.2 s" or runner.duration(0.25) == "0.3 s"
    assert runner.duration(42) == "42 s"
    assert runner.duration(720) == "12 min"
    assert runner.duration(3 * 3600 + 5 * 60) == "3 h 05 min"
