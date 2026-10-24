"""End to end: `valdo run` on the toy screen (needs the full VALDO environment: torch, gemmi,
reciprocalspaceship, scikit-learn).

The tests share one session-scoped run (conftest.toy_run) and run in file order. Tests that
change the project or the work folder put it back; the check that every stage gives the same
result when rerun with plain upstream (`valdo.pipeline STAGE CONFIG --force`) comes last.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import needs_torch, run_valdo, set_yaml_value, valdo_env

pytestmark = needs_torch

STAGES_RUN = ["standardize", "reindex", "scale", "filter", "preprocess", "train", "reconstruct", "rescale",
              "add_phases_and_blobs", "tag_blobs", "report"]
UPSTREAM = STAGES_RUN[:-1]


def workdir_of(toy_run):
    return os.path.join(os.path.dirname(toy_run[0]), "valdo_run")


def events(objs, name):
    return [o for o in objs if o.get("event") == name]


def rows_by_name(wd):
    from valdo.easy.ids import read_datasets_csv
    return {r["original_name"]: r for r in read_datasets_csv(os.path.join(wd, "datasets.csv"))}


def plan_of(folder, env=None):
    proc, objs = run_valdo(["run", "--dry-run", "--json"], cwd=folder, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return {p["stage"]: p for p in objs[-1]["plan"]}


def load_state(wd):
    with open(os.path.join(wd, "valdo_state.json")) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The first run
# ---------------------------------------------------------------------------

def test_run_succeeds(toy_run):
    from valdo.easy import SCHEMA_ID
    path, proc, objs = toy_run
    assert proc.returncode == 0, proc.stderr[-5000:]
    final = objs[-1]
    assert final["schema"] == SCHEMA_ID and final["command"] == "run" and final["ok"] is True
    assert [e["stage"] for e in events(objs, "stage_done")] == STAGES_RUN      # refine does not apply
    assert [e["stage"] for e in events(objs, "stage_start")] == STAGES_RUN
    assert not events(objs, "stage_failed")
    for e in events(objs, "stage_start"):
        assert os.path.isfile(e["log"]) and e["reason"] == "not run yet"
    assert final["datasets"] == {"total": 25, "ok": 23, "dropped": 2}     # broken file; uncorrelated data
    assert final["report"] and os.path.isfile(final["report"]) and final["next"] == [f"open {final['report']}"]
    wd = workdir_of(toy_run)
    state = load_state(wd)
    assert state["stages"]["refine"]["status"] == "skipped"
    assert all(state["stages"][s]["status"] == "done" for s in STAGES_RUN)
    assert state["last_run"]["ok"] is True and state["inputs"]["ops_hkl"] == ["h,k,l", "-h,-k,l"]
    assert not os.path.exists(os.path.join(wd, ".lock"))


def test_datasets_csv_tracks_every_input(toy_run, toy_screen):
    _, truth = toy_screen
    rows = rows_by_name(workdir_of(toy_run))
    assert len(rows) == 25
    for name in truth["broken"]:
        assert rows[name]["status"] == "dropped" and rows[name]["dropped_stage"] == "input"
        assert rows[name]["dropped_reason"]
    for name in truth["bad"]:
        r = rows[name]
        assert (r["status"], r["dropped_stage"]) == ("dropped", "filter") and "min_cc" in r["dropped_reason"], r
    for d in truth["datasets"]:
        r = rows[d["name"]]
        if d["bad"]:
            continue
        assert r["status"] == "ok", r
        assert r["phases_source"] and r["phase_file"] and r["space_group"] == "P 3 2 1"
        if d["flipped"]:     # brought back into the reference's indexing
            assert (r["variant"], r["reindex_op"]) == (f"{r['id']}_1", "-h,-k,l"), r
            assert "reindexed" in r["flags"].split(";")
        else:
            assert (r["variant"], r["reindex_op"]) == (f"{r['id']}_0", "h,k,l"), r
        assert bool(r["bound_model"]) == d["bound"]
    assert "reference" in rows[truth["datasets"][0]["name"]]["flags"].split(";")


def test_stage_configs_are_plain_upstream_configs(toy_run):
    import yaml
    from valdo.commandline.config import validate_config
    wd = workdir_of(toy_run)
    for stage in UPSTREAM:
        path = os.path.join(wd, "configs", f"{stage}.yaml")
        text = open(path).read()
        assert f"valdo.pipeline {stage} {path} --force" in text.splitlines()[1]
        body = [l for l in text.splitlines() if l and not l.startswith("#")]
        assert not body[0].startswith("{"), f"{stage}: block-style YAML expected"
        cfg = yaml.safe_load(text)
        validate_config(stage, dict(cfg))
        for key, value in cfg.items():
            if isinstance(value, str) and ("/" in value) and not key.endswith("pattern"):
                assert os.path.isabs(value), (stage, key, value)
            if isinstance(value, str) and (key.endswith("folder") or key.endswith("_dir") or key == "destination_path"
                                           or key == "source_path" or key == "phasing_path"):
                assert value.endswith("/"), (stage, key, value)


def test_models_of_reindexed_datasets_are_in_the_reference_frame(toy_run, toy_screen):
    """inputs/models/<variant>.pdb and inputs/bound_models/<id>.pdb of a flipped dataset are
    transformed back: they superpose on the apo model / the true ligand position."""
    import gemmi
    folder, truth = toy_screen
    wd = workdir_of(toy_run)
    rows = rows_by_name(wd)
    apo = gemmi.read_structure(os.path.join(folder, "apo.pdb"))
    apo_ca = {r.seqid.num: r["CA"][0].pos for r in apo[0]["A"]}
    for name in truth["flipped"]:
        r = rows[name]
        st = gemmi.read_structure(os.path.join(wd, "inputs", "models", f"{r['variant']}.pdb"))
        dists = [res["CA"][0].pos.dist(apo_ca[res.seqid.num]) for res in st[0]["A"] if res.name != "LIG"]
        assert max(dists) < 0.5, (name, max(dists))
        # the model the user gave is in the flipped frame: far from the apo model
        own = gemmi.read_structure(r["model"])
        assert max(res["CA"][0].pos.dist(apo_ca[res.seqid.num]) for res in own[0]["A"]) > 2.0
        if name in truth["ligands"]:
            bound = gemmi.read_structure(os.path.join(wd, "inputs", "bound_models", f"{r['id']}.pdb"))
            lig = [a.pos for res in bound[0]["A"] if res.name == "LIG" for a in res]
            center = gemmi.Position(*np.mean([[p.x, p.y, p.z] for p in lig], axis=0))
            assert center.dist(gemmi.Position(*truth["ligands"][name]["center"])) < 0.5


def test_model_map_check(toy_run, tmp_path):
    """Found on PTP1B: the apo fallback model sat at another origin than the models the data
    files' phases came from (mean 2mFo-DFc density at its atoms ~0 sigma instead of 1.6-3.8), so
    focal tagging silently used wrong coordinates for those datasets. add_phases_and_blobs now
    measures the fit and flags model_map_mismatch. Here every toy model fits (no flag), and the
    same model moved by half a cell along c does not."""
    import gemmi
    from valdo.easy.stages import MODEL_FIT_MIN, model_map_fit
    wd = workdir_of(toy_run)
    fit = load_state(wd)["stages"]["add_phases_and_blobs"]["outputs"]["model_fit"]
    ok = [r for r in rows_by_name(wd).values() if r["status"] == "ok"]
    assert ok and set(fit) == {r["variant"] for r in ok}
    assert min(fit.values()) > 2 * MODEL_FIT_MIN, fit
    assert not any("model_map_mismatch" in r["flags"] for r in ok)
    stem = sorted(fit)[0]
    phases = os.path.join(wd, "inputs", "phases", f"{stem}.mtz")
    st = gemmi.read_structure(os.path.join(wd, "inputs", "models", f"{stem}.pdb"))
    shift = gemmi.Position(0, 0, st.cell.c / 2)
    for chain in st[0]:
        for res in chain:
            for atom in res:
                atom.pos = atom.pos + shift
    moved = str(tmp_path / "moved.pdb")
    st.write_pdb(moved)
    assert model_map_fit(phases, "2FOFCWT", "PHI2FOFCWT", moved) < MODEL_FIT_MIN


def test_report_finds_the_ligands(toy_run, toy_screen):
    _, truth = toy_screen
    wd = workdir_of(toy_run)
    with open(os.path.join(wd, "report", "report.json")) as handle:
        rep = json.load(handle)
    assert os.path.isfile(os.path.join(wd, "report", "report.html"))
    assert rep["counts"]["datasets_ok"] == 23 and rep["counts"]["blobs_kept"] > 0 and rep["top_hits"]
    v = rep["validation"]
    assert v["n_bound_models"] == len(truth["bound"]) and v["n_analysed"] == len(truth["bound"])
    per = {d["original_name"]: d for d in v["per_dataset"]}
    # every ligand is found, as the strongest blob of its dataset (the reindexed dataset too: its
    # phases and models were transformed back into the reference setting)
    assert all(per[n]["first_ligand_blob_rank_in_dataset"] == 1 for n in truth["bound"]), per
    assert any(n in truth["flipped"] for n in truth["bound"])
    assert all(per[n]["closest_blob_distance_A"] < 3.0 for n in truth["bound"]), per
    # and the ligand blobs head the hit list (apo datasets only have noise)
    assert {h["original_name"] for h in rep["top_hits"][:len(truth["bound"])]} <= set(truth["bound"])
    assert v["auc_score"] is not None and v["auc_score"] > 0.9
    # funnel: the broken file is lost at the input checks (first), filter drops the bad one,
    # and refine (not applicable here) is not shown
    funnel = [(f["stage"], f["count"], f["dropped"]) for f in rep["funnel"]]
    n = len(truth["datasets"]) + len(truth["broken"])
    assert funnel[:2] == [("input", n, 0), ("input_checks", n - len(truth["broken"]), len(truth["broken"]))]
    assert "refine" not in [f[0] for f in funnel] and ("filter", 23, 1) in funnel
    assert [f[1] for f in funnel[:-1]] == sorted([f[1] for f in funnel[:-1]], reverse=True)


# ---------------------------------------------------------------------------
# Resuming, status, changes
# ---------------------------------------------------------------------------

def test_second_run_does_nothing(toy_run):
    folder = os.path.dirname(toy_run[0])
    t0 = time.time()
    proc, objs = run_valdo(["run", "--json"], cwd=folder)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert [e["stage"] for e in events(objs, "stage_skipped")] == STAGES_RUN
    assert not events(objs, "stage_start")
    assert time.time() - t0 < 120


def test_status_after_the_run(toy_run):
    folder = os.path.dirname(toy_run[0])
    proc, objs = run_valdo(["status", "--json"], cwd=folder)
    assert proc.returncode == 0
    s = objs[-1]
    by = {st["stage"]: st for st in s["stages"]}
    assert not by["refine"]["applicable"]
    for name in STAGES_RUN:
        assert by[name]["status"] == "done" and not by[name]["stale"] and by[name]["next_action"] == "skip", by[name]
    assert s["datasets"] == {"total": 25, "ok": 23, "dropped": 2, "dropped_by_stage": {"input": 1, "filter": 1}}
    assert s["lock"] is None and s["last_run"]["ok"] is True
    assert len(s["next"]) == 1 and s["next"][0].startswith("open ") and s["next"][0].endswith("report/report.html")


def test_more_cpus_rerun_nothing(toy_run):
    env = valdo_env()
    env["VALDO_NCPU"] = "3"
    plan = plan_of(os.path.dirname(toy_run[0]), env=env)
    assert all(plan[s]["action"] == "skip" for s in STAGES_RUN), plan


def test_changed_setting_reruns_from_the_stage_that_uses_it(toy_run):
    path = toy_run[0]
    folder = os.path.dirname(path)
    set_yaml_value(path, "radius", 6.0)          # focal_residue.radius -> tag_blobs.focal_radius
    try:
        plan = plan_of(folder)
        assert all(plan[s]["action"] == "skip" for s in STAGES_RUN[:-2]), plan
        assert plan["tag_blobs"]["action"] == "run" and "changed" in plan["tag_blobs"]["reason"]
        assert (plan["report"]["action"], plan["report"]["reason"]) == ("run", "an earlier stage reruns")
    finally:
        set_yaml_value(path, "radius", 5.0)
    plan = plan_of(folder)
    assert all(plan[s]["action"] == "skip" for s in STAGES_RUN), plan


def test_changed_map_columns_rerun_the_blob_stage(toy_run):
    """Review F3: phases.columns.fwt (and delfwt) are copied into the prepared phase files and used by
    the model-map check and the report's context maps, so changing them must rerun
    add_phases_and_blobs and the stages after it (and nothing before)."""
    path = toy_run[0]
    folder = os.path.dirname(path)
    set_yaml_value(path, "fwt", '"F-obs"')            # any other column: only the plan is looked at
    try:
        plan = plan_of(folder)
        assert all(plan[s]["action"] == "skip" for s in STAGES_RUN[:-3]), plan
        assert plan["add_phases_and_blobs"]["action"] == "run" and "changed" in plan["add_phases_and_blobs"]["reason"]
        assert plan["tag_blobs"]["action"] == "run" and plan["report"]["action"] == "run"
    finally:
        set_yaml_value(path, "fwt", '"2FOFCWT"')
    plan = plan_of(folder)
    assert all(plan[s]["action"] == "skip" for s in STAGES_RUN), plan


def test_from_reruns_the_selected_stages(toy_run):
    folder = os.path.dirname(toy_run[0])
    proc, objs = run_valdo(["run", "--from", "tag_blobs", "--json"], cwd=folder)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert [e["stage"] for e in events(objs, "stage_start")] == ["tag_blobs", "report"]
    assert all(e["reason"] == "forced" for e in events(objs, "stage_start"))
    plan = plan_of(folder)
    assert all(plan[s]["action"] == "skip" for s in STAGES_RUN), plan


def test_a_running_valdo_blocks_a_second_one(toy_run):
    folder = os.path.dirname(toy_run[0])
    lock = os.path.join(workdir_of(toy_run), ".lock")
    with open(lock, "w") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": "now"}, handle)
    try:
        proc, objs = run_valdo(["run", "--json"], cwd=folder)
        assert proc.returncode == 1 and objs[-1]["error"]["code"] == "locked"
        assert objs[-1]["next"] == ["valdo status"]          # watch the other run; do not start one
        assert "Do not delete" in objs[-1]["error"]["fix"]
        proc, objs = run_valdo(["status", "--json"], cwd=folder)
        assert objs[-1]["lock"]["pid"] == os.getpid() and objs[-1]["lock"]["stale"] is False
        assert objs[-1]["next"] == ["valdo status"] and objs[-1]["wait"]["poll"] == "valdo status"
    finally:
        os.unlink(lock)


def test_failure_is_explained_and_the_next_run_resumes(toy_run):
    """A folder valdo did not create is never emptied: the stage fails with a clear message,
    later stages are marked to rerun, and once the folder is fixed `valdo run` carries on."""
    folder = os.path.dirname(toy_run[0])
    wd = workdir_of(toy_run)
    marker = os.path.join(wd, "09_rescale", ".valdo-managed")
    os.rename(marker, marker + ".moved")
    try:
        proc, objs = run_valdo(["run", "--from", "rescale", "--json"], cwd=folder)
        assert proc.returncode == 1
        err = objs[-1]["error"]
        assert err["code"] == "foreign_folder" and "09_rescale" in err["message"] and err["fix"]
        state = load_state(wd)
        assert state["stages"]["rescale"]["status"] == "failed"
        assert all(state["stages"][s]["status"] == "pending" for s in ("add_phases_and_blobs", "tag_blobs", "report"))
        assert len(os.listdir(os.path.join(wd, "09_rescale"))) > 1        # nothing was deleted
        proc, objs = run_valdo(["status", "--json"], cwd=folder)
        assert {st["stage"]: st for st in objs[-1]["stages"]}["rescale"]["error"]
    finally:
        os.rename(marker + ".moved", marker)
    proc, objs = run_valdo(["run", "--json"], cwd=folder)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert [e["stage"] for e in events(objs, "stage_start")] == STAGES_RUN[7:]
    assert events(objs, "stage_start")[0]["reason"] == "failed last time"


def test_killed_run_is_picked_up_again(toy_run):
    """As when a batch job hits its time limit: SIGKILL in the middle of a stage."""
    folder = os.path.dirname(toy_run[0])
    wd = workdir_of(toy_run)
    proc = subprocess.Popen([sys.executable, "-m", "valdo.easy.cli", "run", "--from", "train", "--json"], cwd=folder,
                            env=valdo_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    killed = False
    while time.time() < deadline and proc.poll() is None:
        try:
            st = load_state(wd)["stages"]
        except (OSError, ValueError):
            st = {}
        if (st.get("train") or {}).get("status") == "running":
            proc.send_signal(signal.SIGKILL)
            killed = True
            break
        time.sleep(0.02)
    proc.wait(timeout=60)
    if not killed:
        pytest.skip("the run finished before it could be killed")
    assert os.path.exists(os.path.join(wd, ".lock"))                  # a killed run cannot clean up
    st = load_state(wd)["stages"]
    first = next(s for s in STAGES_RUN[5:] if st[s]["status"] != "done")   # normally train itself
    run, objs = run_valdo(["run", "--json"], cwd=folder)
    assert run.returncode == 0, run.stderr[-3000:]
    assert "no longer runs" in run.stderr                               # the stale lock is removed
    assert [e["stage"] for e in events(objs, "stage_start")] == STAGES_RUN[STAGES_RUN.index(first):]
    if st[first]["status"] == "running":
        assert events(objs, "stage_start")[0]["reason"] == "was interrupted"
    assert not os.path.exists(os.path.join(wd, ".lock"))


def test_report_command(toy_run):
    folder = os.path.dirname(toy_run[0])
    proc, objs = run_valdo(["report", "--top", "3", "--no-maps", "--json"], cwd=folder)
    assert proc.returncode == 0, proc.stderr[-3000:]
    res = objs[-1]
    assert res["ok"] and os.path.isfile(os.path.join(res["report_dir"], "report.html"))
    assert len(res["result"]["top_hits"]) <= 3
    # valdo.yaml stays the reference: the next run rebuilds the report with its settings
    plan = plan_of(folder)
    assert plan["report"]["action"] == "run" and all(plan[s]["action"] == "skip" for s in STAGES_RUN[:-1])
    proc, _ = run_valdo(["run", "--json"], cwd=folder)
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# Same results as plain upstream (last: it moves stage folders around)
# ---------------------------------------------------------------------------

def _tree(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f != ".valdo-managed":
                out.append(os.path.relpath(os.path.join(root, f), folder))
    return sorted(out)


class Diff:
    def __init__(self):
        self.max_rel = 0.0
        self.problems = []

    def numbers(self, a, b, where, phase=False):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            self.problems.append(f"{where}: shape {a.shape} != {b.shape}")
            return
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            self.problems.append(f"{where}: NaNs differ")
            return
        inf = np.isinf(a)
        if not np.array_equal(inf, np.isinf(b)) or not np.array_equal(a[inf], b[inf]):
            self.problems.append(f"{where}: infinities differ")
            return
        ok = np.isfinite(a)
        if not ok.any():
            return
        if phase:
            d = np.abs((a[ok] - b[ok] + 180.0) % 360.0 - 180.0) / 180.0
        else:
            scale = max(float(np.abs(a[ok]).max()), 1e-12)
            d = np.abs(a[ok] - b[ok]) / scale
        self.max_rel = max(self.max_rel, float(d.max()))

    def objects(self, a, b, where):
        import pandas as pd
        try:
            import torch
            if isinstance(a, torch.Tensor):
                a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        except ImportError:
            pass
        if isinstance(a, pd.DataFrame):
            if list(a.columns) != list(b.columns) or len(a) != len(b):
                self.problems.append(f"{where}: columns/rows differ")
                return
            self.objects(a.index.to_numpy(), b.index.to_numpy(), f"{where}.index")
            for c in a.columns:
                self.objects(a[c].to_numpy(), b[c].to_numpy(), f"{where}[{c}]")
        elif isinstance(a, pd.Series):
            self.objects(a.to_numpy(), b.to_numpy(), where)
        elif isinstance(a, np.ndarray) and a.dtype.kind in "fiub" and getattr(b, "dtype", None) is not None:
            self.numbers(a, b, where)
        elif isinstance(a, (np.ndarray, list, tuple)):
            if len(a) != len(b):
                self.problems.append(f"{where}: length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self.objects(x, y, f"{where}[{i}]")
        elif isinstance(a, dict):
            if set(a) != set(b):
                self.problems.append(f"{where}: keys differ")
                return
            for k in a:
                self.objects(a[k], b[k], f"{where}.{k}")
        elif isinstance(a, (float, np.floating)):
            self.numbers([a], [b], where)
        elif a != b:
            self.problems.append(f"{where}: {a!r} != {b!r}")

    def files(self, a, b, rel):
        import pandas as pd
        ext = os.path.splitext(rel)[1]
        if ext == ".mtz":
            import reciprocalspaceship as rs
            x, y = rs.read_mtz(a), rs.read_mtz(b)
            if list(x.columns) != list(y.columns) or not x.index.equals(y.index):
                self.problems.append(f"{rel}: columns or reflections differ")
                return
            for c in x.columns:
                self.numbers(x[c].to_numpy(dtype=float), y[c].to_numpy(dtype=float), f"{rel}[{c}]",
                             phase=str(x[c].dtype) == "Phase")
        elif ext == ".npy":
            self.numbers(np.load(a), np.load(b), rel)
        elif ext == ".pkl":
            self.objects(pd.read_pickle(a), pd.read_pickle(b), rel)
        elif ext == ".png":
            pass
        elif open(a, "rb").read() != open(b, "rb").read():
            self.problems.append(f"{rel}: contents differ")


def pipeline_command():
    """The upstream command named in the config headers (`valdo.pipeline`), else python -m."""
    script = os.path.join(os.path.dirname(sys.executable), "valdo.pipeline")
    return [script] if os.access(script, os.X_OK) else [sys.executable, "-m", "valdo.commandline.pipeline"]


# largest relative difference accepted: exact for the deterministic stages; the VAE stages run
# with the same seed and thread count, so they should be (nearly) identical too
TOLERANCE = {"train": 1e-4, "reconstruct": 1e-4, "rescale": 1e-4, "add_phases_and_blobs": 1e-4, "tag_blobs": 1e-4}


def test_every_stage_matches_plain_upstream(toy_run):
    """Each stage folder is moved aside, the stage is rerun with plain upstream
    (`valdo.pipeline STAGE configs/STAGE.yaml --force`, from another folder) and the outputs are
    compared; then the original folder is put back."""
    from valdo.easy.project import load_project
    from valdo.easy.stages import BY_NAME
    wd = workdir_of(toy_run)
    env = valdo_env()
    ncpu = str(load_project(toy_run[0])[0]["compute"]["ncpu"])       # the thread count valdo run used
    env.update(OMP_NUM_THREADS=ncpu, MKL_NUM_THREADS=ncpu, OPENBLAS_NUM_THREADS=ncpu)
    report = {}
    for stage in UPSTREAM:
        folder = os.path.join(wd, BY_NAME[stage].folder)
        aside = folder + ".valdo-easy"
        os.rename(folder, aside)
        try:
            for root, _, _ in os.walk(aside):                # the same (empty) folder layout
                os.makedirs(os.path.join(folder, os.path.relpath(root, aside)), exist_ok=True)
            cmd = pipeline_command() + [stage, os.path.join(wd, "configs", f"{stage}.yaml"), "--force"]
            proc = subprocess.run(cmd, cwd=os.path.dirname(wd), env=env, capture_output=True, text=True,
                                  timeout=900)
            assert proc.returncode == 0, (stage, proc.stderr[-3000:])
            mine, theirs = _tree(aside), _tree(folder)
            assert mine == theirs, (stage, sorted(set(mine) ^ set(theirs)))
            diff = Diff()
            for rel in mine:
                diff.files(os.path.join(aside, rel), os.path.join(folder, rel), rel)
            assert not diff.problems, (stage, diff.problems[:10])
            report[stage] = diff.max_rel
            assert diff.max_rel <= TOLERANCE.get(stage, 1e-9), (stage, diff.max_rel)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            os.rename(aside, folder)
    print("largest relative differences:", report)
    with open(os.path.join(os.path.dirname(wd), "upstream_equivalence.json"), "w") as handle:
        json.dump(report, handle, indent=1)      # kept with the toy screen, for the record
    plan = plan_of(os.path.dirname(toy_run[0]))
    assert all(plan[s]["action"] == "skip" for s in STAGES_RUN), plan
