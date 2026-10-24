"""Running the stages: state file, fingerprints, resume, cleaning, lock, logs, progress.

`Runner(project, inputs, console).run(...)` runs the stages of stages.STAGES in order and
records everything in <workdir>/valdo_state.json; `plan(...)` says what a run would do without
writing anything (used by `--dry-run` and `valdo status`).

- A stage runs if it is not done, if its fingerprint changed, or if it is forced. The
  fingerprint is a SHA-256 of the generated config (minus ncpu), stage-specific extras
  (input files, models, phase files, ...) and the token of the previous applicable stage.
  Every completed stage gets a fresh random token, so a rerun invalidates everything after it.
- Before a stage runs, its folder (and the input folders it owns) is emptied. Only folders
  that carry the `.valdo-managed` marker (or are empty) are touched, entries are removed
  without following symlinks, and nothing outside the workdir is ever removed.
- `.lock` (O_EXCL, heartbeat every 30 s) keeps two runs out of one workdir.
- stdout/stderr of a stage, including worker processes and C code, go to logs/<stage>.log.
"""

import contextlib
import datetime
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import sys
import threading
import time
import traceback
import uuid

from valdo.easy import SCHEMA_ID, UPSTREAM_BASE, hints
from valdo.easy.ids import join_flags, split_flags, write_datasets_csv
from valdo.easy.stages import BY_NAME, NAMES, STAGES, Result, StageError

STATE_FILE = "valdo_state.json"
MARKER = ".valdo-managed"
LOCK_FILE = ".lock"
HEARTBEAT_S = 30
STALE_LOCK_S = 5 * 60


class RunError(Exception):
    """A run stopped. code: short id; exit_code 1 (user/data/stage problem) or 2 (bug)."""

    def __init__(self, code, message, what="", cause="", fix="", log=None, stage=None, exit_code=1):
        super().__init__(message)
        self.code, self.message, self.what, self.cause, self.fix = code, message, what or message, cause, fix
        self.log, self.stage, self.exit_code = log, stage, exit_code

    def as_dict(self):
        return {"code": self.code, "message": self.message, "what": self.what, "cause": self.cause,
                "fix": self.fix, "log": self.log, "stage": self.stage}


class Interrupted(BaseException):
    """SIGTERM (e.g. the batch system stopping the job)."""


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def duration(seconds):
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f} s" if seconds >= 1 else f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    h, m = divmod(int(round(seconds / 60)), 60)
    return f"{h} h {m:02d} min"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_state(project):
    from valdo import __version__
    return {"schema": SCHEMA_ID, "valdo_version": __version__, "upstream_base": UPSTREAM_BASE,
            "project_file": project.get("_path"), "workdir": project["workdir"], "created": now(),
            "updated": now(), "inputs": {}, "stages": {}, "last_run": None}


def load_state(workdir):
    """The state dict, or None if there is none. A corrupt file is moved aside."""
    path = os.path.join(workdir, STATE_FILE)
    try:
        with open(path) as handle:
            state = json.load(handle)
        if isinstance(state, dict) and isinstance(state.get("stages"), dict):
            return state
        raise ValueError("unexpected content")
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        aside = path + f".corrupt-{int(time.time())}"
        try:
            os.replace(path, aside)
        except OSError:
            pass
        return None


def save_state(workdir, state):
    from valdo import __version__
    state["updated"] = now()
    state["valdo_version"] = __version__
    path = os.path.join(workdir, STATE_FILE)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=1, default=str)
    os.replace(tmp, path)


def merged_rows(base_rows, state, before=None):
    """datasets.csv rows: the input scan plus the contributions of every finished stage
    (only the stages that come before stage `before`, if given)."""
    rows = [dict(r) for r in base_rows]
    by_id = {r["id"]: r for r in rows}
    for name in NAMES:
        if name == before:
            break
        st = (state or {}).get("stages", {}).get(name) or {}
        if st.get("status") != "done":
            continue
        contrib = st.get("contrib") or {}
        for ident, flags in (contrib.get("flags") or {}).items():
            row = by_id.get(ident)
            if row is not None:
                row["flags"] = join_flags(split_flags(row["flags"]) + list(flags))
        for ident, values in (contrib.get("values") or {}).items():
            row = by_id.get(ident)
            if row is not None:
                row.update({k: "" if v is None else str(v) for k, v in values.items()})
        for ident, reason in (contrib.get("dropped") or {}).items():
            row = by_id.get(ident)
            if row is not None and row["status"] == "ok":
                row.update(status="dropped", dropped_stage=name, dropped_reason=reason)
    return rows


def fingerprint(stage, cfg, extras, prev_token):
    config = {k: v for k, v in (cfg or {}).items() if k != "ncpu"}
    blob = json.dumps({"schema": SCHEMA_ID, "stage": stage.name, "config": config, "extras": extras,
                       "previous": prev_token}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Console, output capture, lock
# ---------------------------------------------------------------------------

class Console:
    """Human text and JSON Lines. Writes to duplicates of the original stdout/stderr, so
    messages still reach the terminal while a stage's output is redirected to its log."""

    def __init__(self, json_mode=False):
        self.json = json_mode
        self.out = self._grab(sys.stdout)
        self.err = self._grab(sys.stderr)
        self.run_log = None

    @staticmethod
    def _grab(stream):
        try:
            stream.flush()
            fd = os.dup(stream.fileno())
            return io.open(fd, "w", buffering=1, encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            return stream

    def say(self, text=""):
        stream = self.err if self.json else self.out
        stream.write(text + "\n")
        stream.flush()
        self._to_run_log(text)

    def warn(self, text):
        self.err.write(text + "\n")
        self.err.flush()
        self._to_run_log(text)

    def event(self, **obj):
        if self.json:
            self.out.write(json.dumps(obj, default=str) + "\n")
            self.out.flush()

    def _to_run_log(self, text):
        if self.run_log:
            try:
                with open(self.run_log, "a") as handle:
                    handle.write(f"{text}\n")
            except OSError:
                pass

    def close(self):
        for stream in (self.out, self.err):
            if stream not in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                try:
                    stream.close()
                except Exception:
                    pass


@contextlib.contextmanager
def capture_output(log_path, echo=None):
    """Send fd 1/2 (and sys.stdout/err) to `log_path`; with `echo` (a text stream) also copy there."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    saved = (os.dup(1), os.dup(2))
    log = open(log_path, "ab", buffering=0)
    reader = None
    if echo is not None:
        r, w = os.pipe()

        def pump():
            with os.fdopen(r, "rb", buffering=0) as src:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    try:
                        echo.write(chunk.decode("utf-8", "replace"))
                        echo.flush()
                    except Exception:
                        pass

        reader = threading.Thread(target=pump, name="valdo-log-tee", daemon=True)
        reader.start()
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
    else:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    old = (sys.stdout, sys.stderr)
    sys.stdout = io.TextIOWrapper(io.FileIO(1, "w", closefd=False), encoding="utf-8", errors="replace",
                                  line_buffering=True)
    sys.stderr = io.TextIOWrapper(io.FileIO(2, "w", closefd=False), encoding="utf-8", errors="replace",
                                  line_buffering=True)
    try:
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        sys.stdout, sys.stderr = old
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        if reader is not None:
            reader.join(timeout=10)
        log.close()


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError, OverflowError):
        return True
    return True


class Lock:
    def __init__(self, workdir):
        self.path = os.path.join(workdir, LOCK_FILE)
        self._stop = threading.Event()
        self._thread = None
        self.note = None

    def read(self):
        """The lock's content plus its age (seconds since the last heartbeat); None if there is no
        lock; {"unreadable": True, "age_s": ...} if it cannot be parsed (e.g. still being written)."""
        try:
            age = time.time() - os.stat(self.path).st_mtime
        except OSError:
            return None
        try:
            with open(self.path) as handle:
                info = json.load(handle)
            if not isinstance(info, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError):
            return {"unreadable": True, "age_s": age}
        info["age_s"] = age
        return info

    def _stale(self, info):
        if info is None:
            return True, "the lock file disappeared"
        if info.get("unreadable"):   # a lock that another run is writing right now is young
            return (True, "unreadable lock file") if info.get("age_s", 0) > 10 else (False, "")
        if info.get("host") == socket.gethostname() and not _pid_alive(info.get("pid")):
            return True, f"process {info.get('pid')} on this computer no longer runs"
        if info.get("age_s", 0) > STALE_LOCK_S:
            return True, f"no sign of life from {info.get('host')} for {duration(info['age_s'])}"
        return False, ""

    def acquire(self):
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "started": now(), "argv": sys.argv}
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self.read()
                stale, why = self._stale(info)
                if stale:
                    if self._remove_stale(info):
                        self.note = f"removed a stale lock ({why})"
                    continue
                expires = ""
                if info.get("age_s") is not None:
                    at = datetime.datetime.now() + datetime.timedelta(seconds=max(0.0, STALE_LOCK_S - info["age_s"]))
                    expires = f" (at about {at.strftime('%H:%M')})"
                who = ("another valdo run is starting" if info.get("unreadable") else
                       f"pid {info.get('pid')} on {info.get('host')}, started {info.get('started')}")
                raise RunError("locked", f"Another valdo run is using this work directory ({who}).",
                               cause="Two runs in one work directory would overwrite each other's files.",
                               fix=f"Wait for that run to finish ('valdo status' shows progress). If that job was "
                                   f"killed, its lock expires {STALE_LOCK_S // 60} minutes after its last sign of "
                                   f"life{expires}; then run again. Do not delete {self.path} by hand.")
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            self._thread = threading.Thread(target=self._beat, name="valdo-lock", daemon=True)
            self._thread.start()
            return
        raise RunError("locked", f"Could not create {self.path}.",
                       cause="Other valdo runs kept creating it at the same time.",
                       fix="Check with 'valdo status' that no other run is active, then run again.")

    def _remove_stale(self, info):
        """Remove the stale lock described by `info`, safely against other processes doing the same.

        The lock is first renamed to a unique name (atomic: only one process can move a given file).
        If what was moved is not the lock judged stale (another run created a fresh one in between),
        it is put back with link(), which never overwrites. Returns True if the stale lock was removed."""
        aside = f"{self.path}.stale-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            os.rename(self.path, aside)
        except FileNotFoundError:
            return False
        moved = None
        try:
            with open(aside) as handle:
                moved = json.load(handle)
        except (OSError, ValueError):
            pass
        if info is None or info.get("unreadable"):
            same = not isinstance(moved, dict)        # still the same unreadable file
        else:
            same = isinstance(moved, dict) and all(moved.get(k) == info.get(k) for k in ("pid", "host", "started"))
        if not same:
            try:
                os.link(aside, self.path)
            except OSError:
                pass
        try:
            os.unlink(aside)
        except OSError:
            pass
        return same

    def _beat(self):
        while not self._stop.wait(HEARTBEAT_S):
            try:
                os.utime(self.path)
            except OSError:
                pass

    def release(self):
        self._stop.set()
        info = self.read()
        if info and info.get("pid") == os.getpid() and info.get("host") == socket.gethostname():
            try:
                os.unlink(self.path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Context handed to the stage hooks
# ---------------------------------------------------------------------------

class Context:
    """What the stage hooks see. `before` is the stage being planned or run: rows() and alive()
    then reflect the datasets as they enter that stage (later stages' drops are left out), so a
    stage's config and fingerprint do not change when a later stage drops a dataset."""

    def __init__(self, runner):
        self.runner = runner
        self.project = runner.project
        self.inputs = runner.inputs
        self.workdir = runner.workdir
        self.ncpu = runner.ncpu
        self.scratch = {}
        self.before = None

    @property
    def state(self):
        return self.runner.state

    def path(self, rel):
        return os.path.join(self.workdir, rel)

    def dir(self, rel):
        return os.path.join(self.workdir, rel.rstrip("/")) + "/"

    def rows(self):
        return merged_rows(self.inputs.rows, self.state, before=self.before)

    def alive(self):
        return [r for r in self.rows() if r["status"] == "ok"]

    def outputs(self, stage):
        return (self.state["stages"].get(stage) or {}).get("outputs") or {}

    def stage_applicable(self, name):
        return BY_NAME[name].applicable(self)[0]

    @property
    def reindexed(self):
        return bool(self.inputs.reindex and self.inputs.reindex["enabled"])

    @property
    def ops(self):
        return self.inputs.reindex["ops"] if self.reindexed else ["x,y,z"]

    @property
    def ops_hkl(self):
        return self.inputs.reindex["ops_hkl"] if self.reindexed else ["h,k,l"]

    @property
    def reference_id(self):
        return self.inputs.reference["id"] if self.inputs.reference else None


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, project, inputs, console, ncpu=None, verbose=False):
        self.project = project
        self.inputs = inputs
        self.console = console
        self.workdir = project["workdir"]
        self.ncpu = int(ncpu or project["compute"]["ncpu"])
        self.verbose = verbose
        self.state = load_state(self.workdir) or new_state(project)
        self.ctx = Context(self)

    # -- selection ---------------------------------------------------------

    def applicable(self):
        """[(stage, applicable, reason)] for every stage."""
        return [(s,) + tuple(s.applicable(self.ctx)) for s in STAGES]

    def _selection(self, from_=None, until=None, only=None, force=False):
        app = [s for s, ok, _ in self.applicable() if ok]
        names = [s.name for s in app]
        for flag, value in (("--from", from_), ("--until", until), ("--only", only)):
            if value is None:
                continue
            if value not in BY_NAME:
                raise RunError("bad_stage", f"{flag} {value}: unknown stage.",
                               fix=f"Use one of: {', '.join(NAMES)} ('valdo stages' lists them).")
            if value not in names:
                why = BY_NAME[value].applicable(self.ctx)[1]
                raise RunError("stage_not_applicable", f"{flag} {value}: this stage does not run for this project ({why}).",
                               fix="'valdo status' shows the stages that apply.")
        if only and (from_ or until):
            raise RunError("bad_options", "--only cannot be combined with --from or --until.",
                           fix="Use --from S --until T for a range.")
        if only:
            from_, until = only, only
        start = names.index(from_) if from_ else 0
        stop = names.index(until) if until else len(names) - 1
        if stop < start:
            raise RunError("bad_options", f"--until {until} comes before --from {from_}.")
        selected = names[start:stop + 1]
        forced = set(selected) if (force or from_) else set()
        return app, names, selected, forced, start

    def _prev_token(self, stage_names, name):
        i = stage_names.index(name)
        if i == 0:
            return "start"
        prev = self.state["stages"].get(stage_names[i - 1]) or {}
        return prev.get("token") or "missing"

    def _decide(self, stage, names, forced, earlier_runs):
        """(action, reason, cfg, fp) for one selected stage."""
        st = self.state["stages"].get(stage.name) or {}
        if stage.name in forced:
            return "run", "forced", None, None
        status = st.get("status", "pending")
        if status != "done":
            reason = {"pending": "not run yet", "failed": "failed last time", "interrupted": "was interrupted",
                      "running": "was interrupted", "skipped": "not run yet"}.get(status, status)
            return "run", reason, None, None
        if earlier_runs:
            return "run", "an earlier stage reruns", None, None
        self.ctx.before = stage.name
        try:
            cfg = stage.build(self.ctx)
            fp = fingerprint(stage, cfg, stage.extras(self.ctx), self._prev_token(names, stage.name))
        except Exception as exc:
            return "run", f"cannot verify earlier results ({type(exc).__name__}: {exc})", None, None
        if fp != st.get("fingerprint"):
            return "run", "settings, inputs or earlier results changed", cfg, fp
        return "skip", "up to date", cfg, fp

    def plan(self, from_=None, until=None, only=None, force=False):
        """What `run` would do: [{stage, number, action, reason}] (writes nothing)."""
        app, names, selected, forced, start = self._selection(from_, until, only, force)
        out, earlier_runs = [], False
        for stage, ok, why in self.applicable():
            entry = {"stage": stage.name, "number": stage.number}
            if not ok:
                entry.update(action="not_applicable", reason=why)
            elif stage.name not in selected:
                if names.index(stage.name) < start:
                    st = (self.state["stages"].get(stage.name) or {}).get("status")
                    entry.update(action="keep" if st == "done" else "missing",
                                 reason="done (before --from/--only)" if st == "done" else
                                 "not done: needed before the selected stages")
                else:
                    entry.update(action="later", reason="after --until/--only")
            else:
                action, reason, _, _ = self._decide(stage, names, forced, earlier_runs)
                entry.update(action=action, reason=reason)
                earlier_runs = earlier_runs or action == "run"
            out.append(entry)
        return out

    # -- running -----------------------------------------------------------

    def run(self, from_=None, until=None, only=None, force=False, argv=None):
        app, names, selected, forced, start = self._selection(from_, until, only, force)
        for name in names[:start]:
            if (self.state["stages"].get(name) or {}).get("status") != "done":
                raise RunError("prerequisite_missing", f"Stage {name} has not finished, so the selected stages cannot run.",
                               fix=f"Run 'valdo run --until {name}' first (or plain 'valdo run').")
        os.makedirs(os.path.join(self.workdir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "configs"), exist_ok=True)
        self.console.run_log = os.path.join(self.workdir, "logs", "run.log")
        lock = Lock(self.workdir)
        lock.acquire()
        previous_term = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):
            raise Interrupted()

        in_main = threading.current_thread() is threading.main_thread()
        if in_main:
            signal.signal(signal.SIGTERM, on_term)
        started = now()
        label = "valdo report" if only == "report" else "valdo run"   # `valdo report` runs the report stage
        self.console.say(f"{label}: {self.project.get('_path')} -> {self.workdir}  (started {started}, "
                         f"{self.ncpu} CPU(s))")
        if lock.note:
            self.console.say(f"note: {lock.note}")
        skipped = [(s.name, why) for s, ok, why in self.applicable() if not ok]
        if skipped and only != "report":
            self.console.say(f"{len(names)} of {len(STAGES)} stages apply to this project (not: "
                             + "; ".join(f"{n}, {why}" for n, why in skipped) + ")")
        self.state["last_run"] = {"started": started, "finished": None, "ok": None, "argv": argv or sys.argv}
        results = []
        try:
            self._record_inputs()
            for st in self.state["stages"].values():
                if st.get("status") == "running":
                    st["status"] = "interrupted"
            for stage, ok, why in self.applicable():
                if not ok:
                    if (self.state["stages"].get(stage.name) or {}).get("status") != "skipped":
                        self.state["stages"][stage.name] = {"status": "skipped", "reason": why}
            save_state(self.workdir, self.state)
            self._write_datasets()
            total = len(names)
            earlier_runs = False
            for name in names:
                stage = BY_NAME[name]
                index = names.index(name) + 1
                if name not in selected:
                    if names.index(name) < start:
                        continue
                    break
                action, reason, cfg, fp = self._decide(stage, names, forced, earlier_runs)
                if action == "skip":
                    self.console.say(f"[{index}/{total}] {name} ... up to date")
                    self.console.event(event="stage_skipped", stage=name, index=index, total=total, reason=reason)
                    results.append(self._stage_summary(name))
                    continue
                earlier_runs = True
                self._run_stage(stage, names, index, total, reason)
                results.append(self._stage_summary(name))
            ok = True
        except (KeyboardInterrupt, Interrupted):
            self._mark_interrupted()
            self.state["last_run"].update(finished=now(), ok=False)
            save_state(self.workdir, self.state)
            raise
        except RunError:
            self.state["last_run"].update(finished=now(), ok=False)
            save_state(self.workdir, self.state)
            raise
        finally:
            if in_main:
                signal.signal(signal.SIGTERM, previous_term)
            lock.release()
        self.state["last_run"].update(finished=now(), ok=ok)
        save_state(self.workdir, self.state)
        return results

    def _record_inputs(self):
        inp = self.inputs
        self.state["project_file"] = self.project.get("_path")
        self.state["workdir"] = self.workdir
        self.state["inputs"] = {
            "n_files": len(inp.files), "n_usable": len(inp.ok_rows()),
            "reference": inp.reference, "reindex": bool(self.ctx.reindexed),
            "ops": self.ctx.ops, "ops_hkl": self.ctx.ops_hkl, "column_rename": inp.rename,
            "reindex_why": (inp.reindex or {}).get("why"),
        }

    def _mark_interrupted(self):
        for name, st in self.state["stages"].items():
            if st.get("status") == "running":
                st.update(status="interrupted", finished=now())
                self.console.say(f"{name} was interrupted; 'valdo run' starts it again from the beginning.")
                self.console.event(event="stage_failed", stage=name, error={"code": "interrupted"})

    def _write_datasets(self):
        write_datasets_csv(merged_rows(self.inputs.rows, self.state), os.path.join(self.workdir, "datasets.csv"))

    def _stage_summary(self, name):
        st = self.state["stages"].get(name) or {}
        return {"stage": name, "status": st.get("status"), "seconds": st.get("seconds"), "counts": st.get("counts")}

    def _reset_dir(self, rel):
        path = os.path.join(self.workdir, rel)
        work = os.path.abspath(self.workdir)
        if os.path.commonpath([os.path.abspath(path), work]) != work or os.path.abspath(path) == work:
            raise RunError("unsafe_path", f"Refusing to clean {path}: not inside the work directory.", exit_code=2)
        # never follow a symbolic link inside the work directory (the work directory itself may be one)
        probe = work
        for part in os.path.relpath(os.path.abspath(path), work).split(os.sep):
            probe = os.path.join(probe, part)
            if os.path.islink(probe):
                raise RunError("unsafe_path", f"Refusing to clean {probe}: it is a symbolic link.",
                               cause="valdo only empties real folders inside the work directory, so that it can "
                                     "never delete files somewhere else.",
                               fix=f"Replace the link {probe} by a real folder (move its contents back), or put the "
                                   "whole work directory on the other disk by setting 'workdir' in valdo.yaml.")
        if os.path.lexists(path):
            if not os.path.isdir(path):
                raise RunError("not_a_folder", f"{path} exists and is not a folder.",
                               fix=f"Move {path} away and run again.")
            entries = [e for e in os.listdir(path) if e != MARKER]
            if entries and not os.path.isfile(os.path.join(path, MARKER)):
                raise RunError("foreign_folder", f"{path} exists, is not empty and was not created by valdo.",
                               cause="valdo only deletes folders it created itself.",
                               fix=f"Move {path} away (or set another 'workdir') and run again.")
            for entry in entries:
                full = os.path.join(path, entry)
                if os.path.islink(full) or not os.path.isdir(full):
                    os.unlink(full)
                else:
                    shutil.rmtree(full)
        else:
            os.makedirs(path)
        with open(os.path.join(path, MARKER), "w") as handle:
            handle.write("created by valdo-easy; emptied whenever this stage reruns\n")

    def _ensure_dir(self, rel):
        path = os.path.join(self.workdir, rel)
        os.makedirs(path, exist_ok=True)
        marker = os.path.join(path, MARKER)
        if not os.path.exists(marker):
            with open(marker, "w") as handle:
                handle.write("created by valdo-easy\n")

    def _write_config(self, stage, cfg):
        import yaml
        path = os.path.join(self.workdir, "configs", f"{stage.name}.yaml")
        header = (f"# Generated by valdo-easy for stage '{stage.name}' ({now()}).\n"
                  + (f"# Rerun with plain upstream: valdo.pipeline {stage.name} {path} --force\n"
                     if stage.name in ("standardize", "reindex", "scale", "filter", "preprocess", "train",
                                       "reconstruct", "rescale", "add_phases_and_blobs", "tag_blobs") else
                     f"# Settings passed to {stage.upstream}\n"))
        with open(path, "w") as handle:
            handle.write(header)
            yaml.safe_dump(cfg, handle, sort_keys=False, default_flow_style=False, width=1000)

    def _run_stage(self, stage, names, index, total, reason):
        st_all = self.state["stages"]
        later = names[names.index(stage.name) + 1:]
        for name in later:
            if (st_all.get(name) or {}).get("status") not in (None, "pending", "skipped"):
                st_all[name] = {"status": "pending"}
        log_rel = f"logs/{stage.name}.log"
        log_path = os.path.join(self.workdir, log_rel)
        if os.path.exists(log_path):
            os.replace(log_path, os.path.join(self.workdir, "logs", f"{stage.name}.prev.log"))
        started = now()
        st_all[stage.name] = {"status": "running", "started": started, "log": log_rel, "reason": reason}
        save_state(self.workdir, self.state)
        self._write_datasets()
        why = "" if reason == "not run yet" else f"{reason}; "
        self.console.say(f"[{index}/{total}] {stage.name} ... running ({why}log: {log_path})")
        self.console.event(event="stage_start", stage=stage.name, index=index, total=total, reason=reason,
                           log=log_path)
        self.ctx.scratch = {}
        self.ctx.before = stage.name
        res = Result()
        t0 = time.time()
        phase = "prepare"
        try:
            for rel in ((stage.folder,) if stage.clean else ()) + stage.owns:
                self._reset_dir(rel)
            if not stage.clean:
                self._ensure_dir(stage.folder)
            cfg = stage.build(self.ctx)
            fp = fingerprint(stage, cfg, stage.extras(self.ctx), self._prev_token(names, stage.name))
            echo = self.console.err if (self.verbose and self.console.json) else (self.console.out if self.verbose else None)
            with capture_output(log_path, echo=echo):
                print(f"=== valdo-easy {stage.name} ({stage.upstream}) started {started} on {socket.gethostname()}, "
                      f"{self.ncpu} CPU(s)", flush=True)
                try:
                    phase = "pre"
                    if stage.pre:
                        stage.pre(self.ctx, cfg, res)
                    phase = "config"
                    self._write_config(stage, cfg)
                    phase = "execute"
                    print(f"=== running {stage.upstream} with configs/{stage.name}.yaml", flush=True)
                    stage.execute(self.ctx, cfg, res)
                    phase = "post"
                    print("=== checking the results", flush=True)
                    stage.post(self.ctx, cfg, res)
                except StageError as exc:
                    print(f"=== stage stopped: {exc.what}", flush=True)
                    raise
                except (KeyboardInterrupt, Interrupted):
                    print("=== interrupted", flush=True)
                    raise
                except BaseException:
                    traceback.print_exc()
                    sys.stderr.flush()
                    raise
                print(f"=== done in {duration(time.time() - t0)}", flush=True)
        except StageError as exc:
            self._fail(stage, started, t0, log_path, {"code": "stage_check", "what": exc.what, "cause": exc.cause,
                                                      "fix": exc.fix})
            raise RunError("stage_failed", f"{stage.name}: {exc.what}", what=exc.what, cause=exc.cause, fix=exc.fix,
                           log=log_path, stage=stage.name)
        except RunError as exc:
            self._fail(stage, started, t0, log_path, exc.as_dict())
            exc.stage, exc.log = stage.name, exc.log or log_path
            raise
        except (KeyboardInterrupt, Interrupted):
            raise
        except BaseException as exc:
            text = f"{type(exc).__name__}: {exc}"
            if phase == "execute":
                hint = hints.explain(stage.name, hints.tail(log_path, 60) + "\n" + text)
                err = {"code": hint["id"], "what": hint["what"], "cause": hint["cause"], "fix": hint["fix"],
                       "exception": text}
                self._fail(stage, started, t0, log_path, err)
                raise RunError("stage_failed", f"{stage.name} failed: {text}", what=hint["what"],
                               cause=hint["cause"], fix=hint["fix"], log=log_path, stage=stage.name)
            err = {"code": "internal", "what": f"valdo-easy failed while preparing or checking {stage.name}: {text}",
                   "cause": "probably a bug in valdo-easy", "fix": f"Please report it with {log_path}."}
            if isinstance(exc, (OSError, MemoryError)):
                hint = hints.explain(stage.name, text)
                err.update(code=hint["id"], what=hint["what"], cause=hint["cause"], fix=hint["fix"])
            self._fail(stage, started, t0, log_path, err)
            internal = err["code"] == "internal"
            raise RunError("internal_error" if internal else "stage_failed", err["what"], cause=err["cause"],
                           fix=err["fix"], log=log_path, stage=stage.name, exit_code=2 if internal else 1)
        seconds = time.time() - t0
        n_in = res.n_in if res.n_in is not None else len(self.ctx.alive())
        n_out = res.n_out if res.n_out is not None else n_in - len(res.dropped)
        st_all[stage.name] = {
            "status": "done", "fingerprint": fp, "token": uuid.uuid4().hex, "started": started, "finished": now(),
            "seconds": round(seconds, 1), "counts": {"in": n_in, "out": n_out, "dropped": len(res.dropped)},
            "outputs": res.outputs, "log": log_rel, "error": None, "contrib": res.contrib(),
            "messages": res.messages, "reason": reason,
        }
        save_state(self.workdir, self.state)
        self._write_datasets()
        tail = f": {n_out}/{n_in} datasets" if n_in else ""
        if res.dropped:
            tail += f"; {len(res.dropped)} dropped, see datasets.csv"
        self.console.say(f"[{index}/{total}] {stage.name} ... done in {duration(seconds)}{tail}")
        for message in res.messages:
            self.console.say(f"      note: {message}")
            self.console.event(event="message", level="warning", stage=stage.name, text=message)
        self.console.event(event="stage_done", stage=stage.name, index=index, total=total,
                           seconds=round(seconds, 1), counts=st_all[stage.name]["counts"])

    def _fail(self, stage, started, t0, log_path, error):
        self.state["stages"][stage.name] = {
            "status": "failed", "started": started, "finished": now(), "seconds": round(time.time() - t0, 1),
            "log": os.path.relpath(log_path, self.workdir), "error": error}
        save_state(self.workdir, self.state)
        self._write_datasets()
        self.console.event(event="stage_failed", stage=stage.name, error=error)


# ---------------------------------------------------------------------------
# Status (read only)
# ---------------------------------------------------------------------------

def status_summary(project, inputs=None):
    """Recorded state plus (if `inputs` is given) which stages a run would redo."""
    workdir = project["workdir"]
    state = load_state(workdir) or {}
    lock = Lock(workdir).read() if os.path.exists(os.path.join(workdir, LOCK_FILE)) else None
    plan = {}
    if inputs is not None and state:
        runner = Runner.__new__(Runner)
        runner.project, runner.inputs, runner.console = project, inputs, None
        runner.workdir, runner.ncpu, runner.verbose = workdir, project["compute"]["ncpu"], False
        runner.state = state
        runner.ctx = Context(runner)
        try:
            plan = {p["stage"]: p for p in runner.plan()}
        except RunError:
            plan = {}
    stages = []
    for stage in STAGES:
        st = (state.get("stages") or {}).get(stage.name) or {}
        p = plan.get(stage.name, {})
        stages.append({"stage": stage.name, "number": stage.number, "folder": stage.folder,
                       "status": st.get("status", "pending"),
                       "applicable": p.get("action") != "not_applicable" if p else st.get("status") != "skipped",
                       "stale": bool(p) and st.get("status") == "done" and p.get("action") == "run",
                       "next_action": p.get("action"), "reason": p.get("reason") or st.get("reason"),
                       "started": st.get("started"), "finished": st.get("finished"), "seconds": st.get("seconds"),
                       "counts": st.get("counts"), "log": st.get("log"), "error": st.get("error"),
                       "messages": st.get("messages") or []})
    datasets = {"total": 0, "ok": 0, "dropped": 0, "dropped_by_stage": {}}
    csv_path = os.path.join(workdir, "datasets.csv")
    if os.path.isfile(csv_path):
        from valdo.easy.ids import read_datasets_csv
        for row in read_datasets_csv(csv_path):
            datasets["total"] += 1
            if row.get("status") == "ok":
                datasets["ok"] += 1
            else:
                datasets["dropped"] += 1
                key = row.get("dropped_stage") or "?"
                datasets["dropped_by_stage"][key] = datasets["dropped_by_stage"].get(key, 0) + 1
    if lock is not None:
        stale, why = Lock(workdir)._stale(lock)
        lock = dict(lock, stale=stale, stale_reason=why)
    return {"workdir": workdir, "has_state": bool(state), "stages": stages, "datasets": datasets, "lock": lock,
            "last_run": state.get("last_run"), "inputs": state.get("inputs")}
