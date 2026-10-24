"""The `valdo` command: JSON result schema, exit codes, error messages (in-process, fast)."""

import json
import os
import subprocess
import sys

from conftest import REPO, needs_science, needs_valdo_env, run_valdo, valdo_env
from valdo.easy import SCHEMA_ID, cli


def results(capfd):
    out, err = capfd.readouterr()
    lines = [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]
    return lines, err


def test_version_json(capfd):
    assert cli.main(["--version", "--json"]) == 0
    lines, _ = results(capfd)
    last = lines[-1]
    assert last["schema"] == SCHEMA_ID and last["command"] == "version" and last["ok"] is True
    assert last["version"].startswith("0.0.6+easy") and last["upstream_base"] == "1ec32c6"
    assert last["next"] == []


def test_version_human(capfd):
    assert cli.main(["--version"]) == 0
    out, _ = capfd.readouterr()
    assert "upstream VALDO 0.0.6" in out and "1ec32c6" in out


def test_stages_json(capfd):
    assert cli.main(["stages", "--json"]) == 0
    lines, _ = results(capfd)
    stages = lines[-1]["stages"]
    assert [s["name"] for s in stages][:3] == ["standardize", "reindex", "scale"]
    assert set(stages[0]) == {"name", "number", "folder", "upstream", "runs_when", "description"}


def test_json_flag_before_or_after_command(capfd):
    assert cli.main(["--json", "stages"]) == 0
    assert results(capfd)[0][-1]["command"] == "stages"


def test_help_and_usage_errors(capfd):
    assert cli.main(["--help"]) == 0
    assert cli.main(["run", "--frm", "x"]) == 1
    capfd.readouterr()


def test_invalid_project_is_explained(tmp_path, capfd):
    path = tmp_path / "valdo.yaml"
    path.write_text("data: mtz\nmodels:\n  apo: apo.pdb\nadvanced:\n  train:\n    epoch: 5\n")
    assert cli.main(["check", str(path), "--json"]) == 1
    lines, err = results(capfd)
    last = lines[-1]
    assert last["ok"] is False and last["command"] == "check"
    assert last["error"]["code"] == "project_invalid"
    assert any("advanced.train.epochs" in d for d in last["error"]["details"])
    # suggested commands name the project file, which is not ./valdo.yaml here
    assert "Did you mean" in err and last["next"] == [f"valdo check -c {path}"]


def test_yaml_syntax_error_is_explained(tmp_path, capfd):
    """Review R1-08: a YAML syntax error was explained as a strict-settings problem."""
    path = tmp_path / "valdo.yaml"
    path.write_text("data: mtz\nmodels:\n\tapo: apo.pdb\n")       # a tab character
    assert cli.main(["check", str(path), "--json"]) == 1
    err = results(capfd)[0][-1]["error"]
    assert err["code"] == "project_invalid" and "YAML syntax" in err["cause"] and "strictly" not in err["cause"]
    assert "is not valid YAML near line 3" in err["details"][0]
    assert any(d.startswith("YAML parser:") for d in err["details"])


def test_missing_project(tmp_path, capfd):
    assert cli.main(["status", str(tmp_path / "nope.yaml"), "--json"]) == 1
    lines, _ = results(capfd)
    assert "valdo init" in lines[-1]["error"]["details"][0]


def test_init_needs_data_and_model(tmp_path, capfd, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--json"]) == 1
    lines, _ = results(capfd)
    assert lines[-1]["error"]["code"] == "missing_data" and "valdo init" in lines[-1]["error"]["fix"]
    (tmp_path / "d").mkdir()
    assert cli.main(["init", "d", "--json"]) == 1
    lines, _ = results(capfd)
    assert lines[-1]["error"]["code"] == "missing_models" and "--apo" in lines[-1]["error"]["fix"]
    assert cli.main(["init", "d", "--apo", "apo.pdb", "--json"]) == 1
    lines, _ = results(capfd)
    assert lines[-1]["error"]["code"] == "model_missing"
    (tmp_path / "apo.pdb").write_text("END\n")
    assert cli.main(["init", "d", "--apo", "apo.pdb", "--json"]) == 1
    lines, _ = results(capfd)
    assert lines[-1]["error"]["code"] == "no_data"


def test_project_given_twice(tmp_path, capfd):
    assert cli.main(["status", "a.yaml", "-c", "b.yaml", "--json"]) == 1
    assert results(capfd)[0][-1]["error"]["code"] == "bad_options"


def test_import_valdo_easy_cli_does_not_import_torch():
    code = ("import sys; import valdo, valdo.easy.cli, valdo.easy.check, valdo.easy.stages, valdo.easy.runner; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], env=valdo_env(), cwd=REPO).returncode == 0


def test_console_script_entry_point_runs():
    proc = subprocess.run([sys.executable, "-m", "valdo.easy.cli", "--version"], capture_output=True, text=True,
                          env=valdo_env(), cwd=REPO)
    assert proc.returncode == 0 and "valdo 0.0.6+easy" in proc.stdout


@needs_valdo_env
def test_init_check_on_toy(toy_project, capfd):
    path, init = toy_project
    assert init["ok"] and init["command"] == "init" and init["n_datasets"] == 24
    decisions = {d["setting"]: d for d in init["decisions"]}
    assert decisions["phases.source"]["value"] == "input_mtz"
    assert decisions["reindex"]["value"] == "auto -> on" and "-h,-k,l" in decisions["reindex"]["why"]
    assert decisions["reference"]["value"].endswith("(0001)")
    assert init["summary"]["n_files"] == 25 and init["summary"]["n_excluded"] == 1
    assert init["summary"]["scan_seconds"] < 30
    text = open(path).read()
    assert 'tag: "cys10"' in text and 'phwt: "PHI2FOFCWT"' in text and 'rfree: "R-free-flags"' in text
    # refuses to overwrite
    proc, objs = run_valdo(["init", "data", "--apo", "apo.pdb", "--json"], cwd=os.path.dirname(path))
    assert proc.returncode == 1 and objs[-1]["error"]["code"] == "project_exists"
    # check: no errors; the broken file and the small screen are warnings
    assert cli.main(["check", path, "--json"]) == 0
    lines, _ = results(capfd)
    chk = lines[-1]
    assert chk["ok"] and chk["errors"] == 0 and chk["inputs"]["n_usable"] == 24
    codes = {f["code"] for f in chk["findings"]}
    assert {"excluded_inputs", "few_datasets", "reference", "reindex"} <= codes
    excluded = next(f for f in chk["findings"] if f["code"] == "excluded_inputs")
    assert any("0025" in item for item in excluded["items"])
    for f in chk["findings"]:
        assert set(f) == {"level", "code", "title", "what", "why", "fix", "items"}


@needs_valdo_env
def test_check_deep_on_toy(toy_project, capfd):
    path, _ = toy_project
    assert cli.main(["check", path, "--deep", "--json"]) == 0
    lines, _ = results(capfd)
    assert "deep_ok" in {f["code"] for f in lines[-1]["findings"]}


@needs_valdo_env
def test_dry_run_plan(toy_project, capfd):
    path, _ = toy_project
    import shutil
    wd = os.path.join(os.path.dirname(path), "valdo_run")
    existed = os.path.exists(wd)
    assert cli.main(["run", path, "--dry-run", "--json"]) == 0
    lines, _ = results(capfd)
    plan = {p["stage"]: p for p in lines[-1]["plan"]}
    assert plan["refine"]["action"] == "not_applicable" and plan["reindex"]["action"] in ("run", "skip")
    if not existed:
        assert not os.path.exists(wd)      # a dry run writes nothing
        shutil.rmtree(wd, ignore_errors=True)


def test_status_while_a_run_is_active_says_wait(tmp_path, capfd):
    """Review R1-04: with a run active in the work directory, the next step is to poll
    'valdo status', not to start another run (it would only stop with 'locked')."""
    from valdo.easy import runner
    path = tmp_path / "valdo.yaml"
    path.write_text("data: mtz\nmodels:\n  apo: apo.pdb\n")
    work = tmp_path / "valdo_run"
    work.mkdir()
    (work / "valdo_state.json").write_text(json.dumps({"stages": {"standardize": {"status": "done"}}}))
    lock = runner.Lock(str(work))
    lock.acquire()
    try:
        assert cli.main(["status", str(path), "--json"]) == 0
        res = results(capfd)[0][-1]
        assert res["lock"]["stale"] is False
        assert res["wait"]["poll"] == "valdo status" and res["wait"]["seconds"] == 60
        assert res["next"] == [f"valdo status -c {path}"]
    finally:
        lock.release()
    assert cli.main(["status", str(path), "--json"]) == 0
    res = results(capfd)[0][-1]
    assert "wait" not in res and res["lock"] is None and res["next"] == [f"valdo run -c {path}"]
    # the 'locked' error itself points to 'valdo status'
    assert cli._next_after_error("run", "locked") == ["valdo status"]


def test_check_on_a_login_node_suggests_a_batch_job(tmp_path, capfd, monkeypatch):
    """Review R1-04: on a login node 'valdo check' must not suggest 'valdo run'."""
    from valdo.easy import check
    path = tmp_path / "valdo.yaml"
    path.write_text("data: mtz\nmodels:\n  apo: apo.pdb\n")
    report = {"findings": [], "errors": 0, "warnings": 0, "environment": {"login_node": True},
              "inputs": {"n_usable": 120, "n_files": 120}}
    monkeypatch.setattr(check, "run_checks", lambda *a, **k: report)
    monkeypatch.setattr(cli, "_detect_scheduler", lambda: "sge")
    assert cli.main(["check", str(path), "--json"]) == 0
    res = results(capfd)[0][-1]
    assert res["next"] == [f"valdo jobscript -o run_valdo.sh -c {path}", "qsub run_valdo.sh"]
    report["environment"]["login_node"] = False
    assert cli.main(["check", str(path), "--json"]) == 0
    assert results(capfd)[0][-1]["next"] == [f"valdo run -c {path}"]


def test_jobscript_writes_an_executable_script(tmp_path, capfd):
    path = tmp_path / "valdo.yaml"
    path.write_text("data: mtz\nmodels:\n  apo: apo.pdb\ncompute:\n  ncpu: 6\n")
    out = tmp_path / "run_valdo.sh"
    assert cli.main(["jobscript", str(path), "-o", str(out), "--scheduler", "sge", "--json"]) == 0
    res = results(capfd)[0][-1]
    assert res["ok"] and res["scheduler"] == "sge" and res["path"] == str(out)
    assert res["next"] == [f"qsub {out}", f"valdo status -c {path}"]
    text = out.read_text()
    assert os.access(str(out), os.X_OK) and 'valdo run -c "$CONFIG"' in text and str(path) in text
    assert "-pe smp 6" in text                       # compute.ncpu
    assert (tmp_path / "valdo_run" / "logs").is_dir()
    assert "The log directory must exist first" not in text   # it was created before the script was written
    assert res["hours"] == 24.0 and "no data" in res["hours_why"]   # no data folder to estimate from


@needs_science
def test_jobscript_estimates_the_wall_time(toy_project, tmp_path, capfd):
    """Review R1-10: the wall-clock limit follows the size of the screen instead of a flat 24 h."""
    path, _ = toy_project
    out = tmp_path / "job.sh"
    assert cli.main(["jobscript", path, "-o", str(out), "--scheduler", "sge", "--json"]) == 0
    res = results(capfd)[0][-1]
    assert res["slots"] == 2 and res["hours"] == cli.suggest_hours(25, 2, "input_mtz") == 1.0
    assert "25 datasets" in res["hours_why"] and "#$ -l h_rt=01:00:00" in out.read_text()
    assert cli.main(["jobscript", path, "-o", str(out), "--hours", "7", "--scheduler", "sge", "--json"]) == 0
    res = results(capfd)[0][-1]
    assert res["hours"] == 7 and "#$ -l h_rt=07:00:00" in out.read_text()


def test_jobscript_env_setup_keeps_the_current_environment(monkeypatch):
    """Found on Wynton: `--env-setup "source .../ccp4.setup-sh"` (as WYNTON.md suggests for dimple)
    used to replace the activation of the environment that holds valdo, so the job stopped with
    "'valdo' is not on PATH". Extra lines now come after the current environment's activation;
    lines that activate an environment themselves still replace it."""
    venv = ["source /envs/v/bin/activate   # the environment 'valdo jobscript' ran in"]
    monkeypatch.setattr(cli, "_current_env_setup", lambda: venv)
    ccp4 = "source ~/ccp4-9/bin/ccp4.setup-sh"
    assert cli._job_env_setup(None) == venv
    assert cli._job_env_setup([ccp4]) == venv + [ccp4]
    conda = ["source ~/miniforge3/etc/profile.d/conda.sh", "conda activate valdo", ccp4]
    assert cli._job_env_setup(conda) == conda
    monkeypatch.setattr(cli, "_current_env_setup", lambda: None)
    assert cli._job_env_setup([ccp4]) == [ccp4]
