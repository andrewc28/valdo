"""`valdo check` findings for common set-up mistakes (review R1-02, R1-06, R1-07), on the toy screen.

Only the input and settings checks are exercised (check._check_inputs / _check_settings), so the
tests do not depend on the environment (torch, CCP4) of the machine that runs them.
"""

import os

import pytest

from conftest import needs_science

pytestmark = needs_science


def _project(tmp_path, folder, extra=""):
    from valdo.easy import project as P
    path = tmp_path / "valdo.yaml"
    path.write_text(f"data: {folder}/data\nmodels:\n  apo: {folder}/apo.pdb\n" + extra)
    proj, _ = P.load_project(str(path))
    return proj


def _findings(proj):
    from valdo.easy import check
    inp = check.inspect_inputs(proj, threads=4)
    out = check._check_inputs(proj, inp) + check._check_settings(proj, inp, {"cpus": 4, "gpu": {"gpus": 0}})
    found = {}
    for f in out:   # code -> the first finding with that code (excluded_inputs: one per reason)
        found.setdefault(f["code"], f)
    found["_all"] = out
    return found, inp


def test_models_pattern_that_matches_nothing_is_an_error(toy_screen, tmp_path):
    """R1-02: a wrong models.per_dataset used to pass 'valdo check' silently (every dataset then got
    the apo model)."""
    folder, _ = toy_screen
    proj = _project(tmp_path, folder, f"  per_dataset: {folder}/models/{{name}}-final.pdb\n")
    found, inp = _findings(proj)
    f = found["models_pattern_no_match"]
    assert f["level"] == "error" and "matched none of the 24 datasets" in f["what"]
    assert any("TOY-x0001-final.pdb" in item for item in f["items"])
    assert any("TOY-x0001-pandda-input.pdb" in item for item in f["items"])   # what is there instead
    assert "fallback_models" not in found and "missing_models" not in found     # one error, not three
    # the right pattern: no finding about models at all
    proj = _project(tmp_path, folder, f"  per_dataset: {folder}/models/{{name}}-pandda-input.pdb\n")
    found, _ = _findings(proj)
    assert not {"models_pattern_no_match", "fallback_models", "missing_models"} & set(found)


def test_fallback_model_for_many_datasets_is_a_warning(toy_screen, tmp_path):
    folder, truth = toy_screen
    some = tmp_path / "some_models"
    some.mkdir()
    for d in truth["datasets"][:3]:
        os.symlink(os.path.join(folder, "models", f"{d['name']}-pandda-input.pdb"),
                   str(some / f"{d['name']}-pandda-input.pdb"))
    proj = _project(tmp_path, folder, f"  per_dataset: {some}/{{name}}-pandda-input.pdb\n")
    found, _ = _findings(proj)
    f = found["fallback_models"]
    assert f["level"] == "warning" and f["title"].startswith("21 of 24 dataset(s)") and f["fix"]
    assert "models_pattern_no_match" not in found
    for d in truth["datasets"][3:23]:      # now only one dataset lacks its own model: information
        os.symlink(os.path.join(folder, "models", f"{d['name']}-pandda-input.pdb"),
                   str(some / f"{d['name']}-pandda-input.pdb"))
    found, _ = _findings(proj)
    assert found["fallback_models"]["level"] == "info" and found["fallback_models"]["title"].startswith("1 of 24")


def test_wrong_amplitude_columns_name_the_right_ones(toy_screen, tmp_path):
    """R1-06: with columns that no file has, every file is excluded; the fix names the columns the files
    do have, and no other finding repeats the same problem."""
    folder, _ = toy_screen
    proj = _project(tmp_path, folder, "columns:\n  amplitude: FP\n  sigma: SIGFP\n")
    found, inp = _findings(proj)
    assert not inp.ok_rows()
    excluded = {f["title"].split(": ", 1)[1]: f for f in found["_all"] if f["code"] == "excluded_inputs"}
    f = excluded["missing column FP, SIGFP"]
    assert f["level"] == "error" and f["title"].startswith("24 file(s)")
    assert "columns.amplitude: F-obs" in f["fix"] and "columns.sigma: SIGF-obs" in f["fix"]
    assert "the other datasets run without them" not in excluded["unreadable MTZ"]["fix"]   # there are none
    assert {f["code"] for f in found["_all"]} == {"excluded_inputs"}   # no cell/model/reference/too-few noise


def test_missing_bound_models_folder_is_one_clear_warning(toy_screen, tmp_path):
    """R1-07: a validation.bound_models path that does not exist was reported as 'no bound model matched'."""
    folder, _ = toy_screen
    proj = _project(tmp_path, folder, f"validation:\n  bound_models: {folder}/no_such_folder\n")
    found, _ = _findings(proj)
    f = found["bound_models_missing"]
    assert f["level"] == "warning" and "no_such_folder does not exist" in f["what"]
    assert "no_bound_models" not in found
    proj = _project(tmp_path, folder, f"validation:\n  bound_models: {folder}/bound\n")
    found, _ = _findings(proj)
    assert not {"bound_models_missing", "no_bound_models"} & set(found)


def test_init_warns_about_a_models_pattern_that_matches_nothing(toy_screen, tmp_path):
    from conftest import run_valdo
    folder, _ = toy_screen
    out = str(tmp_path / "p.yaml")
    proc, objs = run_valdo(["init", os.path.join(folder, "data"), "--models",
                            os.path.join(folder, "models", "{name}-final.pdb"), "--apo",
                            os.path.join(folder, "apo.pdb"), "-o", out, "--json"], cwd=str(tmp_path), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert any("matched none" in w for w in objs[-1]["warnings"]), objs[-1]["warnings"]
