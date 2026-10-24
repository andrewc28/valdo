"""End to end with `phases.source: refine`: `valdo init --phases refine --program dimple` and
`valdo run` on the toy screen, with a stand-in `dimple` (CCP4 is not needed).

The stand-in reads the scaled data valdo gives it and writes the files real dimple writes
(final.mtz with FWT/PHWT and DELFWT/PHDELWT, final.pdb with REMARK 3 R factors, the screen
line with R/Rfree). Its map coefficients are the toy screen's true ones, expressed in the
setting of the variant it is given; a variant in the wrong indexing gets no true phases and a
high R-free, so the filter stage must choose the correct variant from refine_summary.csv. The
test therefore checks the whole refine path: jobs from the scale stage, model frames,
refine -> filter (lowest R-free), the refined phase columns in add_phases_and_blobs, and that
the planted ligands are found. The stand-in fails for one dataset (every variant): that dataset
must leave the pipeline at refine and stay out of everything after it, the VAE's training set
included (review F1: upstream filter keeps scaled files that have no row in its table).
"""

import json
import os
import stat
import sys

import pytest

from conftest import needs_torch, run_valdo, set_yaml_value, valdo_env

pytestmark = needs_torch

FAKE_DIMPLE = r'''
import os, sys
import reciprocalspaceship as rs

args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("dimple 2.7.0 (stand-in for the valdo-easy tests)")
    sys.exit(0)
mtz, pdb, outdir = args[0], args[1], args[2]
stem = os.path.basename(os.path.dirname(os.path.abspath(outdir)))
if stem.split("_")[0] == os.environ.get("FAKE_DIMPLE_FAIL_ID"):
    print("dimple stand-in: refmac failed (on purpose)", file=sys.stderr)
    sys.exit(1)
data = rs.read_mtz(mtz)
truth = os.path.join(os.environ["FAKE_TRUTH_DIR"], stem + ".mtz")
good = os.path.isfile(truth)
if good:
    out = data.join(rs.read_mtz(truth), how="inner")
else:  # wrong indexing: no sensible phases
    out = data.copy()
    amp = out["F"].to_numpy(float)
    for col, kind, values in (("FWT", "SFAmplitude", amp), ("PHWT", "Phase", 0 * amp),
                              ("DELFWT", "SFAmplitude", 0.1 * amp), ("PHDELWT", "Phase", 0 * amp)):
        out[col] = rs.DataSeries(values, index=out.index, dtype=kind)
os.makedirs(outdir, exist_ok=True)
out.write_mtz(os.path.join(outdir, "final.mtz"))
rfree = "0.22450" if good else "0.41000"
with open(pdb) as fh:
    model = fh.read()
with open(os.path.join(outdir, "final.pdb"), "w") as fh:
    fh.write("REMARK   3   R VALUE            (WORKING SET) :  0.18660\n"
             f"REMARK   3   FREE R VALUE                     :  {rfree}\n" + model)
print(f"  5 refmacat restr   31.3s   8/8   R/Rfree  0.2500/0.2700  ->  0.1866/{rfree}")
'''


def _truth_coefficients(folder, truth, out_dir):
    """<out_dir>/<id>_<symop>.mtz with FWT/PHWT/DELFWT/PHDELWT in the setting of the correct variant."""
    import gemmi
    import reciprocalspaceship as rs
    os.makedirs(out_dir, exist_ok=True)
    rename = {"2FOFCWT": "FWT", "PHI2FOFCWT": "PHWT", "FOFCWT": "DELFWT", "PHIFOFCWT": "PHDELWT"}
    for i, d in enumerate(truth["datasets"], start=1):          # IDs follow the sorted names
        ds = rs.read_mtz(os.path.join(folder, "data", d["name"] + ".mtz"))[list(rename)].rename(columns=rename)
        stem = f"{i:04d}_0"
        if d["flipped"]:  # reindex brings it back with the (self-inverse) twin operator
            ds = ds.apply_symop(gemmi.Op(truth["flip_op"])).hkl_to_asu()
            ds = ds.drop(columns=["M/ISYM"])
            stem = f"{i:04d}_1"
        ds.write_mtz(os.path.join(out_dir, stem + ".mtz"))


def _failing_id(truth):
    """ID of a plain dataset (not the reference, bound, flipped or bad) whose refinement fails."""
    for i, d in enumerate(truth["datasets"], start=1):
        if i > 1 and not (d["bound"] or d["flipped"] or d["bad"]):
            return f"{i:04d}"
    raise AssertionError("no plain dataset in the toy screen")


@pytest.fixture(scope="module")
def refine_run(toy_screen, tmp_path_factory):
    folder, truth = toy_screen
    base = tmp_path_factory.mktemp("refine_mode")
    bindir = base / "bin"
    bindir.mkdir()
    script = base / "fake_dimple.py"
    script.write_text(FAKE_DIMPLE)
    wrapper = bindir / "dimple"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _truth_coefficients(folder, truth, str(base / "truth"))
    env = valdo_env()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env["FAKE_TRUTH_DIR"] = str(base / "truth")
    env["FAKE_DIMPLE_FAIL_ID"] = _failing_id(truth)
    project = str(base / "refine.yaml")
    proc, objs = run_valdo(["init", os.path.join(folder, "data"), "--models",
                            os.path.join(folder, "models", "{name}-pandda-input.pdb"),
                            "--apo", os.path.join(folder, "apo.pdb"), "--phases", "refine", "--program", "dimple",
                            "--focal-seqid", str(truth["focal_seqid"]), "--bound-models",
                            os.path.join(folder, "bound"), "--ncpu", "4", "-o", project, "--json"],
                           cwd=str(base), env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    init = objs[-1]
    set_yaml_value(project, "epochs", 30)
    proc, objs = run_valdo(["run", "-c", project, "--json"], cwd=str(base), env=env)
    return project, init, proc, objs, truth


def test_init_chooses_refinement_columns(refine_run):
    project, init, *_ = refine_run
    decisions = {d["setting"]: d for d in init["decisions"]}
    assert decisions["phases.source"]["value"] == "refine"
    from valdo.easy.project import load_project
    p, _ = load_project(project)
    assert p["phases"]["program"] == "dimple"
    assert (p["phases"]["columns"]["phwt"], p["phases"]["columns"]["phdelwt"]) == ("PHWT", "PHDELWT")


def test_refine_mode_runs_and_finds_the_ligands(refine_run):
    project, _, proc, objs, truth = refine_run
    assert proc.returncode == 0, proc.stderr[-5000:]
    final = objs[-1]
    assert final["ok"] and final["report"]
    done = {e["stage"] for e in objs if e.get("event") == "stage_done"}
    assert "refine" in done and "filter" in done
    wd = final["workdir"]
    import pandas as pd
    summary = pd.read_csv(os.path.join(wd, "04_refine", "refine_summary.csv"), dtype=str)
    fail_id = _failing_id(truth)
    failed = summary[summary["stem"].str.startswith(fail_id + "_")]
    assert len(failed) and "ok" not in set(failed["status"])
    assert set(summary.loc[~summary.index.isin(failed.index), "status"]) == {"ok"}
    assert set(summary["program"]) == {"dimple"}
    # the filter kept the variant with the lower R-free: the correct indexing of every dataset
    from valdo.easy.ids import read_datasets_csv
    rows = {r["original_name"]: r for r in read_datasets_csv(os.path.join(wd, "datasets.csv"))}
    for d in truth["datasets"]:
        r = rows[d["name"]]
        if r["status"] == "ok":
            assert r["variant"].endswith("_1" if d["flipped"] else "_0"), r
            assert r["phases_source"] == "refine"
    assert sum(r["status"] == "ok" for r in rows.values()) == 22      # the bad dataset and the failed one
    with open(os.path.join(wd, "configs", "add_phases_and_blobs.yaml")) as handle:
        import yaml
        cfg = yaml.safe_load(handle)
    assert (cfg["phase_2FOFC_col_in"], cfg["phase_FOFC_col_in"]) == ("PHWT", "PHDELWT")
    assert cfg["phasing_path"].rstrip("/").endswith("04_refine/phases")
    with open(os.path.join(wd, "report", "report.json")) as handle:
        rep = json.load(handle)
    per = {d["original_name"]: d for d in rep["validation"]["per_dataset"]}
    assert all(per[n]["first_ligand_blob_rank_in_dataset"] == 1 for n in truth["bound"]), per
    funnel = {f["stage"]: f for f in rep["funnel"]}
    assert "refine" in funnel


def test_failed_refinement_keeps_the_dataset_out_of_the_vae(refine_run):
    """Review F1: a dataset dropped at refine had scaled files without a row in upstream filter's
    table, which keeps such files: they reached kept_files.txt and the VAE's training set."""
    project, _, proc, objs, truth = refine_run
    assert proc.returncode == 0, proc.stderr[-3000:]
    wd = objs[-1]["workdir"]
    fail_id = _failing_id(truth)
    import pandas as pd
    from valdo.easy.ids import read_datasets_csv
    row = {r["id"]: r for r in read_datasets_csv(os.path.join(wd, "datasets.csv"))}[fail_id]
    assert row["status"] == "dropped" and row["dropped_stage"] == "refine", row
    table = pd.read_csv(os.path.join(wd, "configs", "filter_input.csv"), dtype={"file_idx": str})
    mine = table[table["file_idx"] == fail_id]
    assert len(mine) >= 1 and set(mine["Rf_final"]) == {99.0} and set(mine["selected"]) == {0}
    assert all("no longer in the pipeline" in n and "refine" in n for n in mine["note"])
    with open(os.path.join(wd, "05_filter", "kept_files.txt")) as handle:
        kept = [os.path.basename(line.strip()) for line in handle if line.strip()]
    assert kept and not [k for k in kept if k.startswith(fail_id + "_")]
    alive = {r["variant"] for r in read_datasets_csv(os.path.join(wd, "datasets.csv")) if r["status"] == "ok"}
    assert {k[:-len(".mtz")] for k in kept} == alive
    from valdo.easy.stages import _stem
    inter = pd.read_pickle(os.path.join(wd, "06_preprocess", "intersection.pkl"))
    assert {_stem(str(c)) for c in inter.columns} == alive     # the VAE's datasets


def test_refine_mode_resumes_without_refining_again(refine_run):
    project, _, _, _, _ = refine_run
    base = os.path.dirname(project)
    env = valdo_env()
    env["PATH"] = os.path.join(base, "bin") + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env["FAKE_TRUTH_DIR"] = os.path.join(base, "truth")
    proc, objs = run_valdo(["run", "-c", project, "--json"], cwd=base, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert not [e for e in objs if e.get("event") == "stage_start"]
