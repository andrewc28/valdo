"""The valdo-easy changes to upstream files (docs/easy/DESIGN.md, section 10), one test each."""

import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO, needs_science, needs_torch, valdo_env


def test_version_comes_from_version_file():
    import valdo
    with open(os.path.join(REPO, "valdo", "VERSION")) as handle:
        assert valdo.__version__ == handle.read().strip() == "0.0.6+easy.1"
    assert valdo.getVersionNumber() == valdo.__version__


def test_import_valdo_is_lazy_and_needs_no_pkg_resources():
    code = ("import sys; sys.modules['pkg_resources'] = None\n"
            "import valdo\n"
            "assert 'torch' not in sys.modules, 'torch imported'\n"
            "assert 'valdo.helper' not in sys.modules\n"
            "print(valdo.__version__)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=valdo_env(), cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0.0.6+easy.1"


@needs_torch
def test_lazy_public_api_still_works():
    import valdo
    assert valdo.VAE.__name__ == "VAE" and valdo.Scaler.__name__ == "Scaler"
    assert valdo.Scaler_pool.__name__ == "Scaler_pool"
    for name in ("preprocessing", "reindex", "blobs", "tag", "helper", "knn_tools"):
        assert getattr(valdo, name).__name__ == f"valdo.{name}"
    with pytest.raises(AttributeError):
        valdo.does_not_exist


def _toy_mtz(path, n=200, seed=0, extra=None):
    import gemmi
    import reciprocalspaceship as rs
    rng = np.random.default_rng(seed)
    hkl = []
    for h in range(0, 8):
        for k in range(0, 8):
            for l in range(1, 8):
                hkl.append((h, k, l))
    hkl = np.array(hkl[:n])
    ds = rs.DataSet({c: rs.DataSeries(hkl[:, j], dtype="HKL") for j, c in enumerate("HKL")},
                    cell=gemmi.UnitCell(40, 50, 60, 90, 90, 90), spacegroup=gemmi.SpaceGroup("P 1")).set_index(["H", "K", "L"])
    f = rng.uniform(50, 500, len(hkl))
    ds["F-obs"] = rs.DataSeries(f, dtype="SFAmplitude", index=ds.index)
    ds["SIGF-obs"] = rs.DataSeries(0.05 * f, dtype="Stddev", index=ds.index)
    for name, (values, dtype) in (extra or {}).items():
        ds[name] = rs.DataSeries(values[: len(hkl)], dtype=dtype, index=ds.index)
    ds.merged = True
    ds.write_mtz(str(path))
    return str(path)


@needs_torch
def test_standardize_without_trailing_slash_and_relative_paths(tmp_path, monkeypatch):
    from valdo import helper
    src = tmp_path / "raw"
    src.mkdir()
    for i in (1, 2):
        _toy_mtz(src / f"x{i:04d}_data.mtz", seed=i)
    out = tmp_path / "std"
    out.mkdir()
    done = helper.standardize_input_mtzs(str(src), str(out), r".*x(\d{4})_data\.mtz$", ["F-obs", "SIGF-obs"])
    assert sorted(done) == ["0001.mtz", "0002.mtz"] and sorted(os.listdir(out)) == ["0001.mtz", "0002.mtz"]
    # relative source path: upstream joined it twice and skipped every file
    monkeypatch.chdir(tmp_path)
    out2 = tmp_path / "std2"
    out2.mkdir()
    done = helper.standardize_input_mtzs("raw", "std2", r".*x(\d{4})_data\.mtz$", ["F-obs", "SIGF-obs"])
    assert sorted(done) == ["0001.mtz", "0002.mtz"]


@needs_torch
def test_add_phases_output_folder_without_trailing_slash(tmp_path):
    from valdo import helper
    data = tmp_path / "recons"
    data.mkdir()
    phases = tmp_path / "phases"
    phases.mkdir()
    out = tmp_path / "phased"
    out.mkdir()
    n = 200
    ph = np.linspace(-180, 180, n)
    _toy_mtz(data / "0001_0.mtz")
    _toy_mtz(phases / "0001_0.mtz", extra={"PH2FOFCWT": (ph, "Phase"), "PHFOFCWT": (ph, "Phase")})
    missing = helper.add_phases([str(data / "0001_0.mtz")], str(phases), str(out))
    assert missing == [] and os.listdir(out) == ["0001_0.mtz"]
    out2 = tmp_path / "phased_pool"
    out2.mkdir()
    helper.add_phases_pool([str(data / "0001_0.mtz")], str(phases), str(out2), ncpu=1)
    assert os.listdir(out2) == ["0001_0.mtz"]


@needs_torch
def test_scaler_output_folder_without_trailing_slash(tmp_path):
    from valdo.scaling import Scaler
    ref = _toy_mtz(tmp_path / "ref.mtz", seed=1)
    a = _toy_mtz(tmp_path / "0002.mtz", seed=2)
    out = tmp_path / "scaled"
    out.mkdir()
    Scaler(reference_mtz=ref, columns=["F-obs", "SIGF-obs"]).batch_scaling([a], outputmtz_path=str(out), prefix="",
                                                                          when_opt="never", verbose=False)
    assert sorted(os.listdir(out)) == ["0002.mtz", "scaling_metrics.pkl"]


@needs_torch
def test_determine_locations_folder_without_trailing_slash(tmp_path):
    import pandas as pd
    from valdo import tag
    folder = tmp_path / "mtz"
    folder.mkdir()
    _toy_mtz(folder / "0001_0.mtz")
    df = pd.DataFrame([{"sample": "0001_0", "cenx": 1.0, "ceny": 2.0, "cenz": 3.0}])
    out = tag.determine_locations(df, str(folder))
    assert len(out["all_possible_cart"].iloc[0]) == 1          # P 1: one symmetry copy


@needs_science
def test_filter_wide_ids_and_no_reindex_rfree_cut(tmp_path):
    from valdo.commandline.pipeline import run_filter
    import reciprocalspaceship as rs
    scaled = tmp_path / "scaled"
    scaled.mkdir()
    for stem in ("00001", "00002", "00003"):
        path = _toy_mtz(scaled / f"{stem}.mtz")
        ds = rs.read_mtz(path)
        ds["F-obs-scaled"] = ds["F-obs"]
        ds.write_mtz(path)
    summary = tmp_path / "summary.csv"
    summary.write_text("file_idx,symop,Rf_final\n00001,,0.20\n00002,,0.60\n00003,,0.30\n")
    out = tmp_path / "kept.txt"
    run_filter({"refine_summary": str(summary), "reindex_record": None, "scaled_dir": str(scaled), "metrics": None,
                "max_rfree": 0.45, "min_cc": 0.0, "output": str(out), "_force": True})
    kept = [os.path.basename(l.strip()) for l in open(out) if l.strip()]
    # upstream read file_idx as int (00002 -> '0002') and only matched '{idx}_{symop}.mtz': nothing was cut
    assert kept == ["00001.mtz", "00003.mtz"]


def test_refine_output_prefix_has_no_stray_parenthesis(tmp_path, monkeypatch):
    from valdo.commandline import refine
    for i in (1, 2):
        (tmp_path / f"m{i}.pdb").write_text("END\n")
        (tmp_path / f"d{i}.mtz").write_text("")
    calls = []
    monkeypatch.setattr(refine.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))
    monkeypatch.setattr(sys, "argv", ["valdo.refine", "--pdbpath", str(tmp_path / "m*.pdb"), "--mtzpath",
                                      str(tmp_path / "d*.mtz"), "--output", str(tmp_path / "out"), "--eff", "x.eff"])
    refine.main()
    prefixes = [c[3] for c in calls]
    assert prefixes == [f"output.prefix={tmp_path / 'out' / 'refine_d1'}", f"output.prefix={tmp_path / 'out' / 'refine_d2'}"]


def test_setup_declares_console_script_and_package_data():
    text = open(os.path.join(REPO, "setup.py")).read()
    assert "valdo=valdo.easy.cli:main" in text and '"VERSION"' in text and "refine_drug.eff" in text
    assert 'exclude=["tests", "tests.*"]' in text
    manifest = open(os.path.join(REPO, "MANIFEST.in")).read()
    assert "refine_drug.eff" in manifest


def test_pipeline_module_runs_main():
    """`python -m valdo.commandline.pipeline` used to exit 0 without doing anything."""
    proc = subprocess.run([sys.executable, "-m", "valdo.commandline.pipeline", "init", "standardize"],
                          capture_output=True, text=True, env=valdo_env(), cwd="/")
    assert proc.returncode == 0 and "destination_path" in proc.stdout, proc.stderr
