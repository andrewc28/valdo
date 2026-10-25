"""Tests for valdo.easy.refine (per-dataset refinement with phenix.refine or dimple).

PHENIX and CCP4 are never run here: fake ``phenix.refine`` and ``dimple``
programs (small Python scripts put first on PATH) read the command line the
way the real programs do and write the files the real programs write
(refined MTZ with map coefficients, model with REMARK 3, logs). Their
behaviour per dataset is set with the FAKE_BEHAVIOUR environment variable.
The R-factor parsers are tested on excerpts of real dimple 2.7 / refmac
output (CCP4 9, PTP1B y0001).
"""

import json
import math
import os
import stat
import sys
import textwrap
import time

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from valdo.easy import refine

TRIGONAL = (gemmi.UnitCell(40, 40, 50, 90, 90, 120), gemmi.SpaceGroup("P 31 2 1"))
ORTHO = (gemmi.UnitCell(30, 35, 40, 90, 90, 90), gemmi.SpaceGroup("P 21 21 21"))
ATOMS = [(3.1, 4.2, 5.3), (6.4, 2.5, 9.6), (11.0, 8.7, 4.4), (2.2, 13.3, 15.5), (8.8, 9.9, 20.1)]

# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def asu_hkl(cell, sg, dmin):
    H = rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous=False)
    return H[np.any(H != 0, axis=1)]


def write_dataset(path, cell, sg, dmin=3.0, seed=0, free="phenix", extra_pair=False, misym=False):
    """Random amplitudes (F-obs/SIGF-obs) with optional R-free flags and a second amplitude pair."""
    rng = np.random.default_rng(seed)
    H = asu_hkl(cell, sg, dmin)
    n = len(H)
    F = np.abs(rng.normal(size=n) + 1j * rng.normal(size=n)) * 100.0 + 1.0
    cols = {"H": H[:, 0], "K": H[:, 1], "L": H[:, 2], "F-obs": F, "SIGF-obs": 0.05 * F + 1.0}
    types = {"H": "HKL", "K": "HKL", "L": "HKL", "F-obs": "SFAmplitude", "SIGF-obs": "Stddev"}
    if free == "phenix":
        cols["R-free-flags"] = (rng.random(n) < 0.1).astype(np.int32)
        types["R-free-flags"] = "MTZInt"
    elif free == "ccp4":
        cols["FreeR_flag"] = rng.integers(0, 20, n).astype(np.int32)
        types["FreeR_flag"] = "MTZInt"
    elif free == "single":
        cols["R-free-flags"] = np.zeros(n, dtype=np.int32)
        types["R-free-flags"] = "MTZInt"
    elif free == "both":     # two R-free columns (PHENIX's and CCP4's convention)
        cols["R-free-flags"] = (rng.random(n) < 0.1).astype(np.int32)
        cols["FreeR_flag"] = rng.integers(0, 20, n).astype(np.int32)
        types.update({"R-free-flags": "MTZInt", "FreeR_flag": "MTZInt"})
    if extra_pair:
        cols["FP"] = F * 1.01
        cols["SIGFP"] = 0.05 * F + 1.0
        types.update(FP="SFAmplitude", SIGFP="Stddev")
    ds = rs.DataSet(cols, cell=cell, spacegroup=sg, merged=True).astype(types).set_index(["H", "K", "L"])
    if misym:
        ds = ds.hkl_to_asu()  # adds M/ISYM, as upstream's reindex output has
    ds.write_mtz(str(path))
    return str(path)


def point_coefficients(H, cell, sg, points, B=15.0):
    """Structure factors of point scatterers with full space-group symmetry (consistent phases)."""
    s2 = 1.0 / rs.utils.compute_dHKL(H, cell) ** 2
    F = np.zeros(len(H), dtype=complex)
    for p in points:
        f = cell.fractionalize(gemmi.Position(*p))
        for op in sg.operations():
            x = np.array(op.apply_to_xyz([f.x, f.y, f.z]))
            F += np.exp(2j * np.pi * (H @ x))
    F *= np.exp(-B * s2 / 4.0)
    return np.abs(F), np.rad2deg(np.angle(F))


def write_truth(path, data_mtz, points):
    """Map coefficients FT/PHT on the reflections of `data_mtz` (same setting)."""
    ds = rs.read_mtz(data_mtz)
    H = ds.get_hkls()
    amp, phi = point_coefficients(H, ds.cell, ds.spacegroup, points)
    out = rs.DataSet({"FT": amp, "PHT": phi}, index=ds.index, cell=ds.cell, spacegroup=ds.spacegroup,
                     merged=True).astype({"FT": "SFAmplitude", "PHT": "Phase"})
    out.write_mtz(str(path))
    return str(path)


def write_model(path, cell, sg, positions=ATOMS, aniso=False):
    st = gemmi.Structure()
    st.cell = cell
    st.spacegroup_hm = sg.hm
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, p in enumerate(positions, start=1):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(i, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*p)
        atom.occ = 1.0
        atom.b_iso = 20.0
        if aniso:
            atom.aniso = gemmi.SMat33f(0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.write_pdb(str(path))
    return str(path)


def coords(path):
    st = gemmi.read_structure(str(path))
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in st[0]["A"] for a in r])


def complex_cc(path_a, fa, pa, path_b, fb, pb):
    a = rs.read_mtz(str(path_a))[[fa, pa]].dropna().hkl_to_asu()
    b = rs.read_mtz(str(path_b))[[fb, pb]].dropna().hkl_to_asu()
    j = a[[fa, pa]].join(b[[fb, pb]], how="inner", rsuffix="_b")
    ca = j[fa].to_numpy(float) * np.exp(1j * np.deg2rad(j[pa].to_numpy(float)))
    cb = j.iloc[:, 2].to_numpy(float) * np.exp(1j * np.deg2rad(j.iloc[:, 3].to_numpy(float)))
    return float(np.real(np.sum(ca * np.conj(cb))) / np.sqrt(np.sum(abs(ca) ** 2) * np.sum(abs(cb) ** 2))), len(j)


# ---------------------------------------------------------------------------
# Fake programs
# ---------------------------------------------------------------------------

FAKE_PROGRAM = r'''
import json, os, sys, time
import gemmi
import numpy as np
import reciprocalspaceship as rs

def model_text(src, mode, rw, rf, remark=True):
    st = gemmi.read_structure(src)
    if mode == "reindex":  # the model stays, the data are re-indexed: express the model in the new setting
        for m in st:
            for ch in m:
                for res in ch:
                    for at in res:
                        f = st.cell.fractionalize(at.pos)
                        at.pos = st.cell.orthogonalize(gemmi.Fractional(-f.x, -f.y, f.z))
    text = st.make_pdb_string()
    if remark:
        text = ("REMARK   3   R VALUE     (WORKING + TEST SET) : 0.19000\n"
                f"REMARK   3   R VALUE            (WORKING SET) :  {rw}\n"
                f"REMARK   3   FREE R VALUE                     :  {rf}\n"
                "REMARK   3   FREE R VALUE TEST SET SIZE   (%) : 5.0\n") + text
    return text

def coefficients(data, stem, names, mode):
    truth = os.path.join(os.environ.get("FAKE_TRUTH_DIR", "/nonexistent"), stem + ".mtz")
    out = data.copy()
    if "M/ISYM" in out.columns:
        out = out.drop(columns=["M/ISYM"])
    if mode != "nomaps":
        if os.path.isfile(truth):
            t = rs.read_mtz(truth)
            out = out.join(t[["FT", "PHT"]], how="inner")
            amp, phi = out["FT"], out["PHT"]
            out = out.drop(columns=["FT", "PHT"])
        else:
            amp = out[out.columns[0]].astype(float)
            phi = amp * 0.0
        (f2, p2), (f1, p1) = names
        out[f2] = rs.DataSeries(amp.to_numpy(), index=out.index, dtype="SFAmplitude")
        out[p2] = rs.DataSeries(phi.to_numpy(), index=out.index, dtype="Phase")
        out[f1] = rs.DataSeries(0.1 * amp.to_numpy(), index=out.index, dtype="SFAmplitude")
        out[p1] = rs.DataSeries(phi.to_numpy(), index=out.index, dtype="Phase")
    if mode == "reindex":
        out = out.apply_symop(gemmi.Op("-x,-y,z")).hkl_to_asu()
        out = out.drop(columns=["M/ISYM"])
    if mode == "newsg":
        out.spacegroup = gemmi.SpaceGroup("P 32 2 1")
    return out

def main():
    prog = sys.argv[1]
    args = sys.argv[2:]
    if args[:1] == ["--version"]:
        print("dimple 2.7.0 (fake)")
        return 0
    if prog == "dimple":
        mtz, pdb, outdir = args[0], args[1], args[2]
        stem = os.path.basename(os.path.dirname(os.path.abspath(outdir)))
    else:
        pdb, mtz = args[0], args[1]
        stem = os.path.basename(os.getcwd())
        outdir = os.getcwd()
    with open(os.environ["FAKE_CALLS"], "a") as fh:
        fh.write(json.dumps({"prog": prog, "stem": stem, "argv": args, "cwd": os.getcwd(),
                             "omp": os.environ.get("OMP_NUM_THREADS")}) + "\n")
    mode = json.loads(os.environ.get("FAKE_BEHAVIOUR", "{}")).get(stem, "ok")
    if mode == "fail":
        print("Sorry: fake refinement failed on purpose")
        return 2
    if mode == "sleep":
        time.sleep(120)
        return 0
    if mode == "nooutput":
        print("finished, but wrote nothing")
        return 0
    os.makedirs(outdir, exist_ok=True)
    data = rs.read_mtz(mtz)
    if prog == "dimple":
        out = coefficients(data, stem, (("FWT", "PHWT"), ("DELFWT", "PHDELWT")), mode)
        out.write_mtz(os.path.join(outdir, "final.mtz"))
        with open(os.path.join(outdir, "final.pdb"), "w") as fh:
            fh.write(model_text(pdb, mode, "0.18660", "0.22450", remark=(mode != "nor")))
        if mode == "mr":  # verbatim from a dimple 2.7 run that fell back to molecular replacement
            print("  3 refmac5 rigid    5.3s  10/10   R  0.5538  ->  0.5512")
            print("Run MR for R > 0.4. Volume of asu: 100.0% of model asu.")
            print("  4 phaser          32.1s  TF*0 PAK=0 LLG=3969 *T=1 TFZ==56.0 LLG=13474 *T=1...")
        if mode != "nor":
            print("  5 refmacat restr   31.3s   8/8   R/Rfree  0.2500/0.2700  ->  0.1866/0.2245")
            with open(os.path.join(outdir, "dimple.log"), "w") as fh:
                fh.write("[refmac restr]\noverall_r: 0.1866\nfree_r: 0.2245\n")
    else:
        prefix = [a for a in args if a.startswith("output.prefix=")][0].split("=", 1)[1]
        out = coefficients(data, stem, (("2FOFCWT", "PH2FOFCWT"), ("FOFCWT", "PHFOFCWT")), mode)
        out.write_mtz(os.path.join(outdir, prefix + "_001.mtz"))
        with open(os.path.join(outdir, prefix + "_001.pdb"), "w") as fh:
            fh.write(model_text(pdb, mode, "0.19900", "0.22900"))
        with open(os.path.join(outdir, prefix + "_001.log"), "w") as fh:
            fh.write("=== summary ===\nStart R-work = 0.2500, R-free = 0.2700\n"
                     "Final R-work = 0.2000, R-free = 0.2300\n")
    return 0

sys.exit(main())
'''


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Put fake `dimple` and `phenix.refine` first on PATH; returns a helper object."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = tmp_path / "fake_program.py"
    script.write_text(FAKE_PROGRAM)
    for prog in ("dimple", "phenix.refine"):
        wrapper = bindir / prog
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {prog} "$@"\n')
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"))
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("FAKE_BEHAVIOUR", "{}")
    monkeypatch.delenv("FAKE_TRUTH_DIR", raising=False)

    class Fakes:
        bin = bindir

        @staticmethod
        def behave(**modes):
            monkeypatch.setenv("FAKE_BEHAVIOUR", json.dumps(modes))

        @staticmethod
        def calls():
            if not calls.exists():
                return []
            return [json.loads(line) for line in calls.read_text().splitlines() if line.strip()]

    return Fakes()


def quiet(*args, **kwargs):
    pass


# ---------------------------------------------------------------------------
# Small helpers and parsers
# ---------------------------------------------------------------------------

def test_phase_columns():
    assert refine.phase_columns("phenix") == ("PH2FOFCWT", "PHFOFCWT")
    assert refine.phase_columns("dimple") == ("PHWT", "PHDELWT")
    with pytest.raises(ValueError):
        refine.phase_columns("buster")


def test_detect_programs(tmp_path, monkeypatch, fakes):
    monkeypatch.delenv("CCP4", raising=False)
    phenix_bin = tmp_path / "phenix-1.21.2-5419" / "bin"
    phenix_bin.mkdir(parents=True)
    exe = phenix_bin / "phenix.refine"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(fakes.bin), str(phenix_bin), "/usr/bin", "/bin"]))
    found = refine.detect_programs()
    assert found["dimple"] == str(fakes.bin / "dimple")
    assert found["versions"]["dimple"] == "2.7.0"
    assert found["phenix"] == str(fakes.bin / "phenix.refine")  # first on PATH wins
    assert found["refmac5"] is None
    assert any("refmac5" in n for n in found["notes"]) and any("CCP4" in n for n in found["notes"])
    monkeypatch.setenv("PATH", os.pathsep.join([str(phenix_bin), "/usr/bin", "/bin"]))
    found = refine.detect_programs(probe_versions=False)
    assert found["dimple"] is None and found["versions"]["phenix"] == "1.21.2-5419"


def test_inspect_mtz_columns_and_flags(tmp_path):
    cell, sg = ORTHO
    info = refine.inspect_mtz(write_dataset(tmp_path / "a.mtz", cell, sg, free="phenix", extra_pair=True))
    assert (info["f_col"], info["sigf_col"], info["n_amplitude_pairs"]) == ("F-obs", "SIGF-obs", 2)
    assert (info["free_col"], info["free_value"], info["free_convention"]) == ("R-free-flags", 1, "phenix")
    assert 0.05 < info["n_free"] / info["n_refl"] < 0.15
    assert info["spacegroup"] == "P 21 21 21"
    info = refine.inspect_mtz(str(tmp_path / "a.mtz"), f_col="FP")
    assert (info["f_col"], info["sigf_col"]) == ("FP", "SIGFP")
    with pytest.raises(refine.RefineError):
        refine.inspect_mtz(str(tmp_path / "a.mtz"), f_col="FMODEL")
    info = refine.inspect_mtz(write_dataset(tmp_path / "b.mtz", cell, sg, free="ccp4"))
    assert (info["free_col"], info["free_value"], info["free_convention"]) == ("FreeR_flag", 0, "ccp4")
    info = refine.inspect_mtz(write_dataset(tmp_path / "c.mtz", cell, sg, free="single"))
    assert info["free_col"] is None and "single value" in info["free_note"]
    info = refine.inspect_mtz(write_dataset(tmp_path / "d.mtz", cell, sg, free=None, misym=True))
    assert info["free_col"] is None and info["n_amplitude_pairs"] == 1


def test_parse_phenix_log():
    text = textwrap.dedent("""\
        Start R-work = 0.3000, R-free = 0.3200
        Final R-work = 0.2500, R-free = 0.2800
        ... second run in the same log ...
        Start R-work = 0.2500, R-free = 0.2800
        Final R-work = 0.1987, R-free = 0.2345
        """)
    r = refine.parse_phenix_log(text)
    assert (r["Rw_start"], r["Rf_start"], r["Rw_final"], r["Rf_final"]) == (0.25, 0.28, 0.1987, 0.2345)
    cycles = textwrap.dedent("""\
        | r_work = 0.2600 r_free = 0.2900 bonds = 0.010 |
        | r_work = 0.2400 r_free = 0.2700 bonds = 0.008 |
        | r_work = 0.2200 r_free = 0.2500 bonds = 0.007 |
        """)
    r = refine.parse_phenix_log(cycles)
    assert (r["Rw_start"], r["Rf_start"], r["Rw_final"], r["Rf_final"]) == (0.26, 0.29, 0.22, 0.25)
    assert all(math.isnan(v) for v in refine.parse_phenix_log("nothing useful").values())


def test_parse_model_r_factors():
    refmac = textwrap.dedent("""\
        REMARK   3  FIT TO DATA USED IN REFINEMENT.
        REMARK   3   CROSS-VALIDATION METHOD          : THROUGHOUT
        REMARK   3   FREE R VALUE TEST SET SELECTION  : RANDOM
        REMARK   3   R VALUE     (WORKING + TEST SET) : 0.24692
        REMARK   3   R VALUE            (WORKING SET) :  0.24588
        REMARK   3   FREE R VALUE                     :  0.27265
        REMARK   3   FREE R VALUE TEST SET SIZE   (%) : 5.0
        """)
    assert refine.parse_model_r_factors(refmac) == {"Rw_final": 0.24588, "Rf_final": 0.27265}
    phenix = ("REMARK   3   R VALUE                   (WORKING SET) : 0.1717\n"
              "REMARK   3   FREE R VALUE                            : 0.2089\n")
    assert refine.parse_model_r_factors(phenix) == {"Rw_final": 0.1717, "Rf_final": 0.2089}
    cif = "_refine.ls_R_factor_R_work 0.1717\n_refine.ls_R_factor_R_free 0.2089\n"
    assert refine.parse_model_r_factors(cif) == {"Rw_final": 0.1717, "Rf_final": 0.2089}
    assert all(math.isnan(v) for v in refine.parse_model_r_factors("HEADER x\n").values())


def test_parse_dimple_output():
    # verbatim from a dimple 2.7 run (CCP4 9) on PTP1B-y0001
    screen = textwrap.dedent("""\
          3 refmac5 rigid    5.3s  10/10   R  0.2111  ->  0.2097
        No MR for R < 0.4.
          4 refmacat jelly  28.0s   4/4   R/Rfree  0.2756/0.2920  ->  0.2603/0.2823
            RMS:   bond 0.010 -> 0.012   angle 1.47 -> 1.73   chiral 0.09 -> 0.14
          5 refmacat restr  31.3s   8/8   R/Rfree  0.2543/0.2789  ->  0.2459/0.2726
        """)
    r = refine.parse_dimple_output(screen)
    assert (r["Rw_start"], r["Rf_start"], r["Rw_final"], r["Rf_final"]) == (0.2543, 0.2789, 0.2459, 0.2726)
    log = "overall_r: 0.2097\nini_overall_r: 0.2111\noverall_r: 0.2459\nfree_r: 0.2726\n"
    r = refine.parse_dimple_output(log)
    assert (r["Rw_final"], r["Rf_final"]) == (0.2459, 0.2726)


def test_prepare_eff(tmp_path):
    path, notes = refine.prepare_eff(None, str(tmp_path))
    assert path == str(tmp_path / "phenix_refine.eff")
    text = open(path).read()
    packaged = open(refine.packaged_eff_path()).read()
    if "/n/holyscratch01" in packaged:  # the authors' cluster path is dropped here
        assert "/n/holyscratch01" not in text and any("holyscratch" in n for n in notes)
    assert "refinement" in text
    user = tmp_path / "mine.eff"
    user.write_text("refinement.main.number_of_macro_cycles = 3\n")
    path, notes = refine.prepare_eff(str(user), str(tmp_path))
    assert "number_of_macro_cycles = 3" in open(path).read() and notes == []
    user.write_text('refinement.input.xray_data.r_free_flags.file_name = "/no/such/file.mtz"\n')
    with pytest.raises(refine.RefineError, match="/no/such/file.mtz"):
        refine.prepare_eff(str(user), str(tmp_path))
    with pytest.raises(refine.RefineError):
        refine.prepare_eff(str(tmp_path / "missing.eff"), str(tmp_path))


# ---------------------------------------------------------------------------
# Indexing frames
# ---------------------------------------------------------------------------

def test_normalising_variant():
    cell, sg = TRIGONAL
    assert refine.normalising_variant(gemmi.Op("-x,-y,z"), sg).triplet() == "-x,-y,z"
    i41 = gemmi.SpaceGroup("I 41")
    v = refine.normalising_variant(gemmi.Op("-x,y,-z"), i41)  # gemmi's twin law for I 41: needs another coset member
    assert v is not None and v.triplet() != "-x,y,-z"
    # P 21 21 21 with a = b: swapping a and b needs an origin shift
    assert refine.normalising_variant(gemmi.Op("-y,-x,-z"), gemmi.SpaceGroup("P 21 21 21")) is None


def test_reindex_model_uses_a_safe_operator(tmp_path):
    """A model moved with the operator that re-indexed the data must fit those data.

    Upstream's reindex applies gemmi's twin law to the amplitudes; for I 41 that
    operator (-x,y,-z) does not map the space group onto itself, so the model has
    to be moved with another member of its coset (same amplitudes).
    """
    cell, sg = gemmi.UnitCell(60, 60, 90, 90, 90, 90), gemmi.SpaceGroup("I 41")
    pdb = write_model(tmp_path / "i41.pdb", cell, sg, positions=ATOMS[:3])
    mtz = tmp_path / "i41.mtz"
    H = asu_hkl(cell, sg, 4.0)
    amp, phi = point_coefficients(H, cell, sg, ATOMS[:3])
    rs.DataSet({"H": H[:, 0], "K": H[:, 1], "L": H[:, 2], "FC": amp, "PHIC": phi}, cell=cell, spacegroup=sg,
               merged=True).astype({"H": "HKL", "K": "HKL", "L": "HKL", "FC": "SFAmplitude", "PHIC": "Phase"}) \
        .set_index(["H", "K", "L"]).write_mtz(str(mtz))
    twin = gemmi.find_twin_laws(cell, sg, 3.0, False)[0]
    re_mtz, re_pdb = tmp_path / "re.mtz", tmp_path / "re.pdb"
    refine.reindex_mtz(str(mtz), str(re_mtz), twin)          # what upstream reindex does to the amplitudes
    used = refine.reindex_model(pdb, str(re_pdb), twin)
    assert refine.normalising_variant(used, sg).triplet() == used.triplet()
    ds = rs.read_mtz(str(re_mtz))
    observed = ds["FC"].to_numpy(float)
    model_amp, _ = point_coefficients(ds.get_hkls(), cell, sg, [tuple(r) for r in coords(re_pdb)])
    assert np.corrcoef(observed, model_amp)[0, 1] > 0.999
    if twin.triplet() != used.triplet():  # the raw twin law moves the model off the data
        raw = tmp_path / "raw.pdb"
        refine.reindex_model(pdb, str(raw), twin, normalise=False)
        raw_amp, _ = point_coefficients(ds.get_hkls(), cell, sg, [tuple(r) for r in coords(raw)])
        assert np.corrcoef(observed, raw_amp)[0, 1] < 0.99
    ortho = write_model(tmp_path / "p212121.pdb", gemmi.UnitCell(50, 50, 80, 90, 90, 90),
                        gemmi.SpaceGroup("P 21 21 21"))
    with pytest.raises(ValueError, match="origin shift"):
        refine.reindex_model(ortho, str(tmp_path / "x.pdb"), "-y,-x,-z")


def test_find_frame_op(tmp_path):
    cell, sg = TRIGONAL
    src = write_dataset(tmp_path / "in.mtz", cell, sg, free=None)
    op, cc, ccs = refine.find_frame_op(src, src, "F-obs")
    assert op.triplet() == "x,y,z" and cc > 0.999
    other = tmp_path / "other.mtz"
    rs.read_mtz(src).apply_symop(gemmi.Op("-x,-y,z")).hkl_to_asu().write_mtz(str(other))
    op, cc, ccs = refine.find_frame_op(src, str(other), "F-obs")
    assert op.triplet() == "-x,-y,z" and cc > 0.999 and ccs["x,y,z"] < 0.5
    unrelated = write_dataset(tmp_path / "unrelated.mtz", cell, sg, free=None, seed=5)
    op, cc, ccs = refine.find_frame_op(src, unrelated, "F-obs")
    assert op is None and cc < 0.5


def test_find_frame_op_accepts_rescaled_amplitudes(tmp_path):
    """Found on Wynton with real dimple 2.7 output (CCP4 9): refmac writes the observed amplitudes
    after anisotropic scaling, so CC(input F, refined F) in the right frame was 0.977-0.9998 on 39
    PTP1B datasets (0.473 in the wrong frame). The old fixed 0.98 cut-off failed a valid refinement
    (PTP1B-y0264, CC 0.977); now an absolute floor plus a margin over the other operators decides."""
    assert refine._choose_frame({"x,y,z": 0.977, "-x,-y,z": 0.473}) == "x,y,z"
    assert refine._choose_frame({"x,y,z": 0.30, "-x,-y,z": 0.999}) == "-x,-y,z"
    assert refine._choose_frame({"x,y,z": 0.85, "-x,-y,z": 0.20}) is None       # below the floor
    assert refine._choose_frame({"x,y,z": 0.97, "-x,-y,z": 0.93}) is None       # ambiguous
    assert refine._choose_frame({"x,y,z": 0.95}) == "x,y,z"                      # no alternative
    assert refine._choose_frame({"x,y,z": float("nan")}) is None and refine._choose_frame({}) is None
    # through find_frame_op: amplitudes with an anisotropic scale (as refmac writes them)
    cell, sg = TRIGONAL
    src = write_dataset(tmp_path / "in.mtz", cell, sg, free=None)
    ds = rs.read_mtz(src)
    h = ds.get_hkls().astype(float)
    aniso = 30.0 * np.exp(-0.006 * (h[:, 0] ** 2 + h[:, 1] ** 2) - 0.0015 * h[:, 2] ** 2)
    ds["F-obs"] = rs.DataSeries(ds["F-obs"].to_numpy(dtype=float) * aniso, index=ds.index, dtype="SFAmplitude")
    scaled = tmp_path / "scaled.mtz"
    ds.write_mtz(str(scaled))
    op, cc, ccs = refine.find_frame_op(src, str(scaled), "F-obs")
    assert 0.9 < ccs["x,y,z"] < 0.98 and ccs["-x,-y,z"] < 0.6   # below the old cut-off, yet clear
    assert op is not None and op.triplet() == "x,y,z" and cc == ccs["x,y,z"]


def test_reindex_model_convention(tmp_path):
    """x' = op^-1 x keeps every structure-factor phase: (h R) . x' == h . x (mod 1)."""
    cell = gemmi.UnitCell(40, 40, 50, 90, 90, 120)
    src = write_model(tmp_path / "m.pdb", cell, gemmi.SpaceGroup("P 3"), aniso=True)
    op = gemmi.Op("-y,x-y,z")  # a three-fold: not its own inverse, so the direction matters
    dest = tmp_path / "m2.pdb"
    refine.reindex_model(src, str(dest), op)
    before = gemmi.read_structure(src)
    after = gemmi.read_structure(str(dest))
    rng = np.random.default_rng(1)
    hkl = rng.integers(-6, 7, size=(20, 3))
    for a0, a1 in zip(before[0]["A"], after[0]["A"]):
        f0 = before.cell.fractionalize(a0[0].pos)
        f1 = after.cell.fractionalize(a1[0].pos)
        for h in hkl:
            h2 = np.array(op.apply_to_hkl([int(x) for x in h]))
            d = np.dot(h2, [f1.x, f1.y, f1.z]) - np.dot(h, [f0.x, f0.y, f0.z])
            assert abs(d - round(d)) < 1e-3
        u0, u1 = a0[0].aniso, a1[0].aniso
        assert abs((u0.u11 + u0.u22 + u0.u33) - (u1.u11 + u1.u22 + u1.u33)) < 1e-3
    back = tmp_path / "m3.pdb"
    refine.reindex_model(str(dest), str(back), op.inverse())
    assert np.allclose(coords(back), coords(src), atol=2e-3)


# ---------------------------------------------------------------------------
# refine_datasets with fake programs
# ---------------------------------------------------------------------------

def _jobs(tmp_path, specs):
    cell, sg = TRIGONAL
    jobs = []
    for stem, kw in specs:
        mtz = write_dataset(tmp_path / f"in_{stem}.mtz", cell, sg, seed=len(jobs), **kw)
        pdb = write_model(tmp_path / f"in_{stem}.pdb", cell, sg)
        jobs.append({"stem": stem, "mtz": mtz, "pdb": pdb})
    return jobs


def test_dimple_end_to_end_and_resume(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free="phenix")), ("0002_1", dict(free=None, misym=True))])
    out = tmp_path / "04_refine"
    messages = []
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=2, log=messages.append)
    assert list(table.columns[:len(refine.SUMMARY_COLUMNS)]) == refine.SUMMARY_COLUMNS
    assert table["status"].tolist() == ["ok", "ok"]
    assert table["file_idx"].tolist() == ["0001", "0002"] and table["symop"].tolist() == ["0", "1"]
    row = table.set_index("stem").loc["0001_0"]
    assert (row["Rw_final"], row["Rf_final"]) == (0.1866, 0.2245)     # REMARK 3 of final.pdb
    assert (row["Rw_start"], row["Rf_start"]) == (0.25, 0.27)          # dimple screen output
    assert row["free_col"] == "R-free-flags" and row["reindex_op"] == "" and not row["skipped"]
    for stem in ("0001_0", "0002_1"):
        assert (out / "phases" / f"{stem}.mtz").is_file() and (out / "models" / f"{stem}.pdb").is_file()
        cols = refine.inspect_mtz(str(out / "phases" / f"{stem}.mtz"))["columns"]
        assert {"FWT", "PHWT", "DELFWT", "PHDELWT"} <= set(cols)
        clean = gemmi.read_mtz_file(str(out / "runs" / stem / "input.mtz"))
        labels = [c.label for c in clean.columns]
        assert "M/ISYM" not in labels and labels[3:5] == ["F", "SIGF"]
    calls = {c["stem"]: c for c in fakes.calls()}
    a1 = calls["0001_0"]["argv"]
    assert a1[0] == str(out / "runs" / "0001_0" / "input.mtz") and a1[1] == jobs[0]["pdb"]
    assert a1[2] == str(out / "runs" / "0001_0" / "dimple")
    assert a1[3:] == ["--fcolumn", "F", "--sigfcolumn", "SIGF", "--free-r-flags", a1[0], "--freecolumn",
                      "FreeR_flag=1"]
    assert calls["0002_1"]["argv"][3:] == ["--fcolumn", "F", "--sigfcolumn", "SIGF"]  # no flags: dimple makes them
    assert calls["0001_0"]["omp"] == "1"
    summary = pd.read_csv(out / "refine_summary.csv", dtype={"file_idx": str, "symop": str})
    assert summary["stem"].tolist() == ["0001_0", "0002_1"] and summary["Rf_final"].notna().all()
    assert len(pd.read_csv(out / "refine_summary_ok.csv")) == 2
    assert any("done: 2/2 refined" in m for m in messages)

    # unchanged inputs: nothing runs again
    n_calls = len(fakes.calls())
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=2, log=quiet)
    assert table["skipped"].tolist() == [True, True] and table["status"].tolist() == ["ok", "ok"]
    assert table.set_index("stem").loc["0001_0", "Rf_final"] == 0.2245
    assert len(fakes.calls()) == n_calls
    # a changed input is refined again, the other one is reused
    time.sleep(0.01)
    write_dataset(jobs[1]["mtz"], *TRIGONAL, seed=42, free=None)
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=2, log=quiet)
    assert table["skipped"].tolist() == [True, False]
    assert [c["stem"] for c in fakes.calls()[n_calls:]] == ["0002_1"]


def test_dimple_reindexed_outputs_are_transformed_back(tmp_path, fakes, monkeypatch):
    jobs = _jobs(tmp_path, [("0003_1", dict(free="phenix"))])
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    truth = write_truth(truth_dir / "0003_1.mtz", jobs[0]["mtz"], ATOMS[:3])
    monkeypatch.setenv("FAKE_TRUTH_DIR", str(truth_dir))
    fakes.behave(**{"0003_1": "reindex"})
    out = tmp_path / "04_refine"
    table = refine.refine_datasets(jobs, "dimple", str(out), log=quiet)
    row = table.iloc[0]
    assert row["status"] == "ok", row["reason"]
    assert row["reindex_op"] == "-x,-y,z" and "re-indexed" in row["reason"]
    phases = out / "phases" / "0003_1.mtz"
    # the data columns are back in the input setting ...
    op, cc, _ = refine.find_frame_op(jobs[0]["mtz"], str(phases), "F-obs", f_refined="F")
    assert op.triplet() == "x,y,z" and cc > 0.999
    # ... and so are the phases (complex correlation with the true coefficients) ...
    cc, n = complex_cc(phases, "FWT", "PHWT", truth, "FT", "PHT")
    assert cc > 0.999 and n > 100
    # ... and the model
    assert np.allclose(coords(out / "models" / "0003_1.pdb"), coords(jobs[0]["pdb"]), atol=3e-3)


def test_failures_are_recorded_not_raised(tmp_path, fakes):
    specs = [(s, dict(free="phenix")) for s in ("0001_0", "0002_0", "0003_0", "0004_0", "0005_0", "0006_0",
                                                 "0008_0")]
    jobs = _jobs(tmp_path, specs)
    jobs.append({"stem": "0007_0", "mtz": str(tmp_path / "missing.mtz"), "pdb": jobs[0]["pdb"]})
    fakes.behave(**{"0002_0": "fail", "0003_0": "nooutput", "0004_0": "nomaps", "0005_0": "newsg",
                    "0006_0": "nor", "0008_0": "mr"})
    out = tmp_path / "04_refine"
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=3, log=quiet).set_index("stem")
    assert table.loc["0001_0", "status"] == "ok"
    reasons = {s: table.loc[s, "reason"] for s in table.index}
    assert table.loc["0002_0", "status"] == "failed" and "exited with code 2" in reasons["0002_0"]
    assert "Sorry: fake refinement failed on purpose" in reasons["0002_0"]
    assert "wrote no refined MTZ" in reasons["0003_0"]
    assert "lacks the map coefficients" in reasons["0004_0"]
    assert "space group P 32 2 1" in reasons["0005_0"]
    assert table.loc["0006_0", "status"] == "ok" and math.isnan(table.loc["0006_0", "Rf_final"])
    assert "R-free could not be read" in reasons["0006_0"]
    assert table.loc["0007_0", "status"] == "failed" and "input MTZ not found" in reasons["0007_0"]
    assert table.loc["0008_0", "status"] == "ok" and "molecular replacement" in reasons["0008_0"]
    assert reasons["0001_0"] == ""
    assert (out / "runs" / "0002_0" / "reason.txt").read_text().startswith("dimple exited with code 2")
    ok = pd.read_csv(out / "refine_summary_ok.csv")
    assert ok["stem"].tolist() == ["0001_0", "0008_0"]  # 0006_0 refined, but without an R-free the filter cannot use it
    # failed datasets are retried on the next call; the good ones are reused
    fakes.behave()
    n_calls = len(fakes.calls())
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=3, log=quiet).set_index("stem")
    assert table.loc["0002_0", "status"] == "ok" and table.loc["0001_0", "skipped"]
    rerun = sorted(c["stem"] for c in fakes.calls()[n_calls:])
    assert rerun == ["0002_0", "0003_0", "0004_0", "0005_0"]  # 0006_0 is ok (without R-free): reused


def test_timeout_kills_the_program(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free="phenix"))])
    fakes.behave(**{"0001_0": "sleep"})
    t0 = time.time()
    table = refine.refine_datasets(jobs, "dimple", str(tmp_path / "out"), timeout_min=0.05, log=quiet)
    assert time.time() - t0 < 40
    assert table.iloc[0]["status"] == "failed" and "timed out after 0.05 min" in table.iloc[0]["reason"]


def test_errors_that_stop_everything(tmp_path, fakes, monkeypatch):
    jobs = _jobs(tmp_path, [("0001_0", dict(free="phenix"))])
    with pytest.raises(refine.RefineError, match="unknown refinement program"):
        refine.refine_datasets(jobs, "buster", str(tmp_path / "o"), log=quiet)
    with pytest.raises(refine.RefineError, match="duplicate stem"):
        refine.refine_datasets(jobs + jobs, "dimple", str(tmp_path / "o"), log=quiet)
    with pytest.raises(refine.RefineError, match="lacks"):
        refine.refine_datasets([{"stem": "x", "mtz": jobs[0]["mtz"]}], "dimple", str(tmp_path / "o"), log=quiet)
    with pytest.raises(refine.RefineError, match="bad stem"):
        refine.refine_datasets([dict(jobs[0], stem="a/b")], "dimple", str(tmp_path / "o"), log=quiet)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(refine.RefineError, match="ccp4.setup-sh"):
        refine.refine_datasets(jobs, "dimple", str(tmp_path / "o"), log=quiet)
    with pytest.raises(refine.RefineError, match="phenix_env.sh"):
        refine.refine_datasets(jobs, "phenix", str(tmp_path / "o"), log=quiet)


def test_foreign_run_folder_is_not_deleted(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free="phenix"))])
    out = tmp_path / "out"
    (out / "runs" / "0001_0").mkdir(parents=True)
    (out / "runs" / "0001_0" / "precious.txt").write_text("keep me")
    table = refine.refine_datasets(jobs, "dimple", str(out), log=quiet)
    assert table.iloc[0]["status"] == "failed" and "not created by valdo-easy" in table.iloc[0]["reason"]
    assert "internal error" not in table.iloc[0]["reason"]
    assert sorted(os.listdir(out / "runs" / "0001_0")) == ["precious.txt"]
    assert (out / "runs" / "0001_0" / "precious.txt").read_text() == "keep me"


def test_phenix_command_with_two_free_columns_and_user_labels(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free="both"))])
    out = tmp_path / "04_refine"
    table = refine.refine_datasets(jobs, "phenix", str(out), log=quiet)
    assert table["status"].tolist() == ["ok"]
    argv = fakes.calls()[-1]["argv"]
    # several R-free columns: the one valdo chose is named too (PHENIX would stop at the choice)
    assert [a for a in argv if a.startswith("miller_array.labels.name=")] == [
        "miller_array.labels.name=F-obs,SIGF-obs", "miller_array.labels.name=R-free-flags"]
    # labels or the hexdigest setting given by the user are not given twice
    refine.refine_datasets(jobs, "phenix", str(tmp_path / "mine"), log=quiet,
                           extra_args=["miller_array.labels.name=F-obs,SIGF-obs",
                                       "xray_data.r_free_flags.ignore_pdb_hexdigest=False"])
    argv = fakes.calls()[-1]["argv"]
    assert sum(a.startswith("miller_array.labels.name=") for a in argv) == 1
    assert [a for a in argv if "ignore_pdb_hexdigest" in a] == ["xray_data.r_free_flags.ignore_pdb_hexdigest=False"]
    assert argv[-1] == "--overwrite"


def test_phenix_identity_rule():
    """Found on Wynton with real phenix.refine 1.21.2 and 2.2.1 (no twin law): PTP1B dataset 0006 of the
    40-dataset validation project failed the frame check with 'x,y,z: 1.000, -x,-y,z: 0.921' (its own,
    mildly twinned amplitudes fit the twin law too), while dimple kept it. phenix.refine never re-indexes."""
    assert refine._phenix_identity({"x,y,z": 1.0, "-x,-y,z": 0.921}) == {"-x,-y,z": 0.921}
    assert refine._phenix_identity({"x,y,z": 0.95, "-x,-y,z": 0.40}) == {}
    # a twin-law CC below 0.9 that still blocks the 0.1 margin is reported (and so gets the twin hint)
    assert refine._phenix_identity({"x,y,z": 0.95, "-x,-y,z": 0.86}) == {"-x,-y,z": 0.86}
    assert refine._phenix_identity({"x,y,z": 0.85, "-x,-y,z": 0.20}) is None             # below the floor
    assert refine._phenix_identity({"x,y,z": 0.93, "-x,-y,z": 0.97}) is None             # another one fits better
    assert refine._phenix_identity({"x,y,z": float("nan"), "-x,-y,z": 0.99}) is None
    assert refine._phenix_identity({}) is None


def _twin_amplitudes(path, fraction):
    """Rewrite F-obs of an MTZ as merohedrally twinned data (twin law -h,-k,l, intensities mixed)."""
    ds = rs.read_mtz(str(path))
    mate = ds.apply_symop(gemmi.Op("-x,-y,z")).hkl_to_asu()[["F-obs"]].rename(columns={"F-obs": "F_mate"})
    ds = ds.join(mate, how="inner")
    f1, f2 = ds["F-obs"].to_numpy(dtype=float), ds["F_mate"].to_numpy(dtype=float)
    ds["F-obs"] = rs.DataSeries(np.sqrt((1 - fraction) * f1 ** 2 + fraction * f2 ** 2), index=ds.index,
                                dtype="SFAmplitude")
    ds.drop(columns=["F_mate"]).write_mtz(str(path))


def test_phenix_keeps_the_input_indexing_of_twinned_data(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free=None)), ("0002_0", dict(free=None))])
    for job in jobs:
        _twin_amplitudes(job["mtz"], 0.42)
    op, cc, ccs = refine.find_frame_op(jobs[0]["mtz"], jobs[0]["mtz"], "F-obs")
    assert op is None and ccs["x,y,z"] > 0.999 and 0.9 < ccs["-x,-y,z"] < 0.999   # ambiguous by the margin rule
    table = refine.refine_datasets(jobs[:1], "phenix", str(tmp_path / "phx"), ncpu=1, log=quiet).set_index("stem")
    row = table.loc["0001_0"]
    assert row["status"] == "ok" and row["reindex_op"] == "" and refine.TWIN_HINT in row["reason"]
    assert "phenix.refine keeps the input indexing (x,y,z, CC 1.000)" in row["reason"]
    assert (tmp_path / "phx" / "phases" / "0001_0.mtz").is_file()
    # dimple can re-index (pointless), so the same data still fail its frame check
    table = refine.refine_datasets(jobs[1:], "dimple", str(tmp_path / "dim"), ncpu=1, log=quiet).set_index("stem")
    assert table.loc["0002_0", "status"] == "failed" and refine.TWIN_HINT in table.loc["0002_0", "reason"]


def test_phenix_end_to_end(tmp_path, fakes):
    jobs = _jobs(tmp_path, [("0001_0", dict(free=None)), ("0002_0", dict(free="phenix", extra_pair=True))])
    out = tmp_path / "04_refine"
    table = refine.refine_datasets(jobs, "phenix", str(out), ncpu=2, extra_args=["main.number_of_macro_cycles=1"],
                                   log=quiet).set_index("stem")
    assert table["status"].tolist() == ["ok", "ok"]
    assert (table.loc["0001_0", "Rw_final"], table.loc["0001_0", "Rf_final"]) == (0.2, 0.23)  # log beats REMARK
    assert (table.loc["0001_0", "Rw_start"], table.loc["0001_0", "Rf_start"]) == (0.25, 0.27)
    eff = out / "phenix_refine.eff"
    assert eff.is_file() and "/n/holyscratch01" not in eff.read_text()
    calls = {c["stem"]: c for c in fakes.calls()}
    a1 = calls["0001_0"]["argv"]
    # PHENIX 1.21 and 2.2: data labels with miller_array.labels.name (always: an MTZ can hold intensities
    # too, which PHENIX would prefer), the start model's R-free hexdigest ignored, '--overwrite' last
    # (PHENIX 1.x, Python 2.7 argparse, rejects parameters after an option)
    assert a1[:5] == [jobs[0]["pdb"], jobs[0]["mtz"], "output.prefix=refine_0001_0", str(eff),
                      "miller_array.labels.name=F-obs,SIGF-obs"]
    assert "xray_data.r_free_flags.generate=True" in a1 and not any(a.startswith("xray_data.labels") for a in a1)
    assert "xray_data.r_free_flags.ignore_pdb_hexdigest=True" in a1
    assert a1[-2:] == ["main.number_of_macro_cycles=1", "--overwrite"]
    assert [a for a in a1[1:] if a.startswith("-")] == ["--overwrite"]
    assert calls["0001_0"]["cwd"] == str(out / "runs" / "0001_0")
    a2 = calls["0002_0"]["argv"]
    assert [a for a in a2 if a.startswith("miller_array.labels.name=")] == ["miller_array.labels.name=F-obs,SIGF-obs"]
    assert not any("r_free_flags.generate" in a or "r_free_flags.label" in a for a in a2)
    for stem in ("0001_0", "0002_0"):
        cols = refine.inspect_mtz(str(out / "phases" / f"{stem}.mtz"))["columns"]
        assert {"2FOFCWT", "PH2FOFCWT", "FOFCWT", "PHFOFCWT"} <= set(cols)
    # a different .eff is a different refinement: everything runs again
    n_calls = len(fakes.calls())
    user = tmp_path / "mine.eff"
    user.write_text("refinement.main.number_of_macro_cycles = 2\n")
    table = refine.refine_datasets(jobs, "phenix", str(out), ncpu=2, eff=str(user),
                                   extra_args=["main.number_of_macro_cycles=1"], log=quiet)
    assert not table["skipped"].any() and len(fakes.calls()) == n_calls + 2
