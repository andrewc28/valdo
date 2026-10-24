"""Phases and models of a re-indexed dataset are moved by one and the same operator (review F2).

Upstream reindex moves a dataset's amplitudes with a twin law from gemmi.find_twin_laws. Any
member of that law's point-group coset gives the same amplitudes, but phases and models may only
be moved by a member that maps the space group onto itself (refine.normalising_variant). For
I 41, gemmi's law -x,y,-z is not such a member (y,x,-z is). The phase file and the dataset's own
model must therefore both be moved by the safe member: otherwise the model does not sit in the
map of the phases it is used with. Small synthetic structures; structure factors from gemmi.
"""

import os

import numpy as np
import pytest

from conftest import needs_science

pytestmark = needs_science

COLUMNS = ["FWT", "PHWT", "DELFWT", "PHDELWT"]


def make_structure(hm, cell, frac):
    import gemmi
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = hm
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, f in enumerate(frac):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(i + 1, " ")
        for name, el, shift in (("CA", "C", (0, 0, 0)), ("N", "N", (0.012, 0, 0)), ("O", "O", (0, 0.014, 0))):
            atom = gemmi.Atom()
            atom.name, atom.element, atom.occ, atom.b_iso = name, gemmi.Element(el), 1.0, 20.0
            atom.pos = st.cell.orthogonalize(gemmi.Fractional(*(np.array(f) + shift)))
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


def coefficients(st, dmin=3.0):
    """rs.DataSet with FWT/PHWT (the model's structure factors) and DELFWT/PHDELWT."""
    import gemmi
    import reciprocalspaceship as rs
    sg = gemmi.SpaceGroup(st.spacegroup_hm)
    hkl = gemmi.make_miller_array(st.cell, sg, dmin, 0)
    calc = gemmi.StructureFactorCalculatorX(st.cell)
    st.setup_cell_images()
    f = np.array([calc.calculate_sf_from_model(st[0], [int(x) for x in h]) for h in hkl])
    ds = rs.DataSet({c: rs.DataSeries(hkl[:, j], dtype="HKL") for j, c in enumerate("HKL")}, cell=st.cell,
                    spacegroup=sg, merged=True).set_index(["H", "K", "L"])
    ds["FWT"] = rs.DataSeries(np.abs(f), index=ds.index, dtype="SFAmplitude")
    ds["PHWT"] = rs.DataSeries(np.angle(f, deg=True), index=ds.index, dtype="Phase")
    ds["DELFWT"] = rs.DataSeries(0.2 * np.abs(f), index=ds.index, dtype="SFAmplitude")
    ds["PHDELWT"] = rs.DataSeries(np.angle(f, deg=True), index=ds.index, dtype="Phase")
    return ds


def phase_agreement(a, b):
    """(CC of FWT, median |phase difference| in degrees over the stronger half) of two data sets."""
    j = a[["FWT", "PHWT"]].join(b[["FWT", "PHWT"]], how="inner", rsuffix="_b")
    za = j["FWT"].to_numpy(float) * np.exp(1j * np.deg2rad(j["PHWT"].to_numpy(float)))
    zb = j["FWT_b"].to_numpy(float) * np.exp(1j * np.deg2rad(j["PHWT_b"].to_numpy(float)))
    strong = np.abs(za) > np.median(np.abs(za))
    dphi = np.rad2deg(np.abs(np.angle(za[strong] * np.conj(zb[strong]))))
    return float(np.corrcoef(np.abs(za), np.abs(zb))[0, 1]), float(np.median(dphi))


def _job(tmp_path, st, raw_op):
    import gemmi
    src = coefficients(st)
    src.write_mtz(str(tmp_path / "phases_in.mtz"))
    st.write_pdb(str(tmp_path / "model_in.pdb"))
    data = src[["FWT"]].rename(columns={"FWT": "F-obs"}).apply_symop(gemmi.Op(raw_op)).hkl_to_asu()
    data = data[[c for c in data.columns if c != "M/ISYM"]]
    data.write_mtz(str(tmp_path / "data.mtz"))          # the amplitudes as upstream reindex left them
    return {"id": "0002", "stem": "0002_1", "phwt": "PHWT", "phdelwt": "PHDELWT", "data": str(tmp_path / "data.mtz"),
            "model_src": str(tmp_path / "model_in.pdb"), "model_dest": str(tmp_path / "model_out.pdb"),
            "model_op": raw_op, "phase_src": str(tmp_path / "phases_in.mtz"),
            "phase_dest": str(tmp_path / "phases_out.mtz"), "columns": COLUMNS, "op": raw_op, "transform": True,
            "phase_ready": None, "model_ready": None, "fwt": "FWT"}


def test_phase_operator_choices():
    import gemmi
    from valdo.easy import stages
    assert stages.phase_operator("x,y,z", gemmi.SpaceGroup("I 41")).triplet() == "x,y,z"
    assert stages.phase_operator("-x,y,-z", gemmi.SpaceGroup("I 41")).triplet() == "y,x,-z"
    assert stages.phase_operator("-x,-y,z", gemmi.SpaceGroup("P 31 2 1")).triplet() == "-x,-y,z"  # PTP1B: unchanged
    assert stages.phase_operator("y,x,-z", gemmi.SpaceGroup("P 21 21 21")) is None   # a = b: origin shift


def test_phases_and_model_move_together_in_I41(tmp_path):
    import gemmi
    import reciprocalspaceship as rs
    from valdo.easy import stages
    rng = np.random.default_rng(3)
    raw = "-x,y,-z"
    st = make_structure("I 41", (70, 70, 100, 90, 90, 90), rng.random((12, 3)))
    job = _job(tmp_path, st, raw)
    out = stages.prepare_dataset(job)
    assert out["drop"] is None, out
    assert out["phase_op"] == "y,x,-z" and out["coverage"] == 1.0
    assert out["model_fit"] is not None and out["model_fit"] >= stages.MODEL_FIT_MIN
    assert "model_map_mismatch" not in out["flags"]
    # the phases written equal the structure factors of the model written (same frame) ...
    moved_model = gemmi.read_structure(job["model_dest"])
    cc, dphi = phase_agreement(coefficients(moved_model), rs.read_mtz(job["phase_dest"]))
    assert cc > 0.999 and dphi < 2.0
    # ... which the raw twin law would not have given (what valdo-easy did before)
    src = rs.read_mtz(job["phase_src"])
    raw_moved = src.apply_symop(gemmi.Op(raw)).hkl_to_asu()
    cc_raw, dphi_raw = phase_agreement(coefficients(moved_model), raw_moved)
    assert cc_raw > 0.999 and dphi_raw > 30.0       # same amplitudes, different phases
    # the amplitudes are those of upstream's reindexing with the raw law
    data = rs.read_mtz(job["data"])
    j = data.join(rs.read_mtz(job["phase_dest"])[["FWT"]], how="inner")
    assert len(j) == len(data) and np.allclose(j["F-obs"].to_numpy(float), j["FWT"].to_numpy(float), rtol=1e-4)


def test_fallback_model_is_not_moved(tmp_path):
    """The apo fallback model is in the reference's setting already: it stays as it is."""
    from valdo.easy import stages
    rng = np.random.default_rng(4)
    st = make_structure("I 41", (70, 70, 100, 90, 90, 90), rng.random((6, 3)))
    job = _job(tmp_path, st, "-x,y,-z")
    job["model_op"] = stages.IDENTITY
    out = stages.prepare_dataset(job)
    assert out["drop"] is None and out["phase_op"] == "y,x,-z"
    assert os.path.realpath(job["model_dest"]) == os.path.realpath(job["model_src"])   # a symlink


def test_origin_shift_cases_are_dropped_with_a_reason(tmp_path):
    rng = np.random.default_rng(5)
    from valdo.easy import stages
    st = make_structure("P 21 21 21", (60, 60, 90, 90, 90, 90), rng.random((6, 3)))
    job = _job(tmp_path, st, "y,x,-z")
    out = stages.prepare_dataset(job)
    assert out["drop"] and "origin shift" in out["drop"] and "phases.source: refine" in out["drop"]
    assert not os.path.exists(job["phase_dest"])
