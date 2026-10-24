"""Tests for valdo.easy.report (the HTML/Markdown/JSON report, hit maps, viewer scripts)
and valdo.easy.maps (map files and structure helpers).

A small synthetic work folder is built in the layout valdo-easy writes
(datasets.csv, valdo_state.json, configs/, 10_phases_blobs/mtz, 11_tag,
inputs/models, a known bound model) with blob tables shaped like upstream's
(sample, peakz, peak, score, cenx/y/z, volume, radius, fractional, ...,
focal tag, duplicate, ligand). The phased MTZ files hold difference-map
coefficients of point scatterers placed at the blobs, so map values at the
blobs are known: positive at blob 1, negative at blob 2 (a "left" density),
and blob 2 is stored at a symmetry copy of the place next to the model.
"""

import json
import math
import os
import struct
import sys
import types
import zlib

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from valdo.easy import maps, report

CELL = gemmi.UnitCell(30, 35, 40, 90, 90, 90)
SG = gemmi.SpaceGroup("P 21 21 21")
DMIN = 2.0
LIG_POS = (14.0, 12.5, 13.0)      # blob 1: on the ligand of the bound model; nearest apo atom A/ALA3 O (3.80 A)
P2 = (7.8, 8.0, 8.5)              # blob 2 (negative density) next to A/ALA2 CA (1.50 A) ...
Q3 = (20.0, 12.0, 14.0)           # blob 3 (dataset 0002_1): nearest A/ALA5 O (4.16 A)
WATER = (14.0, 15.5, 13.0)        # 3.0 A from blob 1


def _residue_atoms(i):
    base = np.array([4.0 + 3.8 * (i - 1), 8.0, 10.0])
    offsets = {"N": (-1.2, 0.5, 0.0), "CA": (0, 0, 0), "C": (1.2, 0.6, 0.3), "O": (1.5, 1.7, 0.6),
               "CB": (0.0, -1.0, 1.2)}
    return [(name, name[0], tuple(base + np.array(off))) for name, off in offsets.items()]


def _sym_image(pos, op_index=1):
    """Cartesian position of a symmetry copy of `pos`, wrapped into the unit cell."""
    op = list(SG.operations())[op_index]
    f = CELL.fractionalize(gemmi.Position(*pos))
    g = np.array(op.apply_to_xyz([f.x, f.y, f.z])) % 1.0
    p = CELL.orthogonalize(gemmi.Fractional(*g))
    return (p.x, p.y, p.z)


BLOB2_CENTROID = _sym_image(P2)   # ... but stored at a symmetry copy, as gemmi's flood fill would


def write_structure(path, ligand=None, extra_het=()):
    st = gemmi.Structure()
    st.cell = CELL
    st.spacegroup_hm = SG.hm
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")

    def add(name, seq, atoms, het):
        res = gemmi.Residue()
        res.name = name
        res.seqid = gemmi.SeqId(seq, " ")
        res.het_flag = "H" if het else "A"
        for aname, el, pos in atoms:
            a = gemmi.Atom()
            a.name = aname
            a.element = gemmi.Element(el)
            a.pos = gemmi.Position(*pos)
            a.occ = 1.0
            a.b_iso = 20.0
            res.add_atom(a)
        chain.add_residue(res)

    for i in range(1, 7):
        add("ALA", i, _residue_atoms(i), False)
    add("HOH", 101, [("O", "O", WATER)], True)
    if ligand is not None:
        c = np.array(ligand)
        add("LIG", 201, [(f"C{k + 1}", "C", tuple(c + d)) for k, d in
                         enumerate([(0.8, 0, 0), (-0.8, 0, 0), (0, 0.8, 0.3), (0, -0.8, -0.3)])], True)
    for name, pos in extra_het:
        add(name, 301, [("C1", "C", pos)], True)
    model.add_chain(chain)
    st.add_model(model)
    st.write_pdb(str(path))
    return str(path)


def point_coefficients(H, points, B=15.0):
    s2 = 1.0 / rs.utils.compute_dHKL(H, CELL) ** 2
    F = np.zeros(len(H), dtype=complex)
    for pos, w in points:
        f = CELL.fractionalize(gemmi.Position(*pos))
        for op in SG.operations():
            x = np.array(op.apply_to_xyz([f.x, f.y, f.z]))
            F += w * np.exp(2j * np.pi * (H @ x))
    F *= np.exp(-B * s2 / 4.0)
    return np.abs(F), np.rad2deg(np.angle(F))


def write_coefficients(path, columns):
    """columns: {(F label, PHI label): [(position, weight), ...]}"""
    H = rs.utils.generate_reciprocal_asu(CELL, SG, DMIN, anomalous=False)
    H = H[np.any(H != 0, axis=1)]
    data = {"H": H[:, 0], "K": H[:, 1], "L": H[:, 2]}
    types_ = {"H": "HKL", "K": "HKL", "L": "HKL"}
    for (fl, pl), points in columns.items():
        amp, phi = point_coefficients(H, points)
        data[fl], data[pl] = amp, phi
        types_[fl], types_[pl] = "SFAmplitude", "Phase"
    ds = rs.DataSet(data, cell=CELL, spacegroup=SG, merged=True).astype(types_).set_index(["H", "K", "L"])
    ds.write_mtz(str(path))
    return str(path)


def tiny_png(path):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00\x2a\x78\xd6")) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def _blob(sample, score, peakz, cen, volume, focal=0, duplicate=0, ligand=0):
    f = CELL.fractionalize(gemmi.Position(*cen))
    frac = np.array([f.x, f.y, f.z])
    return {"sample": sample, "peakz": peakz, "peak": peakz * 0.9, "score": score, "cenx": cen[0],
            "ceny": cen[1], "cenz": cen[2], "volume": volume, "radius": (volume / (4 / 3 * np.pi)) ** (1 / 3),
            "fractional": frac, "all_possible_frac": [frac], "all_possible_cart": [np.array(cen)],
            "focal": focal, "duplicate": duplicate, "ligand": ligand}


def make_workdir(root, tables=("tagged", "filtered"), bound=True):
    wd = root / "work"
    for d in ("configs", "10_phases_blobs/mtz", "10_phases_blobs/blobs", "11_tag", "inputs/models",
              "inputs/bound_models", "04_refine/phases", "02_reindex"):
        (wd / d).mkdir(parents=True, exist_ok=True)
    apo = write_structure(wd / "apo.pdb")
    for stem in ("0001_0", "0002_1"):
        os.symlink(apo, wd / "inputs" / "models" / f"{stem}.pdb")
    bound_path = write_structure(wd / "inputs" / "bound_models" / "0001.pdb", ligand=LIG_POS,
                                 extra_het=[("TRS", (25.0, 30.0, 5.0))])
    write_coefficients(wd / "10_phases_blobs" / "mtz" / "0001_0.mtz",
                       {("WDF", "PH2FOFCWT"): [(LIG_POS, 1.0), (P2, -1.0)]})
    write_coefficients(wd / "10_phases_blobs" / "mtz" / "0002_1.mtz", {("WDF", "PH2FOFCWT"): [(Q3, 1.0)]})
    protein = [(pos, 1.0) for i in range(1, 7) for _, _, pos in _residue_atoms(i)]
    write_coefficients(wd / "04_refine" / "phases" / "0001_0.mtz", {("2FOFCWT", "PH2FOFCWT"): protein})
    tiny_png(wd / "02_reindex" / "reindex_cc_diff_histogram.png")

    with open(wd / "configs" / "add_phases_and_blobs.yaml", "w") as fh:
        fh.write(f"output_folder: {wd / '10_phases_blobs' / 'mtz'}\n"
                 f"blob_output_folder: {wd / '10_phases_blobs' / 'blobs'}\n"
                 "blob_diff_col: WDF\nphase_col: PH2FOFCWT\nphase_2FOFC_col_in: PH2FOFCWT\n"
                 f"phasing_path: {wd / '04_refine' / 'phases'}\nmodel_folder: {wd / 'inputs' / 'models'}\n"
                 "cutoff: 4\nradius_in_A: 5\n")
    with open(wd / "configs" / "tag_blobs.yaml", "w") as fh:
        fh.write(f"focal_tag_name: focal\nfocal_seqid: 3\noutput_folder: {wd / '11_tag'}\n")
    with open(wd / "configs" / "filter.yaml", "w") as fh:
        fh.write("max_rfree: 0.35\nmin_cc: 0.9\n")
    state = {"stages": {
        "standardize": {"status": "done", "seconds": 12.0, "counts": {"in": 3, "out": 3}},
        "refine": {"status": "done", "started": "2026-10-24T00:00:00", "finished": "2026-10-24T00:10:00"},
        "filter": {"status": "done", "seconds": 2.0, "counts": {"in": 3, "out": 2}},
        "add_phases_and_blobs": {"status": "done", "seconds": 30.0},
        "tag_blobs": {"status": "done", "seconds": 5.0},
        "train": {"status": "pending"},
        "report": {"status": "running"}}}
    with open(wd / "valdo_state.json", "w") as fh:
        json.dump(state, fh)
    ds = pd.DataFrame({
        "id": ["0001", "0002", "0003"],
        "original_name": ["PTP1B-y0001", "PTP1B-y0002", "PTP1B-y0003"],
        "status": ["ok", "ok", "dropped"],
        "dropped_stage": ["", "", "filter"],
        "dropped_reason": ["", "", "R-free 0.41 > 0.35"],
        "flags": ["", "low_resolution;ambiguous_indexing", ""],
        "variant": ["0", "1", "0"],
        "phase_file": ["", "", ""],
        "bound_model": [bound_path if bound else "", "", ""],
    })
    ds.to_csv(wd / "datasets.csv", index=False)

    blobs = [
        _blob("0001_0", 120.0, 9.0, LIG_POS, 60.0, ligand=1),
        _blob("0001_0", 80.0, 9.5, BLOB2_CENTROID, 40.0),
        _blob("0002_1", 100.0, 8.0, Q3, 50.0),
        _blob("0001_0", 150.0, 12.0, _residue_atoms(3)[1][2], 70.0, focal=1),   # near the focal residue
        _blob("0002_1", 99.0, 7.9, Q3, 50.0, duplicate=1),                       # symmetry duplicate
    ]
    allb = pd.DataFrame(blobs)
    kept = allb[(allb["focal"] == 0) & (allb["duplicate"] == 0)]
    if "tagged" in tables:
        allb.to_pickle(wd / "11_tag" / "blob_stats_tagged.pkl")
    if "filtered" in tables:
        kept.to_pickle(wd / "11_tag" / "filtered_blob_stats_tagged.pkl")
    if "raw" in tables:
        allb.drop(columns=["focal", "duplicate", "ligand"]).to_pickle(wd / "10_phases_blobs" / "blobs" /
                                                                     "blob_stats.pkl")
    return wd


@pytest.fixture
def workdir(tmp_path):
    return make_workdir(tmp_path)


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------

def test_report_tables_and_ranking(workdir):
    res = report.build_report(str(workdir), top_n=20)
    assert res["schema"] == report.REPORT_SCHEMA and res["ok"] and res["rank_by"] == "score"
    assert res["counts"] == {"datasets": 3, "datasets_ok": 2, "datasets_dropped": 1, "blobs_found": 5,
                             "blobs_kept": 3, "datasets_with_blobs": 2,
                             "blob_table": str(workdir / "11_tag" / "filtered_blob_stats_tagged.pkl")}
    blobs = pd.read_csv(workdir / "report" / "blobs.csv", dtype={"id": str, "symop": str})
    assert blobs.columns[:6].tolist() == ["rank", "id", "original_name", "sample", "symop", "rank_in_dataset"]
    assert blobs["sample"].tolist() == ["0001_0", "0002_1", "0001_0"]
    assert blobs["rank"].tolist() == [1, 2, 3] and blobs["rank_in_dataset"].tolist() == [1, 1, 2]
    assert blobs["id"].tolist() == ["0001", "0002", "0001"] and blobs["symop"].tolist() == ["0", "1", "0"]
    assert blobs["original_name"].tolist() == ["PTP1B-y0001", "PTP1B-y0002", "PTP1B-y0001"]
    assert not {"fractional", "all_possible_frac", "all_possible_cart"} & set(blobs.columns)
    assert np.allclose(blobs.loc[0, ["frac_x", "frac_y", "frac_z"]].astype(float),
                       [LIG_POS[0] / 30, LIG_POS[1] / 35, LIG_POS[2] / 40])
    assert blobs.loc[1, "flags"] == "low_resolution;ambiguous_indexing"
    per_ds = pd.read_csv(workdir / "report" / "datasets.csv", dtype=str, keep_default_na=False)
    assert per_ds["n_blobs"].tolist() == ["2", "1", "0"]
    assert per_ds["top_blob_rank"].tolist() == ["1", "2", ""]  # integers, empty without a blob
    assert per_ds["top_blob_sample"].tolist() == ["0001_0", "0002_1", ""]
    assert per_ds.loc[2, "status"] == "dropped" and per_ds.loc[2, "dropped_stage"] == "filter"
    funnel = {f["stage"]: (f["count"], f["dropped"]) for f in res["funnel"]}
    assert funnel["input"] == (3, 0) and funnel["filter"] == (2, 1) and funnel["hits"] == (2, None)
    assert "train" not in funnel  # pending stages are not in the funnel
    assert res["dropped"] == [{"stage": "filter", "reason": "R-free 0.41 > 0.35", "count": 1,
                               "examples": ["0003 (PTP1B-y0003)"]}]
    stages = {s["stage"]: s for s in res["stages"]}
    assert stages["refine"]["seconds"] == 600.0 and stages["standardize"]["n_out"] == 3
    assert stages["report"]["status"] == "this report"   # not 'running' (seen on the PTP1B run)
    assert {"stage": "filter", "key": "max_rfree", "value": 0.35} in res["settings"]
    # ranking by peak z instead
    res = report.build_report(str(workdir), top_n=2, rank_by="peakz")
    blobs = pd.read_csv(workdir / "report" / "blobs.csv", dtype={"id": str})
    assert blobs["peakz"].tolist() == [9.5, 9.0, 8.0] and len(res["top_hits"]) == 2
    with pytest.raises(ValueError):
        report.build_report(str(workdir), rank_by="size")


def test_top_hits_residues_maps_and_signs(workdir):
    res = report.build_report(str(workdir), top_n=3)
    hits = {h["rank"]: h for h in res["top_hits"]}
    h1, h2, h3 = hits[1], hits[2], hits[3]
    assert (h1["sample"], h2["sample"], h3["sample"]) == ("0001_0", "0002_1", "0001_0")
    assert h1["nearest_residue"] == "A/ALA3" and h1["nearest_atom"] == "O"
    assert abs(h1["nearest_distance_A"] - 3.80) < 0.01 and abs(h1["nearest_water_A"] - 3.0) < 0.01
    assert h2["nearest_residue"] == "A/ALA5" and abs(h2["nearest_distance_A"] - 4.16) < 0.01
    # blob 3 sits at a symmetry copy: found next to A/ALA2 CA, and the viewer centre is moved next to it
    assert h3["nearest_residue"] == "A/ALA2" and h3["nearest_atom"] == "CA"
    assert abs(h3["nearest_distance_A"] - 1.5) < 0.01
    assert np.allclose([h3["display_x"], h3["display_y"], h3["display_z"]], P2, atol=0.01)
    assert np.allclose([h3["cenx"], h3["ceny"], h3["cenz"]], BLOB2_CENTROID, atol=0.01)
    assert np.allclose([h1["display_x"], h1["display_y"], h1["display_z"]], LIG_POS, atol=0.01)
    # signed difference density: positive where density appeared, negative where it left
    assert h1["diff_peak_sigma"] > 5 and h3["diff_peak_sigma"] < -5 and h2["diff_peak_sigma"] > 5
    # both extremes and a sign (R1-01: the largest-magnitude value alone made ligands look negative)
    assert h1["diff_max_sigma"] > 5 and h1["diff_min_sigma"] > -3 and h1["diff_sign"] == "positive"
    assert h3["diff_min_sigma"] < -5 and h3["diff_max_sigma"] < 3 and h3["diff_sign"] == "negative"
    assert h1["diff_max_sigma"] >= h1["diff_at_centroid_sigma"] >= h1["diff_min_sigma"]
    assert (h1["blob"], h2["blob"], h3["blob"]) == ("0001_0 #1", "0002_1 #1", "0001_0 #2")
    # the overlap with the known bound model's ligand (only dataset 0001 has one)
    assert (h1["on_known_ligand"], h2["on_known_ligand"], h3["on_known_ligand"]) == (True, None, False)
    rep = workdir / "report"
    for h in (h1, h3):
        assert (rep / h["diff_map"]).is_file() and (rep / h["model"]).is_file()
    assert (rep / h1["context_map"]).is_file() and "PH2FOFCWT" in h1["context_source"]
    assert h2["context_map"] == "" and "context map not written" in h2["notes"]
    assert (rep / "hits" / report.HITS_MARKER).is_file()
    # the boxed map keeps whole-cell statistics and the same values as the full map
    m = gemmi.read_ccp4_map(str(rep / h3["diff_map"]))
    assert abs(m.header_float(22)) < 1e-3 and abs(m.header_float(55) - 1.0) < 1e-3
    assert max(m.grid.shape) < max(maps.map_from_mtz(
        str(workdir / "10_phases_blobs" / "mtz" / "0001_0.mtz"), "WDF", "PH2FOFCWT").shape)
    m.setup(float("nan"))
    full = maps.map_from_mtz(str(workdir / "10_phases_blobs" / "mtz" / "0001_0.mtz"), "WDF", "PH2FOFCWT")
    centre = gemmi.Position(*P2)
    assert abs(m.grid.interpolate_value(centre) - full.interpolate_value(centre)) < 1e-4
    assert full.interpolate_value(centre) < -5
    table = pd.read_csv(rep / "top_hits.csv", dtype={"id": str})
    assert table["rank"].tolist() == [1, 2, 3] and table.loc[2, "nearest_residue"] == "A/ALA2"
    # (blob 2, a lone synthetic peak of ~65 sigma, has a -3.9 sigma Fourier ripple at its edge: the
    # positive side dominates)
    assert table["diff_sign"].tolist() == ["positive", "positive", "negative"]
    assert table["diff_sign"].tolist() == [report.diff_sign(h["diff_max_sigma"], h["diff_min_sigma"])
                                           for h in (h1, h2, h3)]
    assert {"diff_max_sigma", "diff_min_sigma", "on_known_ligand", "blob"} <= set(table.columns)
    md = (rep / "report.md").read_text()
    assert "Diff. max / min (sigma) | Sign | Known ligand |" in md
    row1 = next(ln for ln in md.splitlines() if ln.startswith("| 1 | 0001 |"))
    assert "| positive | yes |" in row1 and f"+{h1['diff_max_sigma']:.1f} / " in row1


def test_diff_sign_levels():
    assert report.diff_sign(6.0, -1.0) == "positive" and report.diff_sign(2.0, -7.0) == "negative"
    assert report.diff_sign(4.0, -3.5) == "mixed" and report.diff_sign(2.9, -2.9) == "weak"
    assert report.diff_sign(3.0, -3.0) == "mixed"   # the level counts
    assert report.diff_sign(3.2, -2.5) == "positive"   # only one side reaches 3 sigma
    # both sides reach 3 sigma: the side at least 1.5 times stronger dominates (PTP1B top hits:
    # +15.5/-6.5 and +10.9/-6.7 are ligands next to displaced water; +14.1/-10.3 is mixed)
    assert report.diff_sign(15.5, -6.5) == "positive" and report.diff_sign(10.9, -6.65) == "positive"
    assert report.diff_sign(14.1, -10.3) == "mixed" and report.diff_sign(5.5, -7.0) == "mixed"
    assert report.diff_sign(4.0, -9.0) == "negative" and report.diff_sign(6.0, -4.0) == "positive"


def test_symmetry_copies_left_out_of_top_hits(tmp_path):
    """R1-03: upstream's duplicate tag only compares neighbours in peak order with a 1 A cut-off, so a
    symmetry copy of a strong blob can stay in the kept list; the top hits leave such copies out."""
    wd = make_workdir(tmp_path)
    kept = pd.read_pickle(wd / "11_tag" / "filtered_blob_stats_tagged.pkl")
    copy = _blob("0001_0", 110.0, 8.9, _sym_image(LIG_POS, 2), 60.0)   # rank 2: a copy of the rank-1 blob
    kept = pd.concat([kept, pd.DataFrame([copy])], ignore_index=True)
    kept.to_pickle(wd / "11_tag" / "filtered_blob_stats_tagged.pkl")
    res = report.build_report(str(wd), top_n=3)
    blobs = pd.read_csv(wd / "report" / "blobs.csv", dtype={"id": str})
    assert blobs["score"].tolist() == [120.0, 110.0, 100.0, 80.0]
    assert blobs["symmetry_copy_of"].fillna(0).astype(int).tolist() == [0, 1, 0, 0]
    assert [h["rank"] for h in res["top_hits"]] == [1, 3, 4]       # rank 2 left out, ranks unchanged
    assert res["top_hits_symmetry_copies_left_out"] == 1
    assert res["counts"]["blobs_kept"] == 4                           # nothing removed from the list
    assert "symmetry copies of blobs listed here are left out" in (wd / "report" / "report.md").read_text()
    # copies ranked below the last listed hit are not counted
    res2 = report.build_report(str(wd), top_n=1)
    assert [h["rank"] for h in res2["top_hits"]] == [1] and res2["top_hits_symmetry_copies_left_out"] == 0


def test_validation_against_known_bound_model(workdir):
    res = report.build_report(str(workdir), top_n=3)
    v = res["validation"]
    assert v["n_bound_models"] == 1 and v["n_analysed"] == 1 and v["n_with_ligand_blob"] == 1
    assert v["n_top1"] == 1 and v["median_rank_in_dataset"] == 1.0
    assert v["ligand_resnames"] == "auto: LIG"   # TRS and HOH are not ligands
    row = v["per_dataset"][0]
    assert row["id"] == "0001" and row["ligand_residues"] == "LIG"
    assert row["first_ligand_blob_rank_in_dataset"] == 1 and row["first_ligand_blob_global_rank"] == 1
    assert abs(row["closest_blob_distance_A"] - 0.8) < 0.05 and row["n_blobs"] == 2
    assert v["n_positive_blobs"] == 1 and v["auc_score"] == 1.0
    assert v["agreement_with_upstream_ligand_tags"] == "3/3 blobs"
    res = report.build_report(str(workdir), top_n=0, ligand_resnames=["TRS"])
    v = res["validation"]
    assert v["ligand_resnames"] == ["TRS"] and v["n_with_ligand_blob"] == 0 and res["top_hits"] == []


def test_known_bound_dir_matching(tmp_path):
    wd = make_workdir(tmp_path, bound=False)
    kb = tmp_path / "known"
    kb.mkdir()
    write_structure(kb / "PTP1B-y0001_bound.pdb", ligand=LIG_POS)
    write_structure(kb / "y0002-model.pdb", ligand=(2.0, 30.0, 35.0))
    write_structure(kb / "unrelated.pdb", ligand=LIG_POS)
    res = report.build_report(str(wd), top_n=0, known_bound_dir=str(kb), write_maps=False)
    v = res["validation"]
    per = {r["id"]: r for r in v["per_dataset"]}
    assert set(per) == {"0001", "0002"}
    assert per["0001"]["bound_model"].endswith("PTP1B-y0001_bound.pdb")
    assert per["0001"]["first_ligand_blob_rank_in_dataset"] == 1
    assert per["0002"]["first_ligand_blob_rank_in_dataset"] is None and per["0002"]["n_blobs"] == 1
    assert not (wd / "report" / "hits").exists() and "coot_script" not in res["files"]
    res = report.build_report(str(wd), top_n=0, known_bound_dir=str(tmp_path / "nope"), write_maps=False)
    assert res["validation"] is None and any("not found" in w for w in res["warnings"])


def test_outputs_html_markdown_json(workdir):
    res = report.build_report(str(workdir), top_n=3)
    rep = workdir / "report"
    html = (rep / "report.html").read_text()
    assert html.startswith("<!doctype html>") and "</html>" in html
    assert "<script" not in html.lower() and "http://" not in html and "https://" not in html
    assert "data:image/png;base64," in html and "Reindexing: CC difference" in html
    assert "PTP1B-y0001" in html and "A/ALA2" in html and "Known bound models" in html
    assert "low_resolution" in html and "R-free 0.41 &gt; 0.35" in html
    assert "prefers-color-scheme:dark" in html
    md = (rep / "report.md").read_text()
    assert "| 1 | 0001 | PTP1B-y0001 | 0001_0 #1 |" in md and "## Known bound models" in md
    data = json.loads((rep / "report.json").read_text())
    assert data["schema"] == report.REPORT_SCHEMA and data["counts"]["blobs_kept"] == 3
    assert data["qc_plots"][0]["title"] == "Reindexing: CC difference" and "data" not in data["qc_plots"][0]
    json.dumps(res)  # the return value is JSON-serialisable
    for key, path in res["files"].items():
        assert os.path.exists(path), key


def test_viewer_scripts(workdir, monkeypatch):
    report.build_report(str(workdir), top_n=3)
    rep = workdir / "report"
    # Coot: run the script against a stand-in `coot` module and check what it asks Coot to do
    calls = []
    fake = types.ModuleType("coot")
    counter = iter(range(100))
    fake.read_pdb = lambda p: calls.append(("pdb", p)) or next(counter)
    fake.handle_read_ccp4_map = lambda p, is_diff: calls.append(("map", p, is_diff)) or next(counter)
    fake.set_contour_level_absolute = lambda imol, level: calls.append(("level", imol, level))
    fake.set_rotation_centre = lambda x, y, z: calls.append(("centre", x, y, z))
    fake.close_molecule = lambda imol: calls.append(("close", imol))
    monkeypatch.setitem(sys.modules, "coot", fake)
    text = (rep / "view_top_hits.py").read_text()
    ns = {"__file__": str(rep / "view_top_hits.py"), "__name__": "valdo_coot_test"}
    exec(compile(text, "view_top_hits.py", "exec"), ns)
    assert [h["rank"] for h in ns["HITS"]] == [1, 2, 3]
    maps_opened = [c for c in calls if c[0] == "map"]
    assert maps_opened[0][1].endswith("hit01_0001_0_diff.ccp4") and maps_opened[0][2] == 1
    assert maps_opened[1][1].endswith("hit01_0001_0_2fofc.ccp4") and maps_opened[1][2] == 0
    assert all(os.path.isfile(c[1]) for c in calls if c[0] in ("map", "pdb"))
    assert ("level", 1, 3.0) in calls and ("level", 2, 1.0) in calls
    assert calls[-1][0] == "centre" and np.allclose(calls[-1][1:], LIG_POS, atol=0.01)
    calls.clear()
    ns["valdo_go"](3)
    assert ("close", 0) in calls and np.allclose(calls[-1][1:], P2, atol=0.01)
    calls.clear()
    ns["valdo_next"]()  # already at the last hit: stays there
    assert np.allclose(calls[-1][1:], P2, atol=0.01)
    # PyMOL: every file it loads exists, maps are contoured at absolute levels, one scene per hit
    pml = (rep / "view_top_hits.pml").read_text().splitlines()
    assert "set normalize_ccp4_maps, 0" in pml
    loads = [ln.split()[1].rstrip(",") for ln in pml if ln.startswith("load ")]
    assert loads and all((rep / p).is_file() for p in loads)
    assert "isomesh hit01_pos, hit01_diff, 3" in pml and "isomesh hit01_neg, hit01_diff, -3" in pml
    assert "isomesh hit01_2fofc, hit01_ctx, 1" in pml
    assert [ln.split(",")[0] for ln in pml if ln.startswith("scene ") and "store" in ln] == \
        ["scene hit01", "scene hit02", "scene hit03"]
    assert any(ln.startswith("pseudoatom hit03_centre, pos=[7.8, 8.0, 8.5]") for ln in pml)
    # R1-13: started from the work folder, both scripts still find the files under report/
    start = pml.index("python")
    block = "\n".join(pml[start + 1:pml.index("python end")])
    monkeypatch.chdir(workdir)
    exec(compile(block, "pml-python", "exec"), {})
    assert os.path.realpath(os.getcwd()) == os.path.realpath(rep)
    monkeypatch.chdir(workdir)
    calls.clear()
    ns = {"__name__": "valdo_coot_test"}   # Coot may run the script without __file__
    exec(compile(text, "view_top_hits.py", "exec"), ns)
    assert all(os.path.isfile(c[1]) for c in calls if c[0] in ("map", "pdb")) and calls


def test_rerun_replaces_hits_and_protects_foreign_folders(workdir, tmp_path):
    report.build_report(str(workdir), top_n=3)
    stale = workdir / "report" / "hits" / "hit99_old_diff.ccp4"
    stale.write_text("old")
    report.build_report(str(workdir), top_n=1)
    names = sorted(os.listdir(workdir / "report" / "hits"))
    assert "hit99_old_diff.ccp4" not in names
    assert [n for n in names if n.endswith(".ccp4")] == ["hit01_0001_0_2fofc.ccp4", "hit01_0001_0_diff.ccp4"]
    os.remove(workdir / "report" / "hits" / report.HITS_MARKER)
    with pytest.raises(RuntimeError, match="not written by valdo-easy"):
        report.build_report(str(workdir), top_n=1)
    assert (workdir / "report" / "hits" / "hit01_0001_0_diff.ccp4").is_file()


@pytest.mark.parametrize("tables, kept, found, warning", [
    (("tagged",), 3, 5, None),
    (("raw",), 5, 5, "tag_blobs has not run"),
    ((), 0, None, "no blob table found"),
])
def test_partial_work_folders(tmp_path, tables, kept, found, warning):
    wd = make_workdir(tmp_path, tables=tables)
    res = report.build_report(str(wd), top_n=2)
    assert res["counts"]["blobs_kept"] == kept and res["counts"]["blobs_found"] == found
    if warning:
        assert any(warning in w for w in res["warnings"])
    else:
        assert res["warnings"] == []
    html = (wd / "report" / "report.html").read_text()
    if kept == 0:
        assert "No candidate blobs" in html and res["top_hits"] == []
        assert pd.read_csv(wd / "report" / "blobs.csv").empty


def test_minimal_work_folder(tmp_path):
    wd = tmp_path / "bare"
    wd.mkdir()
    res = report.build_report(str(wd))
    assert res["ok"] and res["counts"]["datasets"] is None and res["validation"] is None
    assert any("datasets.csv not found" in w for w in res["warnings"])
    assert (wd / "report" / "report.html").is_file()
    with pytest.raises(FileNotFoundError):
        report.build_report(str(tmp_path / "missing"))


# ---------------------------------------------------------------------------
# valdo.easy.maps
# ---------------------------------------------------------------------------

def test_coefficient_pairs():
    cols = {"FWT": "F", "PHWT": "P", "2FOFCWT": "F", "PH2FOFCWT": "P", "DELFWT": "F", "PHDELWT": "P",
            "FOFCWT": "F", "PHFOFCWT": "Q"}
    assert maps.find_coefficient_pairs(cols, "2fofc") == [("2FOFCWT", "PH2FOFCWT"), ("FWT", "PHWT")]
    assert maps.find_coefficient_pairs(cols, "2fofc", preferred_phase="PHWT")[0] == ("FWT", "PHWT")
    assert maps.find_coefficient_pairs(cols, "fofc") == [("DELFWT", "PHDELWT")]  # PHFOFCWT has the wrong type
    assert maps.find_coefficient_pairs(["FOFCWT", "PHIFOFCWT"], "fofc") == [("FOFCWT", "PHIFOFCWT")]
    assert maps.find_coefficient_pairs([], "2fofc") == []


def test_locate_phase_file_mirrors_upstream(tmp_path):
    ph = tmp_path / "phases"
    ph.mkdir()
    for name in ("0001_0.mtz", "refine_0002_1_001.mtz", "refine_0002_1_002.mtz", "0003.mtz"):
        (ph / name).write_text("")
    assert maps.locate_phase_file("/x/0001_0.mtz", str(ph)) == (str(ph / "0001_0.mtz"), 2)
    assert maps.locate_phase_file("/x/0002_1.mtz", str(ph)) == (str(ph / "refine_0002_1_001.mtz"), 3)
    assert maps.locate_phase_file("/x/0003_1.mtz", str(ph)) == (str(ph / "0003.mtz"), 4)
    assert maps.locate_phase_file("/x/0004_0.mtz", str(ph)) == (None, -1)


def test_locate_model(tmp_path):
    (tmp_path / "0001_0.pdb").write_text("")
    (tmp_path / "model_0002_1_final.pdb").write_text("")
    assert maps.locate_model("0001_0", str(tmp_path)) == str(tmp_path / "0001_0.pdb")
    assert maps.locate_model("0002_1", str(tmp_path)) == str(tmp_path / "model_0002_1_final.pdb")
    assert maps.locate_model("0003_0", str(tmp_path)) is None and maps.locate_model("x", None) is None


def test_symmetry_helpers(tmp_path):
    st = gemmi.read_structure(write_structure(tmp_path / "m.pdb"))
    pos, d = maps.symmetry_image_near(CELL, SG, gemmi.Position(*BLOB2_CENTROID), gemmi.Position(*P2))
    assert d < 1e-6 and pos.dist(gemmi.Position(*P2)) < 1e-6
    near = maps.nearest_residue(st, gemmi.Position(*BLOB2_CENTROID))
    assert near["label"] == "A/ALA2" and near["atom"] == "CA" and abs(near["dist"] - 1.5) < 1e-3
    assert abs(near["display_pos"].dist(near["atom_pos"]) - near["dist"]) < 1e-3
    assert near["display_pos"].dist(gemmi.Position(*P2)) < 1e-3
    assert maps.nearest_residue(st, gemmi.Position(*WATER), max_dist=1.0) is None  # waters are skipped
    assert abs(maps.nearest_water_distance(st, gemmi.Position(*LIG_POS)) - 3.0) < 1e-3
    assert maps.residues_within(st, gemmi.Position(*BLOB2_CENTROID), 2.1) == ["A/ALA2"]


def test_map_values_and_ccp4_box(tmp_path):
    mtz = write_coefficients(tmp_path / "d.mtz", {("WDF", "PHWDF"): [(LIG_POS, 1.0), (P2, -1.0)]})
    grid = maps.map_from_mtz(mtz, "WDF", "PHWDF")
    mean, rms = maps.grid_rms(grid)
    assert abs(mean) < 1e-4 and abs(rms - 1.0) < 1e-3
    pos = maps.values_near(grid, gemmi.Position(*LIG_POS), 1.5)
    neg = maps.values_near(grid, gemmi.Position(*BLOB2_CENTROID), 1.5)  # symmetry copy of the negative peak
    assert pos["max_abs"] == pos["max"] > 5 and neg["max_abs"] == neg["min"] < -5
    assert abs(pos["at_center"] - grid.interpolate_value(gemmi.Position(*LIG_POS))) < 1e-6
    with pytest.raises(KeyError, match="no column"):
        maps.map_from_mtz(mtz, "FWT", "PHWT")
    out = tmp_path / "box.ccp4"
    info = maps.write_ccp4(grid, str(out), center=gemmi.Position(*P2), radius=5.0)
    assert info["boxed"] and info["shape"] != info["full_cell_shape"] and not os.path.exists(str(out) + ".tmp")
    m = gemmi.read_ccp4_map(str(out))
    assert abs(m.header_float(22)) < 1e-3 and abs(m.header_float(55) - 1.0) < 1e-3
    full = tmp_path / "full.ccp4"
    info = maps.write_ccp4(grid, str(full))
    assert not info["boxed"] and info["shape"] == info["full_cell_shape"]
    m = gemmi.read_ccp4_map(str(full))
    assert list(m.grid.shape) == list(grid.shape)
    assert math.isclose(m.grid.interpolate_value(gemmi.Position(*P2)), grid.interpolate_value(gemmi.Position(*P2)),
                        abs_tol=1e-5)


def test_dropped_reasons_that_differ_only_in_values_are_grouped():
    """On the full PTP1B screen every min_cc drop had its own CC value, which made one row per
    dataset in the report; they now form one row, with the threshold kept."""
    d = pd.DataFrame({
        "id": ["0001", "0002", "0003", "0004"], "original_name": ["a", "b", "c", "d"],
        "status": ["dropped", "dropped", "dropped", "ok"], "dropped_stage": ["filter", "filter", "input", ""],
        "dropped_reason": ["CC to the reference after scaling 0.349 is below min_cc 0.55",
                           "CC to the reference after scaling 0.254 is below min_cc 0.55",
                           "space group P 32 2 1 differs from the other datasets (P 31 2 1)", ""]})
    rows = report._dropped_summary(d)
    assert rows[0] == {"stage": "filter", "reason": "CC to the reference after scaling ... is below min_cc 0.55",
                       "count": 2, "examples": ["0001 (a): 0.349", "0002 (b): 0.254"]}
    assert rows[1]["reason"] == "space group P 32 2 1 differs from the other datasets (P 31 2 1)"
