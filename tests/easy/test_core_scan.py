"""Input scanning: MTZ header reader, file lists, dataset names, IDs, column/phase detection,
name matching, alternative indexing."""

import os

import pytest

from conftest import needs_science
from valdo.easy import ids, scan


# ---------------------------------------------------------------------------
# IDs and datasets.csv
# ---------------------------------------------------------------------------

def test_natural_sort_and_ids():
    names = ["x10", "x2", "x1", "X3", "x2b"]
    assert scan.sort_names(names) == ["x1", "x2", "x2b", "X3", "x10"]
    m = ids.assign_ids(names)
    assert m["x1"] == "0001" and m["x10"] == "0005"
    assert ids.id_width(9999) == 4 and ids.id_width(10000) == 5
    with pytest.raises(ValueError):
        ids.assign_ids(["a", "a"])


def test_parse_stem():
    assert ids.parse_stem("0042_1") == ("0042", "1")
    assert ids.parse_stem("0042") == ("0042", None)
    assert ids.parse_stem("PTP1B-y0042") == (None, None)


def test_datasets_csv_round_trip(tmp_path):
    rows = [{"id": "0001", "original_name": "a,b", "status": "ok", "flags": "reference;cell_outlier"},
            {"id": "0002", "original_name": "c", "status": "dropped", "dropped_stage": "input",
             "dropped_reason": "unreadable", "extra": "ignored"}]
    path = str(tmp_path / "datasets.csv")
    ids.write_datasets_csv(rows, path)
    back = ids.read_datasets_csv(path)
    assert [r["id"] for r in back] == ["0001", "0002"]          # zero padding kept
    assert back[0]["original_name"] == "a,b" and back[1]["dropped_stage"] == "input"
    assert list(back[0]) == ids.DATASET_COLUMNS
    assert ids.split_flags(back[0]["flags"]) == ["reference", "cell_outlier"]
    assert ids.join_flags(["a", "b", "a", ""]) == "a;b"


# ---------------------------------------------------------------------------
# Files and names
# ---------------------------------------------------------------------------

def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


def test_list_data_files(tmp_path):
    for n in ("a.mtz", "b.mtz", "c.txt"):
        touch(str(tmp_path / "d" / n))
    files, note = scan.list_data_files("d", str(tmp_path))
    assert [os.path.basename(f) for f in files] == ["a.mtz", "b.mtz"] and note == ""
    files, _ = scan.list_data_files(str(tmp_path / "d" / "a*.mtz"))
    assert len(files) == 1
    lst = tmp_path / "list.txt"
    lst.write_text("# comment\nd/b.mtz\n\n" + str(tmp_path / "d" / "a.mtz") + "\n")
    files, _ = scan.list_data_files(str(lst))
    assert sorted(os.path.basename(f) for f in files) == ["a.mtz", "b.mtz"]
    touch(str(tmp_path / "nested" / "x1" / "final.mtz"))
    files, note = scan.list_data_files(str(tmp_path / "nested"))
    assert files == [] and "subfolders" in note and "final.mtz" in note
    files, note = scan.list_data_files(str(tmp_path / "missing"))
    assert files == [] and "does not exist" in note


def test_dataset_names():
    assert scan.dataset_names(["/d/x1.mtz", "/d/x2.mtz"]) == ["x1", "x2"]
    # every file has the same name: the folder names identify the datasets
    assert scan.dataset_names(["/r/x1/dimple.mtz", "/r/x2/dimple.mtz"]) == ["x1", "x2"]
    # stems repeat in places: shortest unique path suffix
    names = scan.dataset_names(["/a/s1/f.mtz", "/a/s2/f.mtz", "/a/s2/g.mtz"])
    assert len(set(names)) == 3 and names[2] == "s2/g"
    # an ending every file shares is dropped (PTP1B), but only at a separator
    assert scan.dataset_names(["/d/PTP1B-y0001_mrflagsref_idxs.mtz", "/d/PTP1B-y0011_mrflagsref_idxs.mtz"]) == \
        ["PTP1B-y0001", "PTP1B-y0011"]
    assert scan.dataset_names(["/d/x10.mtz", "/d/x20.mtz"]) == ["x10", "x20"]
    assert scan.dataset_names(["/d/a_final.mtz"]) == ["a_final"]
    assert scan.strip_common_suffix(["a-1_x", "a-1_y"]) == ["a-1_x", "a-1_y"]


def test_match_files_to_names_boundary_rule(tmp_path):
    for n in ("PTP1B-y1-pandda-input.pdb", "PTP1B-y10-pandda-input.pdb", "unrelated.pdb", "PTP1B-y2.cif"):
        touch(str(tmp_path / "m" / n))
    names = ["PTP1B-y1", "PTP1B-y10", "PTP1B-y2", "PTP1B-y3"]
    found, problems = scan.match_files_to_names(str(tmp_path / "m"), names)
    assert os.path.basename(found["PTP1B-y1"]) == "PTP1B-y1-pandda-input.pdb"     # not y10
    assert os.path.basename(found["PTP1B-y10"]) == "PTP1B-y10-pandda-input.pdb"
    assert os.path.basename(found["PTP1B-y2"]) == "PTP1B-y2.cif"
    assert "PTP1B-y3" not in found and problems == []
    found, _ = scan.match_files_to_names(str(tmp_path / "m" / "{name}-pandda-input.pdb"), names)
    assert set(found) == {"PTP1B-y1", "PTP1B-y10"}


def test_match_files_by_folder_and_ambiguity(tmp_path):
    touch(str(tmp_path / "r" / "x1" / "final.pdb"))
    touch(str(tmp_path / "r" / "x2" / "final.pdb"))
    touch(str(tmp_path / "r" / "x2" / "x2_extra.pdb"))
    found, problems = scan.match_files_to_names(str(tmp_path / "r" / "*" / "*.pdb"), ["x1", "x2"])
    assert os.path.basename(os.path.dirname(found["x1"])) == "x1"
    assert "x2" not in found and problems and "{name}" in problems[0]


# ---------------------------------------------------------------------------
# MTZ headers
# ---------------------------------------------------------------------------

@needs_science
def test_header_reader_matches_gemmi(toy_screen):
    import gemmi
    folder, truth = toy_screen
    paths = sorted(os.path.join(folder, "data", f) for f in os.listdir(os.path.join(folder, "data")))
    headers = scan.scan_headers(paths, threads=4)
    good = [h for h in headers if h["ok"]]
    assert len(good) == len(truth["datasets"]) and len(headers) == len(good) + len(truth["broken"])
    for h in good[:5]:
        mtz = gemmi.read_mtz_file(h["path"])
        assert h["nrefl"] == mtz.nreflections
        assert h["cell"] == pytest.approx(tuple(mtz.cell.parameters), abs=1e-3)
        assert h["sg_name"] == mtz.spacegroup.hm
        assert [c["label"] for c in h["columns"]] == [c.label for c in mtz.columns]
        assert [c["type"] for c in h["columns"]] == [c.type for c in mtz.columns]
        assert h["d_min"] == pytest.approx(mtz.resolution_high(), rel=1e-3)
    bad = [h for h in headers if not h["ok"]]
    assert bad and bad[0]["error"]


def test_header_reader_rejects_non_mtz(tmp_path):
    p = tmp_path / "x.mtz"
    p.write_bytes(b"not an mtz at all, just text" * 3)
    h = scan.read_mtz_header(str(p))
    assert not h["ok"] and "not an MTZ" in h["error"]
    h = scan.read_mtz_header(str(tmp_path / "missing.mtz"))
    assert not h["ok"]


def header(columns, **kw):
    h = {"ok": True, "columns": [{"label": l, "type": t, "dataset": 1} for l, t in columns], "path": "x",
         "nbatch": 0, "sg_name": "P 1", "cell": (10, 10, 10, 90, 90, 90), "d_min": 2.0}
    h.update(kw)
    return h


def test_amplitude_detection():
    h1 = header([("FP", "F"), ("SIGFP", "Q"), ("F-obs", "F"), ("SIGF-obs", "Q"), ("FWT", "F"), ("PHWT", "P")])
    det = scan.detect_amplitude_columns([h1, h1])
    assert (det["amplitude"], det["sigma"]) == ("F-obs", "SIGF-obs") and det["alternatives"] == ["FP,SIGFP"]
    h2 = header([("I", "J"), ("SIGI", "Q")])
    det = scan.detect_amplitude_columns([h2])
    assert det["amplitude"] is None and det["intensity_only"] == 1
    # the pair present in most files wins over priority
    h3 = header([("FP", "F"), ("SIGFP", "Q")])
    assert scan.detect_amplitude_columns([h3, h3, h1])["amplitude"] == "FP"


def test_phase_and_rfree_detection():
    xchem = header([("2FOFCWT", "F"), ("PHI2FOFCWT", "P"), ("FOFCWT", "F"), ("PHIFOFCWT", "P"), ("R-free-flags", "I")])
    refmac = header([("FWT", "F"), ("PHWT", "P"), ("DELFWT", "F"), ("PHDELWT", "P"), ("FreeR_flag", "I")])
    det = scan.detect_phase_columns([xchem, xchem, refmac])
    assert det["convention"] == "xchem" and det["phwt"] == "PHI2FOFCWT" and det["n_files"] == 2
    assert scan.detect_phase_columns([header([("F", "F"), ("SIGF", "Q")])]) is None
    assert scan.detect_rfree_column([refmac]) == "FreeR_flag"
    assert scan.has_phase_columns(refmac, scan.PHASE_CONVENTIONS["refmac"])


def test_summary_flags_outliers_and_space_groups():
    hs = [header([], sg_name="P 3 2 1", cell=(90, 90, 106, 90, 90, 120), path=f"f{i}") for i in range(5)]
    hs.append(header([], sg_name="P 3 2 1", cell=(95, 95, 106, 90, 90, 120), path="big"))
    hs.append(header([], sg_name="C 1 2 1", cell=(90, 90, 106, 90, 100, 90), path="mono"))
    s = scan.summarize_headers(hs)
    assert s["space_group"] == "P 3 2 1" and s["sg_mismatch"] == ["mono"] and s["cell_outliers"] == ["big"]


@needs_science
def test_alternative_indexing_as_upstream():
    import gemmi
    ops, hkl = scan.alternative_indexing((89.675, 89.675, 106.375, 90, 90, 120), "P 31 2 1")
    expected = gemmi.find_twin_laws(gemmi.UnitCell(89.675, 89.675, 106.375, 90, 90, 120),
                                    gemmi.SpaceGroup("P 31 2 1"), 3, False)
    assert [o.triplet() for o in ops] == [o.triplet() for o in expected] and len(hkl) == len(ops) >= 1
    _, hkl = scan.alternative_indexing((50, 60, 70, 90, 90, 90), "P 21 21 21")
    assert hkl == []
    assert scan.hkl_triplet(gemmi.Op("-x,-y,z")) == "-h,-k,l"
    assert scan.hkl_triplet(gemmi.Op("y,x,-z")) == "k,h,-l"
