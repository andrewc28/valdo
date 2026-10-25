"""Refinement-program installations (programs.phenix / programs.ccp4, valdo/easy/programs.py) and
twin refinement (phases.twin_law, valdo/easy/twin.py).

PHENIX and CCP4 are never run here. Fake installation trees (setup script, bin folder, a stand-in
program) take their place; the stand-in refinement programs are the ones of test_refine.py, plus a
small script that adds what dimple 2.7 (``--refmac-key twin``) or phenix.refine
(``xray_data.twin_law=``) write about twinning. The REFMAC excerpts are verbatim from refmacat (CCP4
9.0.017, dimple 2.7) on S2 benchmark datasets (P 31 2 1); the PHENIX excerpts are verbatim from
phenix.refine 2.2.1-6174 and 1.21.2-5419 on S2 dataset 0053 (twin law -h,-k,l, twin fraction 0.26).
"""

import json
import math
import os
import stat
import sys
import textwrap

import pytest

from conftest import needs_science, needs_torch, run_valdo, valdo_env
from valdo.easy import programs, twin


@pytest.fixture(autouse=True)
def _fresh_cache():
    programs.clear_cache()      # resolve() caches per process: every test starts clean
    yield
    programs.clear_cache()


def quiet(*args, **kwargs):
    pass


def _write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def fake_phenix(base, version="2.2-6143", modern=True, program=None):
    """<base>/phenix-<version> with phenix_env.sh and phenix.refine (PHENIX 2.x: phenix_bin/,
    PHENIX 1.x: build/bin/ and no PHENIX_VERSION)."""
    root = base / f"phenix-{version}"
    bin_rel = "phenix_bin" if modern else "build/bin"
    lines = [f'export PHENIX="{root}"', f'export PATH="$PHENIX/{bin_rel}:$PATH"', 'export VALDO_TEST_MARKER=phenix']
    if modern:
        lines.insert(1, f"export PHENIX_VERSION={version}")
    _write(root / "phenix_env.sh", "\n".join(lines) + "\n")
    _write(root / bin_rel / "phenix.refine", program or "#!/bin/sh\necho 'fake phenix.refine'\n", executable=True)
    return root


def fake_ccp4(base, program=None, version="9.0.017", name="ccp4-9"):
    """<base>/<name> with bin/ccp4.setup-sh, bin/dimple, bin/refmac5 and lib/ccp4/MAJOR_MINOR."""
    root = base / name
    _write(root / "bin" / "ccp4.setup-sh", textwrap.dedent(f"""\
        # CCP4 setup for sh/bash (stand-in for the valdo-easy tests)
        export CCP4_MASTER="{base}"
        export CCP4="{root}"
        export PATH="$CCP4/bin:$PATH"
        export VALDO_TEST_MARKER=ccp4
        """))
    _write(root / "bin" / "dimple", program or "#!/bin/sh\necho 'dimple 2.7.0 (fake)'\n", executable=True)
    _write(root / "bin" / "refmac5", "#!/bin/sh\nexit 0\n", executable=True)
    _write(root / "lib" / "ccp4" / "MAJOR_MINOR", version + "\n")
    return root


# ---------------------------------------------------------------------------
# programs.phenix / programs.ccp4
# ---------------------------------------------------------------------------

def test_phenix_installation_from_folder_script_bin_or_program(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "valdo-python"))
    root = fake_phenix(tmp_path)
    before = dict(os.environ)
    for given in (root, root / "phenix_env.sh", root / "phenix_bin", root / "phenix_bin" / "phenix.refine"):
        programs.clear_cache()
        info = programs.resolve("phenix", str(given))
        assert info["error"] is None, (given, info["error"])
        assert info["root"] == str(root) and info["setup_script"] == str(root / "phenix_env.sh")
        assert info["exe"] == str(root / "phenix_bin" / "phenix.refine") and info["source"] == "valdo.yaml"
        assert info["version"] == "2.2-6143"
        env = info["env"]
        assert env["VALDO_TEST_MARKER"] == "phenix" and env["PATH"].split(os.pathsep)[0] == str(root / "phenix_bin")
        assert "PYTHONPATH" not in env           # valdo's Python settings are not passed on to PHENIX
    assert dict(os.environ) == before            # valdo's own environment is not changed
    assert programs.describe(info) == f"PHENIX 2.2-6143 (phenix.refine) at {root} (programs.phenix in valdo.yaml)"
    public = programs.public(info)
    assert "env" not in public and json.loads(json.dumps(public))["exe"] == info["exe"]


def test_phenix_1x_layout_and_folder_without_setup_script(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    root = fake_phenix(tmp_path, version="1.21.2-5419", modern=False)
    info = programs.resolve("phenix", str(root / "build" / "bin" / "phenix.refine"))
    assert info["error"] is None and info["root"] == str(root)
    assert info["exe"] == str(root / "build" / "bin" / "phenix.refine")
    assert info["version"] == "1.21.2-5419"              # no PHENIX_VERSION: from the folder name
    # no phenix_env.sh at all: the folder with phenix.refine is put first on PATH
    bare = tmp_path / "phenix-dev"
    _write(bare / "bin" / "phenix.refine", "#!/bin/sh\nexit 0\n", executable=True)
    info = programs.resolve("phenix", str(bare))
    assert info["error"] is None and info["setup_script"] is None
    assert info["exe"] == str(bare / "bin" / "phenix.refine") and info["version"] is None
    assert info["env"]["PATH"].split(os.pathsep)[0] == str(bare / "bin")
    assert any("no phenix_env.sh found" in n for n in info["notes"])


def test_ccp4_installation_from_folder_script_bin_or_program(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "valdo-home"))
    root = fake_ccp4(tmp_path)
    for given in (root, root / "bin", root / "bin" / "ccp4.setup-sh", root / "bin" / "dimple"):
        programs.clear_cache()
        info = programs.resolve("dimple", str(given))
        assert info["error"] is None, (given, info["error"])
        assert info["root"] == str(root) and info["setup_script"] == str(root / "bin" / "ccp4.setup-sh")
        assert info["exe"] == str(root / "bin" / "dimple") and info["notes"] == []
        assert info["version"] == "CCP4 9.0.017, dimple 2.7.0"
        assert info["env"]["CCP4"] == str(root) and "PYTHONHOME" not in info["env"]
    assert programs.describe(info) == (f"CCP4 9.0.017, dimple 2.7.0 (dimple) at {root} "
                                       "(programs.ccp4 in valdo.yaml)")


def test_installation_problems_are_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    def error(program, path):
        programs.clear_cache()
        return programs.resolve(program, None if path is None else str(path))["error"]

    e = error("dimple", tmp_path / "nowhere")
    assert e["code"] == "program_not_found" and "does not exist" in e["what"] and "programs.ccp4" in e["fix"]
    (tmp_path / "empty").mkdir()
    e = error("dimple", tmp_path / "empty")
    assert e["code"] == "program_not_found" and "no CCP4 setup script" in e["what"]
    e = error("phenix", tmp_path / "empty")
    assert e["code"] == "program_not_found" and "no phenix_env.sh and no phenix.refine" in e["what"]
    csh = _write(tmp_path / "ccp4-csh" / "bin" / "ccp4.setup-csh", "setenv CCP4 /x\n")
    e = error("dimple", csh)
    assert e["code"] == "program_not_found" and "csh script" in e["what"]
    stops = fake_ccp4(tmp_path, name="ccp4-exits")
    (stops / "bin" / "ccp4.setup-sh").write_text("echo 'unsupported shell' >&2\nexit 3\n")
    e = error("dimple", stops)
    assert e["code"] == "program_setup_failed" and "gave no environment back" in e["what"]
    assert "unsupported shell" in e["what"]
    no_dimple = fake_ccp4(tmp_path, name="ccp4-no-dimple")
    (no_dimple / "bin" / "dimple").unlink()
    e = error("dimple", no_dimple)
    assert e["code"] == "program_setup_failed" and "dimple is not on PATH after sourcing" in e["what"]
    e = error("dimple", None)                     # not configured and not on PATH
    assert e["code"] == "refine_program_missing" and "programs.ccp4" in e["fix"]
    e = error("phenix", None)
    assert e["code"] == "refine_program_missing" and "programs.phenix" in e["fix"]
    with pytest.raises(ValueError):
        programs.resolve("buster")


def test_the_setup_script_is_sourced_once_per_process(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    root = fake_ccp4(tmp_path)
    counter = tmp_path / "sourced.txt"
    with open(root / "bin" / "ccp4.setup-sh", "a") as handle:
        handle.write(f'echo x >> "{counter}"\n')
    first = programs.resolve("dimple", str(root))
    assert programs.resolve("dimple", str(root)) is first and counter.read_text().count("x") == 1
    programs.clear_cache()
    programs.resolve("dimple", str(root))
    assert counter.read_text().count("x") == 2


def test_programs_on_path_and_for_project(tmp_path, monkeypatch):
    root = fake_ccp4(tmp_path)
    monkeypatch.setenv("PATH", str(root / "bin") + os.pathsep + "/usr/bin:/bin")
    monkeypatch.delenv("CCP4", raising=False)
    info = programs.resolve("dimple")
    assert info["error"] is None and info["source"] == "PATH" and info["env"] is None
    assert info["exe"] == str(root / "bin" / "dimple") and info["version"] == "dimple 2.7.0"
    assert any("CCP4 variable is not set" in n for n in info["notes"])
    assert programs.describe(info) == f"dimple 2.7.0 (dimple) at {root / 'bin' / 'dimple'} (found on PATH)"
    # valdo init: the installation that the program on PATH belongs to
    found, note = programs.installation_on_path("dimple")
    assert found == str(root) and str(root / "bin" / "dimple") in note
    bare = tmp_path / "phenix-dev" / "bin"
    _write(bare / "phenix.refine", "#!/bin/sh\nexit 0\n", executable=True)
    monkeypatch.setenv("PATH", str(bare) + os.pathsep + "/usr/bin:/bin")
    found, note = programs.installation_on_path("phenix")
    assert found is None and "no phenix_env.sh was found" in note
    project = {"phases": {"source": "refine", "program": "dimple"}, "programs": {"phenix": None, "ccp4": str(root)}}
    assert programs.for_project(project)["root"] == str(root)
    project["phases"]["source"] = "input_mtz"
    assert programs.for_project(project) is None


def test_pasted_lines_other_files_parent_folders_and_links(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    lab = tmp_path / "lab"
    root = fake_phenix(lab, version="2.2.1-6174")
    script = str(root / "phenix_env.sh")
    # a pasted shell line is read as the path in it
    for pasted in (f"source {script}", f'source "{script}" && phenix.refine --version', f". {script};",
                   f"'{script}'", f"source {script} || exit 1"):
        text, note = programs.clean_setting(pasted)
        assert text == script and note.endswith(f"as the path {script}"), pasted
    for command in ("export PATH=/opt/phenix/bin:$PATH", "module load phenix", "ml ccp4", "setenv CCP4 /x"):
        with pytest.raises(ValueError, match="shell command"):
            programs.clean_setting(command)
    # another file of the installation is never sourced: its folder's setup script is used
    readme = _write(root / "README", f'echo sourced > "{tmp_path / "readme-ran"}"\n')
    info = programs.resolve("phenix", str(readme))
    assert info["error"] is None and info["setup_script"] == script and not (tmp_path / "readme-ran").exists()
    notes = _write(tmp_path / "notes" / "phenix-howto.txt", "source /opt/phenix/phenix_env.sh\n")
    programs.clear_cache()
    e = programs.resolve("phenix", str(notes))["error"]
    assert e["code"] == "program_not_found" and "is not PHENIX's setup script" in e["what"]
    # a folder that holds installations: say which
    programs.clear_cache()
    e = programs.resolve("phenix", str(lab))["error"]
    assert e["code"] == "program_not_found" and f"holds one: set programs.phenix to {root}" in e["what"]
    fake_phenix(lab, version="1.21.2-5419", modern=False)
    programs.clear_cache()
    e = programs.resolve("phenix", str(lab))["error"]
    assert "holds 2: phenix-1.21.2-5419, phenix-2.2.1-6174" in e["what"]
    # a link to the program (e.g. in /usr/local/bin): the installation it points into, with its setup script
    links = tmp_path / "usr-local-bin"
    links.mkdir()
    (links / "phenix.refine").symlink_to(root / "phenix_bin" / "phenix.refine")
    programs.clear_cache()
    info = programs.resolve("phenix", str(links / "phenix.refine"))
    assert info["error"] is None and info["root"] == str(root) and info["setup_script"] == script


def test_moved_installation_and_what_the_setup_script_sees(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    ran = tmp_path / "bash_env-ran"
    monkeypatch.setenv("BASH_ENV", str(_write(tmp_path / "bash_env.sh", f'echo x > "{ran}"\n')))
    root = fake_ccp4(tmp_path)
    with open(root / "bin" / "ccp4.setup-sh", "a") as handle:     # what CCP4's script looks at, and prints
        handle.write('export VALDO_TEST_NARGS=$#\nexport VALDO_TEST_ZERO="$0"\necho "Welcome to CCP4"\n')
    info = programs.resolve("dimple", str(root))
    assert info["error"] is None and not ran.exists()                 # BASH_ENV is not read
    assert info["env"]["VALDO_TEST_NARGS"] == "0"                      # the script gets no arguments
    assert info["env"]["VALDO_TEST_ZERO"] == str(root / "bin" / "ccp4.setup-sh")
    assert "BASH_ENV" not in info["env"]
    # the installation was moved after it was installed: its script still points to the old place
    moved = tmp_path / "moved" / "ccp4-9"
    moved.parent.mkdir()
    root.rename(moved)
    programs.clear_cache()
    e = programs.resolve("dimple", str(moved))["error"]
    assert e["code"] == "program_setup_failed" and f"CCP4={root}" in e["what"] and "was moved" in e["what"]
    assert "Install CCP4 again" in e["fix"]


def test_version_differs():
    recorded = {"program": "phenix", "version": "1.21.2-5419", "root": "/opt/phenix-1.21.2-5419"}
    assert programs.version_differs(recorded, {"program": "phenix", "version": "2.2.1-6174"})
    assert not programs.version_differs(recorded, {"program": "phenix", "version": "1.21.2-5419", "root": "/new"})
    assert not programs.version_differs(recorded, {"program": "dimple", "version": "CCP4 9.0.017"})
    assert not programs.version_differs(recorded, {"program": "phenix", "version": None})
    assert not programs.version_differs({"program": "dimple", "version": "CCP4 9.0.017"},
                                        {"program": "dimple", "version": "CCP4 9.0.017, dimple 2.7.0"})
    assert programs.short({"family": "phenix", "version": "2.2.1-6174"}) == "PHENIX 2.2.1-6174"


# ---------------------------------------------------------------------------
# Twin laws
# ---------------------------------------------------------------------------

def test_twin_law_text():
    assert twin.normalise_text("-K, -H, -L") == "-k,-h,-l"            # as REFMAC writes it
    assert twin.normalise_text(" H,  K,  L") == "h,k,l"
    assert twin.normalise_text("[-h,-k,l]") == "-h,-k,l"              # as pointless writes it
    assert twin.normalise_text("h+k, -k, -l") == "h+k,-k,-l"
    assert twin.normalise_text("-H-K, K, -L") == "-h-k,k,-l"
    assert twin.normalise_text("1/2*h+3/2*k,1/2*h-1/2*k,-l") == "1/2*h+3/2*k,1/2*h-1/2*k,-l"
    law = twin.parse_law("-h,-k,l")
    assert twin.det(law) == 1 and twin.matmul(law, law) == twin.IDENTITY
    assert twin.format_law(twin.inverse(twin.parse_law("k,-h-k,l"))) == "-h-k,h,l"
    for bad in ("", "x,y,z", "-y,-x,-z", "h,k", "h,k,q", "h,,l", "h k l"):
        with pytest.raises(twin.TwinLawError):
            twin.parse_law(bad)


def test_twin_in_extra_args():
    assert twin.twin_in_extra_args("phenix", ["twin_law=-h,-k,l"]) == "twin_law=-h,-k,l"
    assert twin.twin_in_extra_args("phenix", ["refinement.twinning.twin_law=-k,-h,-l"])
    assert twin.twin_in_extra_args("phenix", ["main.number_of_macro_cycles=3"]) is None
    assert twin.twin_in_extra_args("dimple", ["--refmac-key", "twin"]) == "--refmac-key twin"
    assert twin.twin_in_extra_args("dimple", ["--refmac-key=TWIN"]) == "--refmac-key=TWIN"
    assert twin.twin_in_extra_args("dimple", ["--refmac-key", "ncyc 10"]) is None


def _same(a, b, rots):
    return twin.same_law(twin.parse_law(a), twin.parse_law(b), rots)


@needs_science
@pytest.mark.parametrize("sg, cell, expected", [
    ("P 31 2 1", (89.5, 89.5, 106.4, 90, 90, 120), ["-h,-k,l"]),
    ("P 3 2 1", (36, 36, 44, 90, 90, 120), ["-h,-k,l"]),               # the toy screen
    ("P 31", (50, 50, 80, 90, 90, 120), ["k,h,-l", "-h,-k,l", "-k,-h,-l"]),
    ("P 41", (50, 50, 80, 90, 90, 90), ["-h,k,-l"]),
    ("P 41 21 2", (50, 50, 80, 90, 90, 90), []),
    ("P 21 21 21", (50, 50.2, 80, 90, 90, 90), ["-k,-h,-l"]),         # pseudo-merohedral: a ~ b
    ("P 21 21 21", (40, 50, 80, 90, 90, 90), []),
    ("P 1", (40, 50, 60, 80, 85, 95), []),
])
def test_possible_twin_laws(sg, cell, expected):
    found = [p["law"] for p in twin.possible_laws(cell, sg)]
    assert sorted(found) == sorted(expected)
    rots = twin.rotations(sg)
    for law in found:
        assert twin.det(twin.parse_law(law)) == 1 and not twin.is_symmetry(twin.parse_law(law), rots)


@needs_science
def test_refmac_equivalent_operators_are_one_twin_law():
    """From the refmacat log (S2, P 31 2 1): 'Twin operator: -K, -H, -L: ... Equivalent operators: ...'"""
    rots = twin.rotations("P 31 2 1")
    for text in "-K, -H, -L; -H,  H+K, -L;  H+K, -K, -L; -H, -K,  L;  H+K, -H,  L; -K,  H+K,  L".split(";"):
        assert _same(text, "-h,-k,l", rots), text
    for text in " H,  K,  L;  K, -H-K,  L; -H-K,  H,  L;  K,  H, -L; -H-K,  K, -L;  H, -H-K, -L".split(";"):
        assert twin.is_symmetry(twin.parse_law(text), rots), text
    rots3 = twin.rotations("P 31")                 # three different twin laws there
    assert not _same("-h,-k,l", "k,h,-l", rots3) and not _same("k,h,-l", "-k,-h,-l", rots3)


@needs_science
def test_check_law():
    cell = (89.5, 89.5, 106.4, 90, 90, 120)
    ok = twin.check_law("-K, -H, -L", cell, "P 31 2 1")
    assert ok["ok"] and ok["law"] == "-k,-h,-l" and ok["equivalent"] == "-h,-k,l" and ok["possible"] == ["-h,-k,l"]
    assert "the same law as -h,-k,l" in ok["message"]
    plain = twin.check_law("-h,-k,l", cell, "P 31 2 1")
    assert plain["ok"] and plain["law"] == "-h,-k,l" and plain["note"] is None
    for law in ("k,h,-l", "h,k,l", "h,-h-k,-l", "-h,-k,-l"):
        res = twin.check_law(law, cell, "P 31 2 1")
        assert res["code"] == "twin_law_is_symmetry" and "-h,-k,l" in res["fix"], law
    inversion = twin.check_law("h,k,-l", cell, "P 31 2 1")       # same twinning as -h,-k,l for amplitudes
    assert inversion["ok"] and inversion["law"] == "-h,-k,l" and "inversion" in inversion["note"]
    assert twin.check_law("2*h,k,l", cell, "P 31 2 1")["code"] == "twin_law_invalid"
    assert twin.check_law("x,y,z", cell, "P 31 2 1")["code"] == "twin_law_syntax"
    assert twin.check_law("k,l,h", (50, 50, 80, 90, 90, 120), "P 31")["code"] == "twin_law_invalid"
    assert twin.check_law("-h,-k,l", (40, 50, 60, 80, 85, 95), "P 1")["code"] == "twin_law_none_possible"
    assert twin.check_law("k,l,h", (50, 50, 80, 90, 90, 90), "P 41 21 2")["code"] == "twin_law_none_possible"
    friedel = twin.check_law("-h,-k,-l", cell, "P 31 2 1")
    assert "only relates Friedel mates" in friedel["message"]


@needs_science
def test_twin_law_auto():
    res = twin.check_law("auto", (89.5, 89.5, 106.4, 90, 90, 120), "P 31 2 1")
    assert res["ok"] and res["law"] == "-h,-k,l" and "auto: the only twin law of P 31 2 1" in res["message"]
    assert twin.check_law("AUTO", (50, 50, 80, 90, 90, 90), "P 41")["law"] == "-h,k,-l"
    several = twin.check_law("auto", (50, 50, 80, 90, 90, 120), "P 31")
    assert several["code"] == "twin_law_ambiguous" and several["possible"] == ["k,h,-l", "-h,-k,l", "-k,-h,-l"]
    assert "phenix.xtriage" in several["fix"]
    for sg, cell in (("P 1", (40, 50, 60, 80, 85, 95)), ("P 41 21 2", (50, 50, 80, 90, 90, 90))):
        none = twin.check_law("auto", cell, sg)
        assert none["code"] == "twin_law_none_possible" and "cannot be merohedrally twinned" in none["message"]
    assert twin.normalise_setting(" Auto ") == "auto" and twin.normalise_setting("-K,-H,-L") == "-k,-h,-l"


@needs_science
def test_twin_law_of_each_indexing_variant():
    # P 31 2 1 (PTP1B): the alternative indexing is the twin law itself, the same in every variant
    assert twin.law_for_variant("-h,-k,l", "h,k,l", "-h,-k,l", "P 31 2 1") == "-h,-k,l"
    assert twin.law_for_variant("-h,-k,l", "-h,-k,l", "h,k,l", "P 31 2 1") == "-h,-k,l"
    # pseudo-cubic P 21 21 21 (a ~ b ~ c): the twofold that swaps a and b, in a variant re-indexed by the
    # twofold that swaps b and c, is the twofold that swaps a and c
    sg = "P 21 21 21"
    rots = twin.rotations(sg)
    moved = twin.law_for_variant("-k,-h,-l", "h,k,l", "-h,-l,-k", sg)
    assert _same(moved, "-l,-k,-h", rots) and not _same(moved, "-k,-h,-l", rots)
    assert twin.law_for_variant("-k,-h,-l", "h,k,l", "-k,-h,-l", sg) == "-k,-h,-l"       # commutes with itself
    assert twin.law_for_variant("-k,-h,-l", "-h,-l,-k", "-h,-l,-k", sg) == "-k,-h,-l"    # the best variant


# ---------------------------------------------------------------------------
# What the programs write about twinning
# ---------------------------------------------------------------------------

# final.pdb of dimple 2.7 --refmac-key twin (refmacat, CCP4 9.0.017), S2 dataset 0585 (twin fraction 0.25)
REFMAC_PDB_TWIN = """\
REMARK   3  NCS RESTRAINTS STATISTICS
REMARK   3   NUMBER OF NCS GROUPS : NULL
REMARK   3
REMARK   3  TWIN DETAILS
REMARK   3   NUMBER OF TWIN DOMAINS  :    2
REMARK   3      TWIN DOMAIN   :    1
REMARK   3      TWIN OPERATOR :  H,  K,  L
REMARK   3      TWIN FRACTION : 0.751
REMARK   3      TWIN DOMAIN   :    2
REMARK   3      TWIN OPERATOR : -K, -H, -L
REMARK   3      TWIN FRACTION : 0.249
REMARK   3
"""
# S2 dataset 0795 (twin fraction 0.05): REFMAC kept no twin domain
REFMAC_PDB_NO_TWIN = """\
REMARK   3  TWIN DETAILS
REMARK   3   NUMBER OF TWIN DOMAINS  : NULL
REMARK   3
"""
REFMAC_CIF_TWIN = """\
data_final
_ccp4_reflns_twin.number_of_domains 2

loop_
_pdbx_reflns_twin.domain_id
_pdbx_reflns_twin.operator
_pdbx_reflns_twin.fraction
_pdbx_reflns_twin.type
1 " H,  K,  L" 0.7511 (pseudo)merohedral
2 "-K, -H, -L" 0.2489 (pseudo)merohedral

_ccp4_refine_ls.jelly_body        no
"""
REFMAC_CIF_NO_TWIN = "data_final\n_ccp4_reflns_twin.number_of_domains 1\n"


def test_parse_refmac_twin():
    assert twin.parse_refmac_twin(REFMAC_PDB_TWIN) == [{"law": "h,k,l", "fraction": 0.751},
                                                       {"law": "-k,-h,-l", "fraction": 0.249}]
    assert twin.parse_refmac_twin(REFMAC_PDB_NO_TWIN) == []
    assert twin.parse_refmac_twin("", REFMAC_CIF_NO_TWIN) == []
    assert twin.parse_refmac_twin("REMARK   3   FREE R VALUE    :  0.2245\n") is None
    assert twin.parse_refmac_twin("", "") is None


@needs_science
def test_parse_refmac_twin_prefers_the_mmcif():
    assert twin.parse_refmac_twin(REFMAC_PDB_TWIN, REFMAC_CIF_TWIN) == [{"law": "h,k,l", "fraction": 0.7511},
                                                                        {"law": "-k,-h,-l", "fraction": 0.2489}]


# phenix.refine 2.2.1-6174 (and 1.21.2-5419), S2 dataset 0053, xray_data.twin_law=-h,-k,l
PHENIX_PDB_TWIN = """\
REMARK   3   PROGRAM     : PHENIX (2.2.1_6174: ???)
REMARK   3  REFINEMENT TARGET : TWIN_LSQ_F
REMARK   3  TWINNING INFORMATION.
REMARK   3   FRACTION: 0.260   
REMARK   3   OPERATOR: -h,-k,l
"""
PHENIX_CIF_221 = "_pdbx_reflns_twin.operator        -h,-k,l\n_pdbx_reflns_twin.fraction        0.2600\n"
PHENIX_CIF_1212 = "_pdbx_reflns_twin.fraction        0.2600\n_pdbx_reflns_twin.operator        -h,-k,l\n"
# the end of a log of xray_data.twin_law=k,h,-l (a symmetry operation of P 31 2 1, which PHENIX accepts):
# the box is printed at the start of a macro cycle, the r_work line after it; the model says 0.480
PHENIX_LOG_T3 = """\
    update twin fraction: r(all,work,free)=0.1670 0.1652 0.1943 n_refl.: 32474 twin_fraction=0.56
r_work=0.1652 r_free=0.1943 twin_fraction=0.56 twin_law=k,h,-l
| twin fraction: 0.56  twin operator: k,h,-l                                  |
    update twin fraction: r(all,work,free)=0.1674 0.1656 0.1963 n_refl.: 32474 twin_fraction=0.48
r_work=0.1656 r_free=0.1963 twin_fraction=0.48 twin_law=k,h,-l
Final R-work = 0.1656, R-free = 0.1963
"""


def test_parse_phenix_twin_real_output():
    assert twin.parse_phenix_twin(pdb_text=PHENIX_PDB_TWIN) == {"law": "-h,-k,l", "fraction": 0.26}
    assert twin.parse_phenix_twin(cif_text=PHENIX_CIF_221) == {"law": "-h,-k,l", "fraction": 0.26}
    assert twin.parse_phenix_twin(cif_text=PHENIX_CIF_1212) == {"law": "-h,-k,l", "fraction": 0.26}
    assert twin.parse_phenix_twin(cif_text="data_refine_0053_001\n#\n" + PHENIX_CIF_221) == {"law": "-h,-k,l",
                                                                                         "fraction": 0.26}
    assert twin.parse_phenix_twin(log_text=PHENIX_LOG_T3) == {"law": "k,h,-l", "fraction": 0.48}   # the end
    box_only = "| twin fraction: 0.30  twin operator: -h,-k,l       |\n| twin fraction: 0.25  twin operator: -h,-k,l |\n"
    assert twin.parse_phenix_twin(log_text=box_only) == {"law": "-h,-k,l", "fraction": 0.25}
    assert twin.parse_phenix_twin("", "REMARK   3   R VALUE  : 0.2\n", "") is None


@needs_science
def test_summarise_domains():
    import gemmi
    rots = twin.rotations("P 31 2 1")
    domains = twin.parse_refmac_twin(REFMAC_PDB_TWIN)
    assert twin.summarise_domains(domains, "-h,-k,l", None, rots) == ("-h,-k,l", 0.249, "")
    # dimple re-indexed the data: the law is brought back into the input's indexing
    assert twin.summarise_domains(domains, "-h,-k,l", gemmi.Op("-x,-y,z"), rots) == ("-h,-k,l", 0.249, "")
    law, fraction, note = twin.summarise_domains([], "-h,-k,l", None, rots)
    assert (law, fraction) == ("", 0.0) and "refined as untwinned" in note
    assert twin.summarise_domains(None, "-h,-k,l", None, rots)[1] is None
    rots3 = twin.rotations("P 31")
    law, fraction, note = twin.summarise_domains([{"law": "h,k,l", "fraction": 0.8}, {"law": "k,h,-l", "fraction": 0.2}],
                                                 "-h,-k,l", None, rots3)
    assert (law, fraction) == ("k,h,-l", 0.2) and "not the configured -h,-k,l" in note
    law, fraction, note = twin.summarise_domains([{"law": "h,k,l", "fraction": 0.6}, {"law": "-h,-k,l", "fraction": 0.25},
                                                  {"law": "k,h,-l", "fraction": 0.15}], "-h,-k,l", None, rots3)
    assert (law, fraction) == ("-h,-k,l + k,h,-l", 0.4) and "several twin domains" in note


# ---------------------------------------------------------------------------
# refine_datasets: the configured installation, twin refinement
# ---------------------------------------------------------------------------

def _test_refine():
    """test_refine.py's synthetic inputs and stand-in programs."""
    pytest.importorskip("gemmi")
    pytest.importorskip("reciprocalspaceship")
    import test_refine
    return test_refine


def test_refinement_runs_in_the_configured_installation(tmp_path, monkeypatch):
    TR = _test_refine()
    from valdo.easy import refine
    script = _write(tmp_path / "fake_program.py", TR.FAKE_PROGRAM)
    env_log = tmp_path / "env.log"
    dimple = textwrap.dedent(f"""\
        #!/bin/sh
        if [ "$1" = "--version" ]; then echo 'dimple 2.7.0 (fake)'; exit 0; fi
        printf '%s|%s|%s\\n' "${{VALDO_TEST_MARKER-}}" "${{PYTHONPATH-unset}}" "${{CCP4-}}" >> "{env_log}"
        exec "{sys.executable}" "{script}" dimple "$@"
        """)
    root = fake_ccp4(tmp_path, program=dimple)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")              # no dimple on PATH
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "valdo-python"))
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("FAKE_BEHAVIOUR", "{}")
    monkeypatch.delenv("FAKE_TRUTH_DIR", raising=False)
    info = programs.resolve("dimple", str(root))
    jobs = TR._jobs(tmp_path, [("0001_0", dict(free="phenix"))])
    messages = []
    table = refine.refine_datasets(jobs, "dimple", str(tmp_path / "out"), log=messages.append, program_info=info)
    row = table.iloc[0]
    assert row["status"] == "ok", row["reason"]
    assert row["program_version"] == "CCP4 9.0.017, dimple 2.7.0" and row["twin_law"] == ""
    assert env_log.read_text().split() == [f"ccp4|unset|{root}"]   # the installation's environment, no PYTHONPATH
    where = f"CCP4 9.0.017, dimple 2.7.0 (dimple) at {root} (programs.ccp4 in valdo.yaml)"
    assert f"[refine] program: {where}" in messages
    assert f"# program: {where}" in (tmp_path / "out" / "runs" / "0001_0" / "refine.log").read_text()
    with pytest.raises(refine.RefineError, match="programs.ccp4"):
        refine.refine_datasets(jobs, "dimple", str(tmp_path / "o2"), log=quiet,
                               program_info=programs.resolve("dimple", str(tmp_path / "nowhere")))
    with pytest.raises(refine.RefineError, match="describes 'dimple'"):
        refine.refine_datasets(jobs, "phenix", str(tmp_path / "o3"), log=quiet, program_info=info)


TWIN_DIMPLE = r'''
"""Runs after test_refine's stand-in dimple: what dimple 2.7 adds (pointless's reindexed.mtz and
alt_reindex record; with --refmac-key twin, REFMAC's TWIN DETAILS in final.pdb)."""
import json, os, sys
import gemmi
import reciprocalspaceship as rs

args = sys.argv[1:]
if args[:1] == ["--version"]:
    sys.exit(0)
mtz, pdb, outdir = args[0], args[1], args[2]
stem = os.path.basename(os.path.dirname(os.path.abspath(outdir)))
mode = json.loads(os.environ.get("FAKE_BEHAVIOUR", "{}")).get(stem, "ok")
kind = json.loads(os.environ.get("FAKE_TWIN", "{}")).get(stem, "twin")
final = os.path.join(outdir, "final.pdb")
if not os.path.isfile(final):
    sys.exit(0)
data = rs.read_mtz(mtz)
op, other = "[h,k,l]", "[-h,-k,l]"
if mode == "reindex":                     # pointless re-indexed the data to match the model
    data = data.apply_symop(gemmi.Op("-x,-y,z")).hkl_to_asu().drop(columns=["M/ISYM"])
    op, other = other, op
if kind not in ("nothing", "log_only", "nolog"):     # nothing: dimple did not run pointless
    data.write_mtz(os.path.join(outdir, "reindexed.mtz"))
if kind not in ("nothing", "nolog"):
    with open(os.path.join(outdir, "dimple.log"), "a") as fh:
        fh.write("[pointless]\nargs: HKLIN ../input.mtz HKLOUT reindexed.mtz XYZIN ini.pdb\nalt_reindex: "
                 + json.dumps([{"op": op, "cc": 0.681, "cell_deviat": 0.1},
                               {"op": other, "cc": 0.678, "cell_deviat": 0.1}]) + "\n\n[refmacat restr]\n")
if kind == "nolog" and os.path.isfile(os.path.join(outdir, "dimple.log")):
    os.remove(os.path.join(outdir, "dimple.log"))
if "--refmac-key" in args and args[args.index("--refmac-key") + 1] == "twin":
    lines = ["REMARK   3  TWIN DETAILS"]
    if kind == "null":
        lines.append("REMARK   3   NUMBER OF TWIN DOMAINS  : NULL")
    else:
        lines.append("REMARK   3   NUMBER OF TWIN DOMAINS  :    2")
        for n, (law, frac) in enumerate(((" H,  K,  L", "0.751"), ("-K, -H, -L", "0.249")), start=1):
            lines += [f"REMARK   3      TWIN DOMAIN   :    {n}", f"REMARK   3      TWIN OPERATOR : {law}",
                      f"REMARK   3      TWIN FRACTION : {frac}"]
    with open(final) as fh:
        text = fh.read()
    with open(final, "w") as fh:
        fh.write("\n".join(lines) + "\n" + text)
'''

TWIN_PHENIX = r'''
"""Runs after test_refine's stand-in phenix.refine: the twin records as PHENIX 2.2.1-6174 writes them
(model REMARK 3 TWINNING INFORMATION block; log: the '| twin fraction: ... |' box at the start of a
macro cycle and the 'r_work=... twin_fraction=... twin_law=...' line after it). A bare twin_law= stops
the run, as in PHENIX (it is also neutron_data.twin_law)."""
import json, os, sys

args = sys.argv[1:]
bare = [a for a in args if a.startswith("twin_law=")]
if bare:
    print("Sorry: Ambiguous parameter definition: twin_law = " + bare[0].split("=", 1)[1])
    sys.exit(1)
prefixes = [a.split("=", 1)[1] for a in args if a.startswith("output.prefix=")]
laws = [a.split("=", 1)[1] for a in args if a.startswith("xray_data.twin_law=")]
if not prefixes or not laws:
    sys.exit(0)
prefix = prefixes[0]
kind = json.loads(os.environ.get("FAKE_TWIN", "{}")).get(os.path.basename(os.getcwd()), "pdb")
if kind == "pdb":
    with open(prefix + "_001.pdb") as fh:
        text = fh.read()
    with open(prefix + "_001.pdb", "w") as fh:
        fh.write("REMARK   3  TWINNING INFORMATION.\nREMARK   3   FRACTION: 0.312   \n"
                 f"REMARK   3   OPERATOR: {laws[0]}\n" + text)
elif kind == "log":      # PHENIX 2.2.1, T3f: the box shows the start of the last cycle (0.56), not the end
    with open(prefix + "_001.log", "a") as fh:
        fh.write(f"r_work=0.1652 r_free=0.1943 twin_fraction=0.56 twin_law={laws[0]}\n"
                 f"| twin fraction: 0.56  twin operator: {laws[0]}                                  |\n"
                 "    update twin fraction: r(all,work,free)=0.1674 0.1656 0.1963 n_refl.: 32474 twin_fraction=0.48\n"
                 f"r_work=0.1656 r_free=0.1963 twin_fraction=0.48 twin_law={laws[0]}\n")
'''


@pytest.fixture
def twin_fakes(tmp_path, monkeypatch):
    """test_refine's stand-in dimple and phenix.refine, each followed by its twin script, first on PATH."""
    TR = _test_refine()
    bindir = tmp_path / "fakebin"
    fake = _write(tmp_path / "fake_program.py", TR.FAKE_PROGRAM)
    for prog, post in (("dimple", TWIN_DIMPLE), ("phenix.refine", TWIN_PHENIX)):
        after = _write(tmp_path / f"twin_{prog.split('.')[0]}.py", post)
        _write(bindir / prog, f'#!/bin/sh\n"{sys.executable}" "{fake}" {prog} "$@" || exit $?\n'
                              f'exec "{sys.executable}" "{after}" "$@"\n', executable=True)
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"))
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("FAKE_BEHAVIOUR", "{}")
    monkeypatch.setenv("FAKE_TWIN", "{}")
    monkeypatch.delenv("FAKE_TRUTH_DIR", raising=False)

    class Fakes:
        tr = TR

        @staticmethod
        def behave(modes=None, twin=None):
            monkeypatch.setenv("FAKE_BEHAVIOUR", json.dumps(modes or {}))
            monkeypatch.setenv("FAKE_TWIN", json.dumps(twin or {}))

        @staticmethod
        def calls():
            if not calls.exists():
                return []
            return [json.loads(line) for line in calls.read_text().splitlines() if line.strip()]

    return Fakes()


# dimple.log of dimple 2.7 (CCP4 9) on S2 dataset 0585 given in the other indexing (-h,-k,l)
DIMPLE_LOG_REINDEXED = """\
[rwcontents]
num_mol: 6

[pointless]
start_time: 2026-10-24 14:45:55
input: TOLERANCE 5
  end
prog: /wynton/home/jo/achoe/valdo-twin/software/ccp4-9/bin/pointless
args: HKLIN ../input.mtz HKLOUT reindexed.mtz XYZIN ini.pdb
end_time: 2026-10-24 14:45:56
alt_reindex: [{"op": "[-h,-k,l]", "cc": 0.895, "cell_deviat": 0.14}, {"op": "[h,k,l]", "cc": 0.334, "cell_deviat": 0.14}]
output_cell: (89.52, 89.52, 106.45, 90.0, 90.0, 120.0)

[refmac5 rigid]
start_time: 2026-10-24 14:45:56
"""


@needs_science
def test_dimple_log_says_whether_dimple_reindexed(tmp_path):
    from valdo.easy import refine
    assert refine._dimple_used_reindexed(DIMPLE_LOG_REINDEXED)
    assert not refine._dimple_used_reindexed(DIMPLE_LOG_REINDEXED.replace("alt_reindex:", "other:"))
    assert not refine._dimple_used_reindexed("[pointless]\nalt_reindex: []\n")          # dimple: "no good indexing"
    assert not refine._dimple_used_reindexed("[workflow]\nversion: 2.7\n\n[refmac5 rigid]\n")   # pointless did not run
    # pointless failed (dimple's workflow writes exit_status only then): dimple refined the input as it was
    failed = DIMPLE_LOG_REINDEXED.replace("end_time: 2026-10-24 14:45:56\n", "end_time: 2026-10-24 14:45:56\nexit_status: 1\n")
    assert "exit_status: 1" in failed and not refine._dimple_used_reindexed(failed)
    path = tmp_path / "dimple.log"
    path.write_text(DIMPLE_LOG_REINDEXED)
    assert refine._dimple_log_reindex_op(str(path)).triplet() == "-x,-y,z"
    path.write_text(DIMPLE_LOG_REINDEXED.replace('"[-h,-k,l]", "cc": 0.895', '"[h,k,l]", "cc": 0.895', 1))
    assert refine._dimple_log_reindex_op(str(path)).triplet() == "x,y,z"


def test_dimple_twin_refinement(tmp_path, twin_fakes):
    import pandas as pd
    from valdo.easy import refine
    TR = twin_fakes.tr
    stems = ["0001_0", "0002_0", "0003_1", "0004_1", "0005_0", "0006_1", "0007_0"]
    jobs = TR._jobs(tmp_path, [(s, dict(free="phenix")) for s in stems])
    modes = {"0003_1": "reindex", "0004_1": "reindex", "0006_1": "reindex"}
    twin_fakes.behave(modes, twin={"0002_0": "null", "0004_1": "log_only", "0005_0": "nothing",
                                   "0006_1": "nothing", "0007_0": "nolog"})
    out = tmp_path / "04_refine"
    messages = []
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=3, log=messages.append,
                                   twin_law="-h,-k,l").set_index("stem")
    argv = {c["stem"]: c["argv"] for c in twin_fakes.calls()}
    assert argv["0001_0"][-2:] == ["--refmac-key", "twin"]
    assert any("twin refinement, twin law -h,-k,l" in m for m in messages)
    # REFMAC's -K, -H, -L is the configured law in P 31 2 1
    row = table.loc["0001_0"]
    assert row["status"] == "ok" and row["reindex_op"] == "", row["reason"]
    assert (row["twin_law"], row["twin_fraction"], row["twin_note"]) == ("-h,-k,l", 0.249, "")
    # REFMAC kept no twin domain
    row = table.loc["0002_0"]
    assert row["status"] == "ok" and (row["twin_law"], row["twin_fraction"]) == ("", 0.0)
    assert "refined as untwinned" in row["twin_note"]
    # dimple re-indexed the data: told by reindexed.mtz (0003_1) or by dimple.log (0004_1)
    for stem in ("0003_1", "0004_1"):
        row = table.loc[stem]
        assert row["status"] == "ok" and row["reindex_op"] == "-x,-y,z", (stem, row["reason"])
        assert (row["twin_law"], row["twin_fraction"]) == ("-h,-k,l", 0.249)
        op, cc, _ = refine.find_frame_op(jobs[stems.index(stem)]["mtz"], str(out / "phases" / f"{stem}.mtz"),
                                         "F-obs", f_refined="F")
        assert op.triplet() == "x,y,z" and cc > 0.999          # back in the input's indexing
    # pointless did not run (no [pointless] record): dimple refined the data as given
    row = table.loc["0005_0"]
    assert row["status"] == "ok" and row["reindex_op"] == "" and row["twin_law"] == "-h,-k,l", row["reason"]
    # ... but if the data were re-indexed all the same, the amplitude check stops it
    row = table.loc["0006_1"]
    assert row["status"] == "failed" and "does not match the input data" in row["reason"]
    # no dimple.log: unknown, so the dataset fails rather than risk misplaced phases
    row = table.loc["0007_0"]
    assert row["status"] == "failed" and "dimple wrote no dimple.log" in row["reason"]
    summary = pd.read_csv(out / "refine_summary.csv", dtype=str, keep_default_na=False).set_index("stem")
    assert summary.loc["0001_0", "twin_law"] == "-h,-k,l" and float(summary.loc["0001_0", "twin_fraction"]) == 0.249
    # the same twin law again: only the failed datasets run; the others keep their twin results
    n_calls = len(twin_fakes.calls())
    twin_fakes.behave(modes, twin={"0002_0": "null", "0004_1": "log_only", "0005_0": "nothing"})
    table = refine.refine_datasets(jobs, "dimple", str(out), ncpu=3, log=quiet, twin_law="-h,-k,l").set_index("stem")
    assert sorted(c["stem"] for c in twin_fakes.calls()[n_calls:]) == ["0006_1", "0007_0"]
    assert table.loc[stems[:5], "skipped"].all() and (table["status"] == "ok").all()
    assert table.loc["0001_0", "twin_fraction"] == 0.249 and table.loc["0006_1", "reindex_op"] == "-x,-y,z"
    # without a twin law it is another refinement: refined again
    table = refine.refine_datasets(jobs[2:4], "dimple", str(out), ncpu=2, log=quiet).set_index("stem")
    assert not table["skipped"].any() and (table["twin_law"] == "").all()
    assert table["status"].tolist() == ["ok"] * 2 and table["reindex_op"].tolist() == ["-x,-y,z"] * 2
    assert all("--refmac-key" not in c["argv"] for c in twin_fakes.calls()[-2:])


def test_phenix_twin_refinement(tmp_path, twin_fakes):
    from valdo.easy import refine
    TR = twin_fakes.tr
    jobs = TR._jobs(tmp_path, [(s, dict(free="phenix")) for s in ("0001_0", "0002_0", "0003_0")])
    twin_fakes.behave(twin={"0002_0": "log", "0003_0": "none"})
    out = tmp_path / "04_refine"
    table = refine.refine_datasets(jobs, "phenix", str(out), ncpu=3, log=quiet, twin_law="-K, -H, -L",
                                   extra_args=["main.number_of_macro_cycles=1"]).set_index("stem")
    argv = {c["stem"]: c["argv"] for c in twin_fakes.calls()}
    # xray_data.twin_law (a bare twin_law= is ambiguous in PHENIX), the user's arguments, '--overwrite' last
    assert argv["0001_0"][-3:] == ["xray_data.twin_law=-k,-h,-l", "main.number_of_macro_cycles=1", "--overwrite"]
    assert table["status"].tolist() == ["ok"] * 3
    assert table.loc["0001_0", ["twin_law", "twin_fraction"]].tolist() == ["-k,-h,-l", 0.312]   # model REMARK 3
    assert table.loc["0002_0", ["twin_law", "twin_fraction"]].tolist() == ["-k,-h,-l", 0.48]    # last r_work line
    row = table.loc["0003_0"]
    assert row["twin_law"] == "-k,-h,-l" and math.isnan(row["twin_fraction"])
    assert "twin fraction not found" in row["twin_note"]
    with pytest.raises(refine.RefineError, match="given twice"):
        refine.refine_datasets(jobs, "phenix", str(out), log=quiet, twin_law="-h,-k,l", extra_args=["twin_law=-h,-k,l"])
    with pytest.raises(refine.RefineError, match="given twice"):
        refine.refine_datasets(jobs, "phenix", str(out), log=quiet, twin_law="-h,-k,l",
                               extra_args=["xray_data.twin_law=-h,-k,l"])
    with pytest.raises(refine.RefineError, match="resolved first"):
        refine.refine_datasets(jobs, "phenix", str(out), log=quiet, twin_law="auto")
    # what PHENIX says to a bare twin_law= (valdo.yaml refuses it in extra_args; this is the program's answer)
    bare = refine.refine_datasets(jobs[:1], "phenix", str(tmp_path / "bare"), log=quiet,
                                  extra_args=["twin_law=-h,-k,l"]).set_index("stem")
    assert bare.loc["0001_0", "status"] != "ok" and "Ambiguous parameter definition" in bare.loc["0001_0", "reason"]
    with pytest.raises(refine.RefineError, match="given twice"):
        refine.refine_datasets(jobs, "dimple", str(out), log=quiet, twin_law="-h,-k,l",
                               extra_args=["--refmac-key", "twin"])
    with pytest.raises(refine.RefineError, match="twin law"):
        refine.refine_datasets(jobs, "phenix", str(out), log=quiet, twin_law="x,y,z")


@needs_science
def test_upstream_filter_reads_a_refine_summary_with_twin_columns(tmp_path):
    """valdo.pipeline's run_filter (upstream, unchanged) on refine_summary.csv as valdo-easy writes it with
    twin refinement: notes with commas and quotes, an empty law, NaN fractions, a failed variant."""
    import gemmi
    import numpy as np
    from types import SimpleNamespace
    from valdo.commandline.pipeline import run_filter
    from valdo.easy import refine
    scaled = tmp_path / "scaled"
    scaled.mkdir()
    stems = ("0001_0", "0001_1", "0002_0", "0002_1")
    for stem in stems:
        m = gemmi.Mtz(with_base=True)
        m.spacegroup = gemmi.SpaceGroup("P 1")
        m.set_cell_for_all(gemmi.UnitCell(30, 30, 30, 90, 90, 90))
        m.add_dataset("d")
        for label, kind in (("F-obs", "F"), ("SIGF-obs", "Q"), ("F-obs-scaled", "F")):
            m.add_column(label, kind)
        hkl = np.array([(h, k, l) for h in range(4) for k in range(-3, 4) for l in range(-3, 4) if (h, k, l) != (0, 0, 0)])
        m.set_data(np.hstack([hkl, np.full((len(hkl), 3), 10.0)]).astype(np.float32))
        m.write_to_file(str(scaled / f"{stem}.mtz"))
    ctx = SimpleNamespace(program="dimple", out_dir=str(tmp_path), version="CCP4 9.0.017, dimple 2.7")
    jobs = [{"stem": s, "id": s[:4], "symop": s[-1]} for s in stems]
    results = {
        "0001_0": refine._record("0001_0", ctx, jobs[0], status="ok", Rf_final=0.22, twin_law="-h,-k,l",
                                 twin_fraction=0.249, twin_note='REFMAC kept twin law k,h,-l, not "the configured" -h,-k,l'),
        "0001_1": refine._record("0001_1", ctx, jobs[1], status="ok", Rf_final=0.23, twin_law="-h,-k,l + k,h,-l",
                                 twin_fraction=0.75),
        "0002_0": refine._record("0002_0", ctx, jobs[2], status="ok", Rf_final=0.60),
        "0002_1": refine._record("0002_1", ctx, jobs[3], status="failed", reason="dimple exited with code 1: a, b"),
    }
    refine._write_summary(jobs, results, str(tmp_path), ctx, partial=False)
    for name in ("refine_summary.csv", "refine_summary_ok.csv"):
        out = tmp_path / f"kept_{name}.txt"
        run_filter({"refine_summary": str(tmp_path / name), "reindex_record": None, "scaled_dir": str(scaled),
                    "metrics": None, "max_rfree": 0.45, "min_cc": 0.0, "output": str(out), "_force": True})
        kept = sorted(os.path.basename(x.strip()) for x in out.read_text().splitlines() if x.strip())
        assert kept[:2] == ["0001_0.mtz", "0001_1.mtz"] and "0002_0.mtz" not in kept, (name, kept)


# ---------------------------------------------------------------------------
# valdo.yaml, stage config, job script, report
# ---------------------------------------------------------------------------

MINIMAL = "data: mtz\nmodels:\n  apo: apo.pdb\n"
REFINE = MINIMAL + "phases:\n  source: refine\n  program: dimple\n"


def test_project_programs_and_twin_law(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    from valdo.easy import project as P
    path = tmp_path / "valdo.yaml"
    path.write_text(REFINE + "  twin_law: -K, -H, -L\nprograms:\n  ccp4: ccp4-9\n")
    proj, notes = P.load_project(str(path))
    assert proj["phases"]["twin_law"] == "-k,-h,-l" and any("read the twin law" in n for n in notes)
    assert proj["programs"] == {"phenix": None, "ccp4": str(tmp_path / "ccp4-9")}   # relative to the file
    # a pasted setup line is read as the script's path
    path.write_text(MINIMAL + "programs:\n  phenix: source /opt/phenix-2.2-6143/phenix_env.sh\n"
                    "  ccp4: '. \"/opt/ccp4 9/bin/ccp4.setup-sh\"'\n")
    pasted, notes = P.load_project(str(path))
    assert pasted["programs"] == {"phenix": "/opt/phenix-2.2-6143/phenix_env.sh", "ccp4": "/opt/ccp4 9/bin/ccp4.setup-sh"}
    assert any("programs.phenix: read 'source /opt/phenix-2.2-6143/phenix_env.sh' as the path" in n for n in notes)
    assert programs.clean_setting("/opt/ccp4-9") == ("/opt/ccp4-9", "")
    for text, fragment in (
            (MINIMAL + "phases:\n  twin_law: -h,-k,l\n", "only used for twin refinement"),
            (REFINE + "  twin_law: x,y,z\n", "uses x, y, z"),
            (REFINE + "  twin_law: 3\n", "must be a twin law"),
            (REFINE + "  twin_law: -h,-k,l\nadvanced:\n  refine:\n    extra_args: ['--refmac-key', 'twin']\n",
             "set it in one place"),
            (MINIMAL + "programs:\n  coot: /opt/coot\n", "programs.coot: unknown setting")):
        path.write_text(text)
        with pytest.raises(P.ProjectError) as err:
            P.load_project(str(path))
        assert fragment in "\n".join(err.value.messages), text
    # the writer: programs in part 1, notes from init after a setting's comment, a round trip
    plain = {k: v for k, v in proj.items() if not k.startswith("_")}
    out = tmp_path / "written.yaml"
    P.write_project(plain, str(out), comments={"programs.ccp4": "found on PATH by valdo init"})
    text = out.read_text()
    part1, part2 = text.index("1. YOUR PROJECT"), text.index("2. DETECTED BY")
    assert part1 < text.index("programs:") < part2
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if line.strip().startswith("ccp4:"))
    assert lines[at - 1].strip() == "# found on PATH by valdo init"     # on its own line, above the setting
    again = yaml.safe_load(text)
    assert again["phases"]["twin_law"] == "-k,-h,-l" and again["programs"]["ccp4"] == str(tmp_path / "ccp4-9")
    default = yaml.safe_load(P.render_project(P.default_project()))
    assert default["programs"] == {"phenix": None, "ccp4": None} and default["phases"]["twin_law"] is None


def test_refine_stage_config_names_the_twin_law_only_when_set(tmp_path):
    from types import SimpleNamespace
    from valdo.easy import stages
    from valdo.easy.project import default_project
    p = default_project()
    p["phases"].update(source="refine", program="dimple")
    p["programs"]["ccp4"] = str(tmp_path / "ccp4-9")
    ctx = SimpleNamespace(project=p, dir=lambda rel: f"/w/{rel}", path=lambda rel: f"/w/{rel}", ncpu=4,
                          inputs=None)
    cfg = stages.build_refine(ctx)
    assert "twin_law" not in cfg            # unchanged for projects without twin refinement
    assert "ccp4-9" not in json.dumps(cfg)   # where the program lives is not part of the stage config
    p["phases"]["twin_law"] = "-h,-k,l"
    assert stages.build_refine(ctx)["twin_law"] == "-h,-k,l"
    p["phases"]["twin_law"] = "auto"         # no reference lattice: stays auto, and the refine stage refuses it
    assert stages.build_refine(ctx)["twin_law"] == "auto"
    with pytest.raises(stages.StageError, match="auto"):
        stages.execute_refine(SimpleNamespace(project=p, scratch={}), {"twin_law": "auto"}, None)


def _lattice_inputs(cell, sg):
    from types import SimpleNamespace
    return SimpleNamespace(reference={"name": "ref", "id": "0001"}, names=["ref"],
                           headers=[{"cell": cell, "sg_name": sg}], twin=None)


@needs_science
def test_refine_stage_resolves_auto_and_gives_each_variant_its_law():
    from types import SimpleNamespace
    from valdo.easy import check, stages
    from valdo.easy.project import default_project
    p = default_project()
    p["phases"].update(source="refine", program="phenix", twin_law="auto")
    ctx = SimpleNamespace(project=p, dir=lambda rel: f"/w/{rel}", path=lambda rel: f"/w/{rel}", ncpu=4,
                          inputs=_lattice_inputs((36, 36, 44, 90, 90, 120), "P 3 2 1"))
    assert stages.build_refine(ctx)["twin_law"] == "-h,-k,l"          # auto: the lattice's only law
    p["phases"]["twin_law"] = "h,k,-l"                                 # with an inversion: the same law without
    assert stages.build_refine(ctx)["twin_law"] == "-h,-k,l"
    p["phases"]["twin_law"] = "auto"
    ctx.inputs = _lattice_inputs((50, 50, 80, 90, 90, 120), "P 31")     # several laws: not resolved
    assert stages.build_refine(ctx)["twin_law"] == "auto" and check.resolve_twin_law(p, None) == "auto"
    # pseudo-cubic P 21 21 21: variants re-indexed by an operator that does not commute with the law
    sg, law = "P 21 21 21", "-k,-h,-l"
    ops_hkl = ["h,k,l", "-k,-h,-l", "-h,-l,-k"]
    ctx = SimpleNamespace(inputs=_lattice_inputs((50, 50, 50.1, 90, 90, 90), sg), reindexed=True, ops_hkl=ops_hkl,
                          outputs=lambda stage: {"best_symop": {"0001": 0, "0002": 2}})
    jobs = [{"stem": f"{i}_{v}"} for i in ("0001", "0002") for v in range(3)]
    stages._variant_twin_laws(ctx, {"twin_law": law}, jobs)
    got = {j["stem"]: j.get("twin_law") for j in jobs}
    assert got["0001_0"] is None and got["0001_1"] is None and got["0002_2"] is None
    for stem, best in (("0001_2", 0), ("0002_0", 2), ("0002_1", 2)):
        variant = int(stem[-1])
        assert got[stem] == twin.law_for_variant(law, ops_hkl[best], ops_hkl[variant], sg) != law, stem
    # without reindexing every job refines the configured law
    plain = [{"stem": "0001"}]
    stages._variant_twin_laws(SimpleNamespace(reindexed=False), {"twin_law": law}, plain)
    assert "twin_law" not in plain[0]


def test_refine_stage_counts_datasets_that_look_twinned(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from valdo.easy import stages
    from valdo.easy.refine import TWIN_HINT
    out = tmp_path / "04_refine"
    out.mkdir()
    frame = "could not relate the refined MTZ to the input data (best amplitude CC 1.000; tried x,y,z: 1.000, -x,-y,z: 0.912)"
    # 0004_0: a phenix.refine result kept in its input indexing, with the twin hint (refine._phenix_identity)
    kept = ("the refined amplitudes also fit -x,-y,z (CC 0.921); phenix.refine keeps the input indexing "
            f"(x,y,z, CC 1.000), so that one was used; {TWIN_HINT}")
    pd.DataFrame([{"stem": "0001_0", "status": "failed", "reason": f"{frame}; {TWIN_HINT}", "Rf_final": ""},
                  {"stem": "0002_0", "status": "ok", "reason": "", "Rf_final": "0.25"},
                  {"stem": "0003_0", "status": "failed", "reason": "phenix.refine exited with code 1", "Rf_final": ""},
                  {"stem": "0004_0", "status": "ok", "reason": kept, "Rf_final": "0.27"}]
                 ).to_csv(out / "refine_summary.csv", index=False)
    ids = ("0001", "0002", "0003", "0004")
    ctx = SimpleNamespace(alive=lambda: [{"id": i} for i in ids], scratch={},
                          outputs=lambda stage: {"stems": {i: [f"{i}_0"] for i in ids}})
    res = stages.Result()
    res.n_in = 4
    stages.post_refine(ctx, {"out_dir": str(out)}, res)
    assert set(res.dropped) == {"0001", "0003"} and TWIN_HINT in res.dropped["0001"]
    assert [m for m in res.messages if "fits two indexings" in m] == [
        "2 dataset(s) have a refined variant that fits two indexings, as twinned data do (0001, 0004): if the "
        "crystals are twinned, set phases.twin_law ('valdo check' lists the possible laws)"]


@pytest.mark.parametrize("program, text", [
    ("dimple", "1 refined as untwinned (REFMAC kept no twin domain)"),
    # real phenix.refine 1.21.2 on an untwinned synthetic dataset: 'FRACTION: 0.000' (no REFMAC involved)
    ("phenix", "1 with a refined twin fraction of 0"),
])
def test_refine_stage_twin_summary_names_the_program(tmp_path, program, text):
    import pandas as pd
    from types import SimpleNamespace
    from valdo.easy import stages
    out = tmp_path / "04_refine"
    out.mkdir()
    pd.DataFrame([{"stem": "0001_0", "status": "ok", "reason": "", "Rf_final": "0.09", "twin_law": "",
                   "twin_fraction": "0.0", "twin_note": ""},
                  {"stem": "0002_0", "status": "ok", "reason": "", "Rf_final": "0.08", "twin_law": "-h,-k,l",
                   "twin_fraction": "0.26", "twin_note": ""}]).to_csv(out / "refine_summary.csv", index=False)
    ctx = SimpleNamespace(alive=lambda: [{"id": i} for i in ("0001", "0002")], scratch={},
                          outputs=lambda stage: {"stems": {i: [f"{i}_0"] for i in ("0001", "0002")}})
    res = stages.Result()
    res.n_in = 2
    stages.post_refine(ctx, {"out_dir": str(out), "program": program, "twin_law": "-h,-k,l"}, res)
    summary = [m for m in res.messages if m.startswith("twin refinement")]
    assert summary == ["twin refinement (-h,-k,l): median twin fraction 0.130 (range 0.000-0.260) over 2 refined "
                       f"variant(s); {text}"]
    assert program == "dimple" or "REFMAC" not in summary[0]


@needs_science
def test_twin_checks_for_auto_several_laws_and_unknown_lattices():
    from valdo.easy import check

    def run(law, cell, sg):
        project = {"phases": {"source": "refine", "program": "phenix", "twin_law": law}}
        inp = _lattice_inputs(cell, sg)
        if cell is None:
            inp.headers = [{"cell": None, "sg_name": None}]
        return {f["code"]: f for f in check._check_twin(project, inp)}, inp

    found, inp = run("auto", (36, 36, 44, 90, 90, 120), "P 3 2 1")
    assert "auto: the only twin law of P 3 2 1" in found["twin_law"]["what"] and inp.twin["law"] == "-h,-k,l"
    found, _ = run("auto", (50, 50, 80, 90, 90, 120), "P 31")
    f = found["twin_law_ambiguous"]
    assert f["level"] == "error" and "k,h,-l, -h,-k,l, -k,-h,-l" in f["fix"] and "phenix.xtriage" in f["fix"]
    found, _ = run("auto", (50, 50, 80, 90, 90, 90), "P 41 21 2")
    assert found["twin_law_none_possible"]["level"] == "error"
    found, _ = run(None, (50, 50, 80, 90, 90, 120), "P 31")              # off: every possible law in the fix
    assert "k,h,-l, -h,-k,l, -k,-h,-l" in found["twin_law"]["fix"] and "phenix.xtriage" in found["twin_law"]["fix"]
    found, _ = run(None, (36, 36, 44, 90, 90, 120), "P 3 2 1")
    assert 'phases.twin_law: "-h,-k,l"' in found["twin_law"]["fix"] and "auto" in found["twin_law"]["fix"]
    found, _ = run("-h,-k,l", None, None)                                  # no lattice to check against
    assert found["twin_law_unchecked"]["level"] == "warning"
    found, _ = run("auto", None, None)
    assert found["twin_law_unchecked"]["level"] == "error"
    assert run(None, None, None)[0] == {}


def test_filter_keeps_the_indexing_twin_refinement_can_tell_apart():
    """With twin refinement R-free cannot choose between indexings (the twin fraction becomes 1 - f
    in the other one): the lower twin fraction decides, else the reindex CC; without a twin law the
    R-free rule is unchanged."""
    from valdo.easy import stages

    class Ctx:
        reindexed = True

        def __init__(self, refine_out):
            self._out = {"scale": {"stems": {"0001": ["0001_0"], "0002": ["0002_0", "0002_1"],
                                             "0003": ["0003_0", "0003_1"], "0004": ["0004_0", "0004_1"]}},
                         "reindex": {"best_symop": {"0001": 0, "0002": 0, "0003": 1, "0004": 0}},
                         "refine": refine_out}

        def outputs(self, name):
            return self._out[name]

        def stage_applicable(self, name):
            return True

        def alive(self):
            return [{"id": i} for i in ("0001", "0002", "0003", "0004")]

    rfree = {"0001_0": 0.25, "0002_0": 0.25, "0002_1": 0.24, "0003_0": 0.25, "0003_1": 0.26,
             "0004_0": 0.25, "0004_1": 0.24}
    refine_out = {"rfree": rfree, "status": {s: "ok" for s in rfree}}
    plain = {i: c for i, (c, _) in stages._filter_choice(Ctx(refine_out)).items()}
    assert plain == {"0001": "0001_0", "0002": "0002_1", "0003": "0003_0", "0004": "0004_1"}   # lowest R-free
    twin_out = dict(refine_out, twin_law="-h,-k,l", twin={
        "0001_0": ["-h,-k,l", 0.3, ""],
        "0002_0": ["-h,-k,l", 0.3, ""], "0002_1": ["-h,-k,l", 0.7, ""],      # the fraction decides
        "0003_0": ["-h,-k,l", 0.49, ""], "0003_1": ["-h,-k,l", 0.51, ""],    # too close: reindex CC
        "0004_0": ["-h,-k,l", None, ""], "0004_1": ["-h,-k,l", 0.2, ""]})    # unknown: reindex CC
    choice = stages._filter_choice(Ctx(twin_out))
    assert {i: c for i, (c, _) in choice.items()} == {"0001": "0001_0", "0002": "0002_0", "0003": "0003_1",
                                                       "0004": "0004_0"}
    assert "lowest refined twin fraction" in choice["0002"][1]["0002_0"][2]
    assert choice["0002"][1]["0002_1"][0] == 99.0 and choice["0002"][1]["0002_0"][0] == 0.25


def test_jobscript_says_where_the_refinement_program_comes_from(tmp_path, monkeypatch):
    from valdo.easy import cli, jobscript
    from valdo.easy.project import default_project
    p = default_project()
    p["phases"].update(source="refine", program="dimple")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    notes, warnings = cli._job_program_notes(p, None)
    assert notes == [] and "programs.ccp4 is not set" in warnings[0] and "would stop" in warnings[0]
    root = fake_ccp4(tmp_path)
    monkeypatch.setenv("PATH", str(root / "bin") + os.pathsep + "/usr/bin:/bin")
    notes, warnings = cli._job_program_notes(p, None)
    assert "dimple from PATH" in notes[0] and "may differ" in warnings[0]
    notes, warnings = cli._job_program_notes(p, ["source /opt/ccp4-9/bin/ccp4.setup-sh"])
    assert "set up by the environment lines above" in notes[0] and warnings == []
    p["programs"]["ccp4"] = str(root)
    notes, warnings = cli._job_program_notes(p, None)
    assert "programs.ccp4 in valdo.yaml" in notes[0] and "no setup line is needed" in " ".join(notes) and not warnings
    text = jobscript.render("sge", str(tmp_path / "valdo.yaml"), notes=notes, log_dir=str(tmp_path))
    assert "# Refinement program: programs.ccp4 in valdo.yaml" in text
    p["phases"]["source"] = "input_mtz"
    assert cli._job_program_notes(p, None) == ([], [])


def test_jobscript_keeps_long_paths_whole_and_warns_about_variables(tmp_path, monkeypatch):
    from valdo.easy import cli, jobscript
    from valdo.easy.project import default_project
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    deep = tmp_path / "lab-software" / "x-ray-crystallography" / "programs-2026" / "phenix-2.2.1-6174-linux-x86-64"
    root = fake_phenix(deep.parent, version="2.2.1-6174")
    root.rename(deep)
    _write(deep / "phenix_env.sh", f'export PHENIX="{deep}"\nexport PATH="$PHENIX/phenix_bin:$PATH"\n')
    p = default_project()
    p["phases"].update(source="refine", program="phenix")
    p["programs"]["phenix"] = str(deep)
    notes, warnings = cli._job_program_notes(p, None)
    assert warnings == [] and notes[1] == f"  {deep}"
    text = jobscript.render("sge", str(tmp_path / "valdo.yaml"), notes=notes, log_dir=str(tmp_path))
    assert f"#   {deep}\n" in text                               # never wrapped inside the path
    # a variable the batch job may not have; a folder that is not an installation
    p["_unexpanded"] = {"programs.phenix": "$LAB_SOFT/phenix-2.2.1-6174"}
    notes, warnings = cli._job_program_notes(p, None)
    assert len(warnings) == 1 and "$LAB_SOFT" in warnings[0] and "must set them" in warnings[0]
    p["_unexpanded"] = {"programs.phenix": "$HOME/phenix-2.2.1-6174"}
    assert cli._job_program_notes(p, None)[1] == []
    p.pop("_unexpanded")
    p["programs"]["phenix"] = str(tmp_path / "nowhere")
    notes, warnings = cli._job_program_notes(p, None)
    assert "does not exist" in warnings[0] and "the refine stage would stop" in warnings[0]


def test_report_refinement_lines():
    pytest.importorskip("pandas")
    from valdo.easy import report
    ref = {"program": "dimple", "version": "CCP4 9.0.017, dimple 2.7", "location": "/opt/ccp4-9",
           "source": "valdo.yaml", "setting": "programs.ccp4", "twin_law": "-h,-k,l",
           "twin": {"n": 20, "median": 0.24, "min": 0.0, "max": 0.52, "n_untwinned": 2, "n_above_half": 1,
                    "n_with_note": 2}}
    lines = report._refinement_lines(ref)
    assert lines[0] == "Program: dimple (CCP4) CCP4 9.0.017, dimple 2.7 at /opt/ccp4-9 (programs.ccp4 in valdo.yaml)"
    assert "twin law -h,-k,l: twin fraction median 0.240 (range 0.000-0.520) over 20 datasets" in lines[1]
    assert "2 refined as untwinned (REFMAC kept no twin domain)" in lines[1] and "1 above 0.5" in lines[1]
    # phenix.refine refines every dataset with the law: a fraction of 0 is not REFMAC's decision
    phenix = report._refinement_lines(dict(ref, program="phenix", version="1.21.2-5419", setting="programs.phenix"))
    assert "; 2 with a refined twin fraction of 0;" in phenix[1] and "REFMAC" not in phenix[1]
    assert report._refinement_lines(dict(ref, twin_law=None))[1].startswith("No twin refinement")
    assert report._refinement_lines(None) == []
    # refined by several versions (an earlier run's datasets are not refined again); fractions not found
    mixed = dict(ref, version="2.2.1-6174", versions={"1.21.2-5419": 30, "2.2.1-6174": 10},
                 twin=dict(ref["twin"], n_missing=3))
    lines = report._refinement_lines(mixed)
    assert lines[1].startswith("Refined variants by program version: 1.21.2-5419: 30, 2.2.1-6174: 10")
    assert "twin fraction not found for 3 datasets" in lines[2]
    assert len(report._refinement_lines(dict(ref, versions={"CCP4 9.0.017, dimple 2.7": 40}))) == 2


def test_report_counts_refined_variants_per_version(tmp_path):
    pd = pytest.importorskip("pandas")
    from valdo.easy import report
    (tmp_path / "04_refine").mkdir()
    pd.DataFrame({"stem": ["0001_0", "0002_0", "0003_0", "0004_0"], "status": ["ok", "ok", "ok", "failed"],
                  "program_version": ["1.21.2-5419", "2.2.1-6174", "2.2.1-6174", "2.2.1-6174"]}
                 ).to_csv(tmp_path / "04_refine" / "refine_summary.csv", index=False)
    state = {"stages": {"refine": {"status": "done", "outputs": {
        "twin_law": "-h,-k,l", "program": {"program": "phenix", "version": "2.2.1-6174", "source": "valdo.yaml",
                                           "root": "/opt/phenix-2.2.1-6174", "setting": "programs.phenix"}}}}}
    datasets = pd.DataFrame({"status": ["ok", "ok", "ok"], "twin_fraction": ["0.26", "", "0.3"],
                             "twin_note": ["", "twin fraction not found in the phenix.refine output", ""]})
    ref = report._refinement({"state": state, "workdir": str(tmp_path)}, datasets)
    assert ref["versions"] == {"1.21.2-5419": 1, "2.2.1-6174": 2}
    assert ref["twin"]["n"] == 2 and ref["twin"]["n_missing"] == 1


# ---------------------------------------------------------------------------
# valdo check and valdo init on the toy screen (P 3 2 1)
# ---------------------------------------------------------------------------

@needs_science
def test_check_findings_for_programs_and_twin_laws(toy_screen, tmp_path, monkeypatch):
    from valdo.easy import check
    from valdo.easy import project as P
    folder, _ = toy_screen
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    workdir = tmp_path / "run"
    root = fake_ccp4(tmp_path)
    old = fake_ccp4(tmp_path, version="8.0.019", name="ccp4-8")
    inputs = {}

    def findings(extra):
        path = tmp_path / "valdo.yaml"
        path.write_text(f"data: {folder}/data\nmodels:\n  apo: {folder}/apo.pdb\nworkdir: {workdir}\n"
                        "phases:\n  source: refine\n  program: dimple\n" + extra)
        proj, _ = P.load_project(str(path))
        if "inp" not in inputs:
            inputs["inp"] = check.inspect_inputs(proj, threads=4)
        inp = inputs["inp"]
        programs.clear_cache()
        env = check.environment(proj, need_refine=True)
        found = {f["code"]: f for f in check._check_environment(proj, env) + check._check_twin(proj, inp)}
        return found, env, inp

    found, env, inp = findings("")
    f = found["refine_program_missing"]
    assert f["level"] == "error" and "programs.ccp4" in f["fix"]
    f = found["twin_law"]
    assert f["level"] == "info" and "Twin refinement is off" in f["title"] and "-h,-k,l" in f["what"]
    assert inp.twin == {"space_group": "P 3 2 1", "possible": ["-h,-k,l"], "configured": None, "ok": None,
                        "law": None}
    found, env, inp = findings(f"  twin_law: -h,-k,l\nprograms:\n  ccp4: {root}\n  phenix: {tmp_path}\n")
    assert "refine_program_missing" not in found and found["refine_program"]["level"] == "info"
    assert f"at {root}" in found["refine_program"]["what"] and "CCP4 9.0.017" in found["refine_program"]["what"]
    assert env["refine_program"]["root"] == str(root) and json.dumps(env["refine_program"])
    assert found["programs_unused"]["level"] == "info"          # programs.phenix, but the program is dimple
    assert "Twin refinement is on" in found["twin_law"]["title"]
    assert inp.twin["ok"] is True and inp.twin["law"] == "-h,-k,l"
    found, *_ = findings(f"programs:\n  ccp4: {tmp_path / 'nowhere'}\n")
    assert found["program_not_found"]["level"] == "error"
    for law, code in (("k,h,-l", "twin_law_is_symmetry"), ("k,l,h", "twin_law_invalid")):
        found, *_ = findings(f"  twin_law: {law}\n")
        assert found[code]["level"] == "error" and "-h,-k,l" in found[code]["fix"], law
    # the refine stage ran with another CCP4: say so (results would mix program versions)
    programs.clear_cache()
    before = programs.public(programs.resolve("dimple", str(old)))
    workdir.mkdir()
    (workdir / "valdo_state.json").write_text(json.dumps(
        {"stages": {"refine": {"status": "done", "outputs": {"program": before}}}}))
    found, *_ = findings(f"programs:\n  ccp4: {root}\n")
    f = found["program_changed"]
    assert f["level"] == "warning" and "CCP4 8.0.019" in f["what"] and "CCP4 9.0.017" in f["what"]
    assert "valdo run --from refine" in f["fix"]
    report = check.run_checks(P.load_project(str(tmp_path / "valdo.yaml"))[0], inputs["inp"])
    assert report["inputs"]["twin"]["possible"] == ["-h,-k,l"]


def test_program_changed_counts_versions(tmp_path):
    pd = pytest.importorskip("pandas")
    from valdo.easy import check
    project = {"workdir": str(tmp_path)}
    (tmp_path / "04_refine").mkdir()

    def warn(recorded, versions, now):
        rows = [{"stem": f"{i:04d}_0", "status": "ok", "program_version": v}
                for i, v in enumerate(v for v, n in versions.items() for _ in range(n))]
        rows.append({"stem": "9999_0", "status": "failed", "program_version": "0.9"})
        pd.DataFrame(rows).to_csv(tmp_path / "04_refine" / "refine_summary.csv", index=False)
        (tmp_path / "valdo_state.json").write_text(json.dumps(
            {"stages": {"refine": {"status": "done", "outputs": {"program": recorded}}}}))
        return check._check_program_versions(project, now)

    old = {"program": "phenix", "family": "phenix", "version": "1.21.2-5419", "root": "/a/phenix-1.21.2-5419"}
    new = {"program": "phenix", "family": "phenix", "version": "2.2.1-6174", "root": "/b/phenix-2.2.1-6174"}
    f = warn(old, {"1.21.2-5419": 3}, new)[0]
    assert f["code"] == "program_changed" and "differs" in f["title"]
    assert f["what"] == "Existing refinements: 1.21.2-5419 (3 refined variants); set up now: PHENIX 2.2.1-6174."
    f = warn(new, {"1.21.2-5419": 3, "2.2.1-6174": 1}, new)[0]
    assert "different program versions" in f["title"] and "2.2.1-6174 (1 refined variant)" in f["what"]
    assert warn(new, {"2.2.1-6174": 4}, dict(new, root="/moved/phenix-2.2.1-6174")) == []   # same version elsewhere
    ccp4 = {"program": "dimple", "family": "ccp4", "version": "CCP4 9.0.017, dimple 2.7.0"}
    assert warn(ccp4, {"CCP4 9.0.017": 2, "CCP4 9.0.017, dimple 2.7.0": 3}, ccp4) == []     # one version, two spellings
    assert warn(old, {"1.21.2-5419": 3}, ccp4) == []                                       # another program


def _init(folder, out, extra, env):
    return run_valdo(["init", os.path.join(folder, "data"), "--apo", os.path.join(folder, "apo.pdb"),
                      "-o", str(out), "--json"] + extra, cwd=os.path.dirname(str(out)), env=env, timeout=300)


@needs_science
def test_init_writes_programs_and_twin_law(toy_screen, tmp_path):
    import yaml
    folder, _ = toy_screen
    root = fake_ccp4(tmp_path)
    env = valdo_env()
    env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "/usr/bin:/bin")   # init finds this CCP4
    out = tmp_path / "a.yaml"
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--twin-law=-K,-H,-L"], env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    decisions = {d["setting"]: d for d in objs[-1]["decisions"]}
    assert decisions["programs.ccp4"]["value"] == str(root)
    assert decisions["phases.twin_law"]["value"] == "-k,-h,-l" and "-h,-k,l" in decisions["phases.twin_law"]["why"]
    text = out.read_text()
    saved = yaml.safe_load(text)
    assert saved["programs"]["ccp4"] == str(root) and saved["phases"]["twin_law"] == "-k,-h,-l"
    assert "found on PATH by valdo init" in text and "possible for P 3 2 1 with this cell: -h,-k,l" in text
    # given with --ccp4 (the installation's script), and no twin law: init says which laws are possible
    out = tmp_path / "b.yaml"
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--ccp4",
                                     str(root / "bin" / "ccp4.setup-sh")], valdo_env())
    assert proc.returncode == 0, proc.stderr[-3000:]
    decisions = {d["setting"]: d for d in objs[-1]["decisions"]}
    assert "given with --ccp4" in decisions["programs.ccp4"]["why"]
    assert decisions["phases.twin_law"]["value"] == "null" and "-h,-k,l" in decisions["phases.twin_law"]["why"]
    assert yaml.safe_load(out.read_text())["programs"]["ccp4"] == str(root / "bin" / "ccp4.setup-sh")


@needs_science
def test_init_twin_law_forms_force_and_unused_flags(toy_screen, tmp_path):
    import yaml
    folder, _ = toy_screen
    root = fake_ccp4(tmp_path)
    phenix = fake_phenix(tmp_path)
    env = valdo_env()
    env["PATH"] = "/usr/bin:/bin" + os.pathsep + os.path.dirname(sys.executable)     # no dimple on PATH
    out = tmp_path / "a.yaml"
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--ccp4", str(root),
                                     "--phenix", str(phenix), "--twin-law", "-h,-k,l"], env)   # no '=' needed
    assert proc.returncode == 0, proc.stderr[-3000:]
    decisions = {d["setting"]: d for d in objs[-1]["decisions"]}
    assert decisions["phases.twin_law"]["value"] == "-h,-k,l"
    assert decisions["programs.ccp4"]["why"].startswith("given with --ccp4: CCP4 9.0.017")
    assert any(w.startswith("--phenix is written to programs.phenix but not used: --program is dimple")
               for w in objs[-1]["warnings"])
    # init --force without --ccp4 keeps the installation of the file it replaces; the twin law is not kept
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--force"], env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    saved = yaml.safe_load(out.read_text())
    assert saved["programs"]["ccp4"] == str(root) and saved["programs"]["phenix"] == str(phenix)
    decisions = {d["setting"]: d for d in objs[-1]["decisions"]}
    assert "kept from the valdo.yaml this replaces" in decisions["programs.ccp4"]["why"]
    assert saved["phases"]["twin_law"] is None
    assert any("had phases.twin_law: -h,-k,l" in w for w in objs[-1]["warnings"])
    # auto: written as auto, the decision says which law it stands for
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--force", "--twin-law", "auto"], env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    decisions = {d["setting"]: d for d in objs[-1]["decisions"]}
    assert decisions["phases.twin_law"]["value"] == "auto -> -h,-k,l"
    assert yaml.safe_load(out.read_text())["phases"]["twin_law"] == "auto"
    # usage errors are JSON results too
    proc, objs = run_valdo(["check", str(out), "--bogus", "--json"], cwd=str(tmp_path), env=env)
    assert proc.returncode == 1 and objs[-1]["ok"] is False and objs[-1]["error"]["code"] == "bad_options"
    assert "--bogus" in objs[-1]["error"]["message"] and objs[-1]["command"] == "check"
    proc, objs = run_valdo(["check", str(out), "--bogus"], cwd=str(tmp_path), env=env)
    assert proc.returncode == 1 and "unrecognized arguments: --bogus" in proc.stderr and objs == []


def test_twin_law_value_may_start_with_a_minus():
    from valdo.easy import cli
    assert cli.join_option_values(["init", "d", "--twin-law", "-h,-k,l", "--json"]) == [
        "init", "d", "--twin-law=-h,-k,l", "--json"]
    assert cli.join_option_values(["init", "--twin-law", "auto"]) == ["init", "--twin-law", "auto"]
    assert cli.join_option_values(["init", "--twin-law", "--json"]) == ["init", "--twin-law", "--json"]
    args = cli.build_parser().parse_args(cli.join_option_values(["init", "d", "--twin-law", "-K,-H,-L"]))
    assert args.twin_law == "-K,-H,-L"


@needs_science
def test_init_refuses_unusable_programs_and_twin_laws(toy_screen, tmp_path):
    folder, _ = toy_screen
    env = valdo_env()
    env["PATH"] = "/usr/bin:/bin" + os.pathsep + os.path.dirname(sys.executable)
    cases = [(["--phases", "refine", "--program", "dimple", "--twin-law=k,h,-l"], "twin_law_is_symmetry"),
             (["--phases", "refine", "--program", "dimple", "--twin-law=x,y,z"], "twin_law_invalid"),
             (["--twin-law=-h,-k,l"], "bad_options"),
             (["--phases", "refine", "--program", "dimple", "--ccp4", str(tmp_path / "nowhere")],
              "program_not_found")]
    for n, (extra, code) in enumerate(cases):
        out = tmp_path / f"c{n}.yaml"
        proc, objs = _init(folder, out, extra, env)
        assert proc.returncode != 0 and objs[-1]["error"]["code"] == code, (extra, proc.stderr[-2000:])
        assert not out.exists()
    out = tmp_path / "c-auto.yaml"                 # no reference lattice to take auto from: init says so
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple", "--twin-law", "AUTO",
                                     "--reference", "nothing-of-that-name"], env)
    assert proc.returncode == 0 and any(w.startswith("--twin-law auto could not be checked")
                                        for w in objs[-1]["warnings"]), proc.stderr[-2000:]
    # no dimple anywhere: the project is written, with a warning that says what to set
    out = tmp_path / "d.yaml"
    proc, objs = _init(folder, out, ["--phases", "refine", "--program", "dimple"], env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert any("set programs.ccp4 in valdo.yaml" in w for w in objs[-1]["warnings"])


# ---------------------------------------------------------------------------
# valdo run (until filter) with twin refinement and a stand-in dimple
# ---------------------------------------------------------------------------

TWIN_TAIL = r'''
# --- what dimple 2.7 adds: pointless kept the indexing; with --refmac-key twin, REFMAC's twin domains
import json
data.write_mtz(os.path.join(outdir, "reindexed.mtz"))
with open(os.path.join(outdir, "dimple.log"), "a") as fh:
    fh.write("[pointless]\nalt_reindex: " + json.dumps([{"op": "[h,k,l]", "cc": 0.9, "cell_deviat": 0.1}]) + "\n")
if "--refmac-key" in args and args[args.index("--refmac-key") + 1] == "twin":
    untwinned = stem.split("_")[0] == os.environ.get("FAKE_UNTWINNED_ID")
    block = ["REMARK   3  TWIN DETAILS"] + (["REMARK   3   NUMBER OF TWIN DOMAINS  : NULL"] if untwinned else [
        "REMARK   3   NUMBER OF TWIN DOMAINS  :    2",
        "REMARK   3      TWIN DOMAIN   :    1", "REMARK   3      TWIN OPERATOR :  H,  K,  L",
        "REMARK   3      TWIN FRACTION : 0.800",
        "REMARK   3      TWIN DOMAIN   :    2", "REMARK   3      TWIN OPERATOR : -K, -H, -L",
        "REMARK   3      TWIN FRACTION : 0.200"])
    with open(os.path.join(outdir, "final.pdb")) as fh:
        text = fh.read()
    with open(os.path.join(outdir, "final.pdb"), "w") as fh:
        fh.write("\n".join(block) + "\n" + text)
'''


@needs_torch
def test_run_with_twin_refinement_fills_datasets_csv(toy_screen, tmp_path):
    import test_e2e_refine as E
    from valdo.easy.ids import read_datasets_csv
    folder, truth = toy_screen
    bindir = tmp_path / "bin"
    script = _write(tmp_path / "fake_dimple.py", E.FAKE_DIMPLE + TWIN_TAIL)
    _write(bindir / "dimple", f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', executable=True)
    E._truth_coefficients(folder, truth, str(tmp_path / "truth"))
    fail_id = E._failing_id(truth)
    plain = [f"{i:04d}" for i, d in enumerate(truth["datasets"], start=1)
             if i > 1 and not (d["bound"] or d["flipped"] or d["bad"]) and f"{i:04d}" != fail_id]
    env = valdo_env()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env.update(FAKE_TRUTH_DIR=str(tmp_path / "truth"), FAKE_DIMPLE_FAIL_ID=fail_id, FAKE_UNTWINNED_ID=plain[0])
    project = tmp_path / "twin.yaml"
    proc, objs = run_valdo(["init", os.path.join(folder, "data"), "--models",
                            os.path.join(folder, "models", "{name}-pandda-input.pdb"), "--apo",
                            os.path.join(folder, "apo.pdb"), "--phases", "refine", "--program", "dimple",
                            "--twin-law", "auto", "--ncpu", "4", "-o", str(project), "--json"],
                           cwd=str(tmp_path), env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    proc, objs = run_valdo(["run", "-c", str(project), "--until", "filter", "--json"], cwd=str(tmp_path), env=env)
    assert proc.returncode == 0, proc.stderr[-5000:]
    wd = objs[-1]["workdir"]
    rows = {r["id"]: r for r in read_datasets_csv(os.path.join(wd, "datasets.csv"))}
    ok = [r for r in rows.values() if r["status"] == "ok"]
    assert len(ok) >= 15 and rows[fail_id]["status"] == "dropped" and rows[fail_id]["twin_law"] == ""
    untwinned = rows[plain[0]]
    assert (untwinned["twin_law"], untwinned["twin_fraction"]) == ("", "0")
    assert "refined as untwinned" in untwinned["twin_note"]
    others = [r for r in ok if r["id"] != plain[0]]
    assert {(r["twin_law"], r["twin_fraction"], r["twin_note"]) for r in others} == {("-h,-k,l", "0.2", "")}
    proc, objs = run_valdo(["status", "-c", str(project), "--json"], cwd=str(tmp_path), env=env)
    recorded = objs[-1]["refine_program"]
    assert recorded["program"] == "dimple" and recorded["source"] == "PATH" and recorded["exe"] == str(bindir / "dimple")
