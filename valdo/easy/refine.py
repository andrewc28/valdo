"""Per-dataset refinement for valdo-easy: phenix.refine or CCP4 dimple.

Upstream ``valdo.refine`` (valdo/commandline/refine.py) runs phenix.refine on
one dataset after another and leaves R-factor parsing, file naming and
failures to the user. :func:`refine_datasets` does the same refinement for
the ``refine`` stage of ``valdo run``, but

* runs ``ncpu`` datasets at a time (one external process each), with a
  per-dataset timeout;
* is resumable: a dataset whose inputs, program and options are unchanged
  and whose outputs exist is not refined again (``runs/<stem>/result.json``);
* never stops for one bad dataset: a failure is recorded with its reason and
  the log to read;
* parses R-work/R-free (PHENIX: the "Final R-work = ..., R-free = ..." summary
  of the .log, as PTP1B_pipeline/parse_refine_logs.py does, then PDB REMARK 3;
  dimple: REMARK 3 of final.pdb written by refmac, then the dimple output);
* writes normalised outputs ``phases/<stem>.mtz`` and ``models/<stem>.pdb``
  (the exact-basename names upstream's add_phases_and_blobs looks up first),
  always in the indexing frame of the input MTZ, and ``refine_summary.csv``
  with the columns upstream's ``filter`` stage reads (file_idx, symop,
  Rf_final).

Programs
--------
phenix
    ``phenix.refine <pdb> <mtz> output.prefix=refine_<stem> <eff>
    miller_array.labels.name=<F>,<SIGF> [xray_data.r_free_flags.generate=True]
    xray_data.r_free_flags.ignore_pdb_hexdigest=True [xray_data.twin_law=<law>] <extra args>
    --overwrite``, as valdo.refine runs it plus what PHENIX 1.20 and later need (checked with
    PHENIX 1.21.2-5419 and 2.2.1-6174, see :func:`_build_command`), with the packaged
    ``refine_drug.eff`` by default. The copy that is used (``phenix_refine.eff`` in the output
    folder) is repaired where real PHENIX rejects upstream's file (:func:`prepare_eff`).
    Outputs: ``runs/<stem>/refine_<stem>_001.{mtz,pdb,log}`` (``.cif`` when the .eff asks for it;
    ``refine_<stem>_data.mtz`` holds generated R-free flags). Map coefficients: 2FOFCWT/PH2FOFCWT
    and FOFCWT/PHFOFCWT (also FC/PHFC with the packaged .eff, 2FOFCWT_no_fill with PHENIX's
    defaults).
dimple
    ``dimple input.mtz <pdb> runs/<stem>/dimple --fcolumn F --sigfcolumn SIGF``
    plus ``--free-r-flags input.mtz --freecolumn FreeR_flag=N`` when the MTZ
    has R-free flags. ``runs/<stem>/input.mtz`` is a copy of the input MTZ
    with only the amplitude, sigma and R-free columns, relabelled F, SIGF and
    FreeR_flag: CCP4 programs reject some labels (pointless cannot read, and
    cad cannot take, the M/ISYM column that reciprocalspaceship's
    hkl_to_asu adds). dimple asks pointless whether the data should be
    re-indexed to match the model and, if so, writes final.mtz in the model's
    setting; the outputs are then transformed back into the input MTZ's
    setting (phases and model) and the operator is recorded; if that needs an
    origin shift (rare, e.g. P 21 21 21 with a = b) the dataset fails with
    that reason. Map coefficients: FWT/PHWT and DELFWT/PHDELWT.

Neither program is bundled. :func:`detect_programs` reports what is on PATH; with
``program_info`` (from :func:`valdo.easy.programs.resolve`, i.e. ``programs.phenix`` /
``programs.ccp4`` in valdo.yaml) the program and its environment come from that installation
instead.

Twin refinement (``twin_law``)
------------------------------
phenix
    ``xray_data.twin_law=<law>`` is added to the command line (a bare ``twin_law=`` is ambiguous in
    PHENIX 1.21 and 2.2); phenix.refine refines that law's twin fraction (target twin_lsq_f). The
    law and fraction are read from the refined model (mmCIF ``_pdbx_reflns_twin``, PDB REMARK 3
    "TWINNING INFORMATION") or the log; the formats are those of real PHENIX 1.21.2 and 2.2.1 runs
    (:func:`valdo.easy.twin.parse_phenix_twin`). PHENIX does not check the law: valdo does
    (``valdo check``).
dimple
    ``--refmac-key twin`` is added: dimple passes the keyword to its jelly-body and final
    restrained refmacat runs (not to the rigid-body refmac5 run or molecular replacement), and
    REFMAC finds the twin laws itself, keeps those with a twin fraction of at least 0.07 and
    refines their fractions. The law(s) REFMAC kept (REMARK 3 "TWIN DETAILS" of final.pdb,
    ``_pdbx_reflns_twin`` of final.mmcif) are compared with ``twin_law`` and reported.

With twinned data the amplitudes of the refined MTZ no longer identify its indexing (REFMAC writes
detwinned amplitudes, and at a twin fraction near 0.5 both indexings fit equally well), so in twin
mode the frame is taken from the program instead: phenix.refine never re-indexes; dimple
re-indexes only when its log shows that pointless ran with an ``alt_reindex`` result, and then its
``reindexed.mtz`` is compared exactly with the input amplitudes (fallback: the ``alt_reindex``
record of dimple.log). Without ``twin_law`` the frame is found as before.
"""

import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

import numpy as np
import pandas as pd

SUMMARY_COLUMNS = [
    "stem", "file_idx", "symop", "Rw_final", "Rf_final", "status", "program",
    "phases_mtz", "model_pdb", "log",
]
EXTRA_COLUMNS = [
    "Rw_start", "Rf_start", "seconds", "reason", "reindex_op", "f_col", "sigf_col",
    "free_col", "skipped", "program_version", "twin_law", "twin_fraction", "twin_note",
]

PROGRAM_EXECUTABLES = {"phenix": "phenix.refine", "dimple": "dimple"}
MAP_COEFFICIENTS = {
    "phenix": (("2FOFCWT", "PH2FOFCWT"), ("FOFCWT", "PHFOFCWT")),
    "dimple": (("FWT", "PHWT"), ("DELFWT", "PHDELWT")),
}

RESULT_FILE = "result.json"
# Added to the reason when a refined MTZ fits the input under two indexings (without twin refinement)
TWIN_HINT = ("both indexings fit, as for twinned data: if the crystals are twinned, set phases.twin_law "
             "('valdo check' lists the possible laws)")
# Twin mode: the refined amplitudes must still resemble the input under the operator taken from the
# program (REFMAC's detwinned F: CC 0.89-0.999 against the twinned input for twin fractions 0.5-0.05,
# S2 benchmark data, CCP4 9); this only catches gross mix-ups.
TWIN_FRAME_MIN_CC = 0.5
# Indexing-frame detection of refined MTZs (find_frame_op). dimple/refmac write the observed
# amplitudes after anisotropic scaling, so CC(input F, refined F) in the right frame is not 1:
# 0.977-0.9998 on 39 PTP1B datasets (CCP4 9, dimple 2.7), against 0.47 under the wrong indexing
# operator. A fixed 0.98 cut-off rejected a valid refinement; the decision is now an absolute floor
# plus a margin over every other candidate operator.
FRAME_MIN_CC = 0.9
FRAME_MIN_MARGIN = 0.1
RUN_MARKER = ".valdo_refine_run"
FINGERPRINT_VERSION = 1
# Version of the phenix.refine command line (part of PHENIX results' fingerprints). 2: data labels with
# miller_array.labels.name, ignore_pdb_hexdigest, xray_data.twin_law, '--overwrite' last, repaired .eff.
PHENIX_COMMAND_VERSION = 2

# Amplitude columns that are map coefficients or model values, never data.
_NOT_DATA = {
    "FC", "FC_ALL", "FC_ALL_LS", "FWT", "DELFWT", "2FOFCWT", "FOFCWT", "FMODEL",
    "F-model", "FCALC", "FCAL", "2FOFCWT_fill", "2FOFCWT_no_fill", "FAN", "recons",
    "diff", "WDF",
}
_PREFERRED_DATA = ("F-obs", "FP", "F", "FOBS", "F-obs-filtered", "F_obs")
_FREE_NAMES = ("R-free-flags", "FreeR_flag", "FREE", "RFREE", "FREER", "FreeRflag", "R-free", "Rfree")


class RefineError(RuntimeError):
    """A problem that affects every dataset (missing program, bad .eff, bad jobs)."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def phase_columns(program):
    """(phase_2FOFC_col_in, phase_FOFC_col_in) written by ``program``.

    These are the add_phases_and_blobs keys to use with the normalised
    ``phases/<stem>.mtz`` files: PHENIX (PH2FOFCWT, PHFOFCWT); dimple/refmac
    (PHWT, PHDELWT).
    """
    try:
        (_, p2), (_, p1) = MAP_COEFFICIENTS[program]
    except KeyError:
        raise ValueError(f"unknown refinement program {program!r}; use 'phenix' or 'dimple'") from None
    return p2, p1


def detect_programs(probe_versions=True, timeout_s=60):
    """Report which refinement programs are on PATH.

    Returns ``{"phenix": path or None, "dimple": path or None,
    "versions": {"phenix": str or None, "dimple": str or None},
    "refmac5": path or None, "notes": [str, ...]}``. Never raises.
    """
    phenix = shutil.which(PROGRAM_EXECUTABLES["phenix"])
    dimple = shutil.which(PROGRAM_EXECUTABLES["dimple"])
    refmac = shutil.which("refmac5")
    versions = {"phenix": None, "dimple": None}
    notes = []
    if dimple:
        if not refmac:
            notes.append("dimple is on PATH but refmac5 is not: set up CCP4 first "
                         "(source <ccp4>/bin/ccp4.setup-sh)")
        if not os.environ.get("CCP4"):
            notes.append("the CCP4 environment variable is not set: source <ccp4>/bin/ccp4.setup-sh "
                         "before running dimple")
        if probe_versions:
            out = _quick_output([dimple, "--version"], timeout_s)
            m = re.search(r"dimple\s+([0-9][\w.\-]*)", out or "")
            versions["dimple"] = m.group(1) if m else None
    if phenix:
        m = re.search(r"phenix-(\d+\.\d+(?:\.\d+)?(?:[-.]\d+)?)", phenix)
        if m:
            versions["phenix"] = m.group(1)
        elif probe_versions:
            ver_exe = os.path.join(os.path.dirname(phenix), "phenix.version")
            if os.path.isfile(ver_exe):
                out = _quick_output([ver_exe], timeout_s) or ""
                m = re.search(r"(?:Version|Release tag)\s*:\s*(\S+)", out)
                versions["phenix"] = m.group(1) if m else None
    return {"phenix": phenix, "dimple": dimple, "versions": versions, "refmac5": refmac, "notes": notes}


def packaged_eff_path():
    """Path of upstream's refine_drug.eff inside the installed valdo package."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "commandline", "refine_drug.eff")


def prepare_eff(eff, out_dir):
    """Write the .eff file phenix.refine will use and return (path, notes).

    ``eff=None`` uses the packaged refine_drug.eff (upstream's file, unchanged on disk), with three
    repairs, each reported in the notes, so that real PHENIX (1.21.2-5419, 2.2.1-6174) reads it:

    * its ``xray_data.r_free_flags.file_name=...`` line points at a file on the authors' cluster and
      is dropped when that file does not exist here;
    * its ``refine {`` scope is not closed before ``main {``, so the file as a whole never closes:
      PHENIX did not read it at all ("not a recognized phil file", and the run silently used
      PHENIX's defaults). The missing ``}`` is added before ``main {`` (``refinement.main`` is not
      part of ``refinement.refine``);
    * ``output.quiet`` is not a phenix.refine parameter (PHENIX stops at it once the file is read)
      and is dropped.

    A user-supplied .eff is copied unchanged, but every ``file_name=`` it names must exist and its
    braces must balance (PHENIX ignores a file whose braces do not); otherwise RefineError, before
    anything runs.
    """
    notes = []
    if eff is None:
        src = packaged_eff_path()
        if not os.path.isfile(src):
            raise RefineError(
                f"the packaged PHENIX parameter file is missing ({src}). Reinstall valdo "
                "(pip install -e .) or give your own .eff file."
            )
        kept = []
        with open(src) as fh:
            for line in fh:
                path = _eff_file_name(line)
                if path is not None and not os.path.isfile(path):
                    notes.append(f"dropped from the packaged .eff (file not found here): {line.strip()}")
                    continue
                if re.match(r"^\s*quiet\s*=", line.split("#", 1)[0]):
                    notes.append(f"dropped from the packaged .eff (not a phenix.refine parameter): "
                                 f"output.{line.strip()}")
                    continue
                kept.append(line)
        text, closed = _close_refine_before_main("".join(kept))
        if closed:
            notes.append("repaired the packaged .eff: closed its 'refine {' scope before 'main {' "
                         "(PHENIX did not read the file otherwise)")
        if _brace_balance(text) != 0:
            raise RefineError(f"the packaged PHENIX parameter file {src} has unbalanced braces that valdo "
                              "cannot repair; give your own .eff file (advanced.refine.eff).")
        origin = src
    else:
        eff = os.path.abspath(os.path.expanduser(eff))
        if not os.path.isfile(eff):
            raise RefineError(f"PHENIX parameter file not found: {eff}")
        with open(eff) as fh:
            text = fh.read()
        for line in text.splitlines():
            path = _eff_file_name(line)
            if path is not None and not os.path.isfile(path):
                raise RefineError(
                    f"{eff} refers to a file that does not exist: {path}\n"
                    "Fix the path or delete that line from the .eff file."
                )
        balance = _brace_balance(text)
        if balance != 0:
            raise RefineError(
                f"{eff}: its braces do not balance ({'more' if balance > 0 else 'fewer'} '{{' than '}}'), "
                "so PHENIX would ignore the whole file and refine with its defaults. Fix the file.")
        origin = eff
    dest = os.path.join(out_dir, "phenix_refine.eff")
    header = f"# used by valdo-easy refine; source: {origin}\n"
    _atomic_write_text(dest, header + text)
    return dest, notes


def _phil_tokens(line):
    """'name {' and '}' tokens of one .eff line (comments and quoted text left out)."""
    code = re.sub(r'"[^"]*"', '""', line.split("#", 1)[0])
    return [m.group(1) or "}" for m in re.finditer(r"([A-Za-z_][\w.]*)\s*\{|\}", code)]


def _brace_balance(text):
    """Number of '{' minus number of '}' (comments and quoted text left out)."""
    tokens = [t for line in text.splitlines() for t in _phil_tokens(line)]
    return sum(1 for t in tokens if t != "}") - tokens.count("}")


def _close_refine_before_main(text):
    """Close a ``refine {`` scope that is still open where ``main {`` starts. Returns (text, changed)."""
    out, stack, changed = [], [], False
    for line in text.splitlines(keepends=True):
        tokens = _phil_tokens(line)
        if tokens[:1] == ["main"] and stack and stack[-1] == "refine":
            indent = re.match(r"^\s*", line).group(0)
            out.append(f"{indent}}}  # closes 'refine {{' (added by valdo-easy)\n")
            stack.pop()
            changed = True
        out.append(line)
        for token in tokens:
            if token == "}":
                if stack:
                    stack.pop()
            else:
                stack.append(token)
    return "".join(out), changed


def inspect_mtz(path, f_col=None, sigf_col=None, free_col=None):
    """Read an MTZ and pick the data columns a refinement program should use.

    Returns a dict: f_col, sigf_col (amplitude + sigma; None if not found),
    n_amplitude_pairs, free_col, free_value (flag value of the test set),
    free_convention ('phenix' 0/1 with 1 = free, or 'ccp4' 0..N with 0 = free),
    n_free, n_refl, spacegroup, cell, d_min, columns ({label: type}).
    """
    import gemmi

    mtz = gemmi.read_mtz_file(path)
    types = {c.label: c.type for c in mtz.columns}
    labels = [c.label for c in mtz.columns]
    pairs = []
    for label in labels:
        if types[label] != "F" or label in _NOT_DATA or label.startswith(("ESF_", "WESF_")):
            continue
        sig = "SIG" + label
        if types.get(sig) == "Q":
            pairs.append((label, sig))
    info = {
        "columns": types,
        "n_amplitude_pairs": len(pairs),
        "n_refl": int(mtz.nreflections),
        "spacegroup": mtz.spacegroup.hm if mtz.spacegroup else None,
        "cell": [round(x, 4) for x in mtz.cell.parameters],
        "d_min": float(mtz.resolution_high()),
    }
    if f_col:
        if types.get(f_col) != "F":
            raise RefineError(f"{os.path.basename(path)}: amplitude column {f_col!r} not found (type F)")
        sigf_col = sigf_col or "SIG" + f_col
        if types.get(sigf_col) != "Q":
            raise RefineError(f"{os.path.basename(path)}: sigma column {sigf_col!r} not found (type Q)")
    elif pairs:
        ranked = sorted(pairs, key=lambda p: (p[0] not in _PREFERRED_DATA,
                                              _PREFERRED_DATA.index(p[0]) if p[0] in _PREFERRED_DATA else 0))
        f_col, sigf_col = ranked[0]
    info["f_col"], info["sigf_col"] = f_col, sigf_col

    # R-free flags
    if free_col is None:
        cands = [c for c in labels if types[c] == "I" and (c in _FREE_NAMES or "free" in c.lower())]
        cands.sort(key=lambda c: c not in _FREE_NAMES)
        free_col = cands[0] if cands else None
        info["n_free_candidates"] = len(cands)
    elif types.get(free_col) != "I":
        raise RefineError(f"{os.path.basename(path)}: R-free column {free_col!r} not found (type I)")
    info.update(free_col=None, free_value=None, free_convention=None, n_free=0)
    if free_col:
        vals = np.array(mtz.column_with_label(free_col), copy=False)
        vals = vals[np.isfinite(vals)]
        uniq, counts = np.unique(vals, return_counts=True)
        if len(uniq) >= 2:
            if set(uniq.tolist()) <= {0.0, 1.0}:
                free_value = int(uniq[np.argmin(counts)]) if counts[0] != counts[1] else 1
                convention = "phenix" if free_value == 1 else "ccp4"
            else:
                free_value, convention = 0, "ccp4"
            n_free = int(np.sum(vals == free_value))
            frac = n_free / max(len(vals), 1)
            if 0.005 <= frac <= 0.5:
                info.update(free_col=free_col, free_value=free_value,
                            free_convention=convention, n_free=n_free)
            else:
                info["free_note"] = (f"ignored R-free column {free_col}: test-set fraction "
                                     f"{frac:.3f} is implausible")
        else:
            info["free_note"] = f"ignored R-free column {free_col}: it holds a single value"
    return info


def parse_phenix_log(text):
    """R factors from a phenix.refine log.

    Uses the closing summary block ("Start R-work = a, R-free = b" /
    "Final R-work = c, R-free = d"), last occurrence wins; falls back to the
    first and last macro-cycle lines that mention both r_work and r_free (the
    same rules as PTP1B_pipeline/parse_refine_logs.py). Missing values are NaN.
    """
    start_re = re.compile(r"^\s*Start R-work\s*=\s*([0-9.]+),\s*R-free\s*=\s*([0-9.]+)", re.M)
    final_re = re.compile(r"^\s*Final R-work\s*=\s*([0-9.]+),\s*R-free\s*=\s*([0-9.]+)", re.M)
    out = dict(Rw_start=math.nan, Rf_start=math.nan, Rw_final=math.nan, Rf_final=math.nan)
    starts, finals = start_re.findall(text), final_re.findall(text)
    if finals:
        out["Rw_final"], out["Rf_final"] = (float(x) for x in finals[-1])
        if starts:
            out["Rw_start"], out["Rf_start"] = (float(x) for x in starts[-1])
        return out
    lines = [ln for ln in text.splitlines() if "r_work" in ln.lower() and "r_free" in ln.lower()]

    def _pair(line):
        parts = line.replace(",", " ").replace("=", " ").split()
        low = [p.lower() for p in parts]
        return float(parts[low.index("r_work") + 1]), float(parts[low.index("r_free") + 1])

    if len(lines) >= 2:
        try:
            out["Rw_start"], out["Rf_start"] = _pair(lines[0])
            out["Rw_final"], out["Rf_final"] = _pair(lines[-1])
        except (ValueError, IndexError):
            pass
    return out


def parse_model_r_factors(text):
    """Final R-work/R-free from a model file header.

    PDB ``REMARK   3`` lines written by refmac and phenix.refine
    ("R VALUE (WORKING SET) : x", "FREE R VALUE : y"), or mmCIF
    ``_refine.ls_R_factor_R_work`` / ``_refine.ls_R_factor_R_free``.
    """
    out = dict(Rw_final=math.nan, Rf_final=math.nan)
    rw = re.findall(r"^REMARK   3\s+R VALUE\s+\(WORKING SET\)\s*:\s*([0-9.]+)", text, re.M)
    rf = re.findall(r"^REMARK   3\s+FREE R VALUE\s*:\s*([0-9.]+)", text, re.M)
    if not rw:
        rw = re.findall(r"^_refine\.ls_R_factor_R_work\s+([0-9.]+)", text, re.M)
    if not rf:
        rf = re.findall(r"^_refine\.ls_R_factor_R_free\s+([0-9.]+)", text, re.M)
    if rw:
        out["Rw_final"] = float(rw[0])
    if rf:
        out["Rf_final"] = float(rf[0])
    return out


def parse_dimple_output(text):
    """R factors from dimple's screen output or dimple.log (fallback only).

    dimple reports refmac results as "R/Rfree 0.2534/0.2735 -> 0.2013/0.2369"
    on screen and as ``overall_r``/``free_r`` values in dimple.log; the last
    match is the final restrained refinement.
    """
    out = dict(Rw_start=math.nan, Rf_start=math.nan, Rw_final=math.nan, Rf_final=math.nan)
    arrows = re.findall(r"R/Rfree\s+([0-9.]+)/([0-9.]+)\s*->\s*([0-9.]+)/([0-9.]+)", text)
    if arrows:
        a, b, c, d = (float(x) for x in arrows[-1])
        out.update(Rw_start=a, Rf_start=b, Rw_final=c, Rf_final=d)
        return out
    rw = re.findall(r"^\s*overall_r\s*[:=]\s*\[?\s*([0-9.]+)", text, re.M)
    rf = re.findall(r"^\s*free_r\s*[:=]\s*\[?\s*([0-9.]+)", text, re.M)
    if rw and rf:
        out["Rw_final"], out["Rf_final"] = float(rw[-1]), float(rf[-1])
    return out


def _choose_frame(ccs, min_cc=FRAME_MIN_CC, min_margin=FRAME_MIN_MARGIN):
    """The operator triplet with the highest CC, if it reaches ``min_cc`` and beats every other
    candidate by at least ``min_margin``; else None (no match, or ambiguous)."""
    ranked = sorted(((k, v) for k, v in ccs.items() if np.isfinite(v)), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < min_cc:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < min_margin:
        return None
    return ranked[0][0]


def _phenix_identity(ccs, min_cc=FRAME_MIN_CC, min_margin=FRAME_MIN_MARGIN):
    """phenix.refine never re-indexes, and its refined MTZ carries the input's own amplitudes, so the
    identity stands when :func:`_choose_frame` cannot decide because twinned amplitudes also fit a
    twin law (real PTP1B data, PHENIX 1.21.2 and 2.2.1: x,y,z 1.000, -x,-y,z 0.921). Returns the other
    operators that came within ``min_margin`` of the identity ({triplet: cc}; the ones that kept
    :func:`_choose_frame` from deciding), or None when the identity is not the best operator or is
    below ``min_cc`` (a gross mix-up, which still fails)."""
    finite = {k: v for k, v in ccs.items() if np.isfinite(v)}
    ident = finite.get("x,y,z")
    if ident is None or ident < min_cc or ident < max(finite.values()):
        return None
    return {k: v for k, v in finite.items() if k != "x,y,z" and ident - v < min_margin}


def find_frame_op(input_mtz, refined_mtz, f_input, f_refined=None, min_cc=FRAME_MIN_CC,
                  min_margin=FRAME_MIN_MARGIN):
    """Find the indexing operator that maps ``refined_mtz`` onto ``input_mtz``.

    Tries the identity and every twin law of the input lattice
    (``gemmi.find_twin_laws(cell, sg, 3.0, False)``, as upstream reindex.py
    does) and returns the operator under which the refined file's amplitudes
    agree with the input amplitudes (Pearson CC on common reflections).

    Returns (gemmi.Op or None, best_cc, {op triplet: cc}). The op is None when
    no candidate reaches ``min_cc``, when the best one does not beat every other
    candidate by ``min_margin`` (ambiguous), or when no amplitude column can be
    compared.
    """
    import gemmi
    import reciprocalspaceship as rs

    ref = rs.read_mtz(input_mtz)
    ref = ref[[f_input]].dropna().hkl_to_asu()[[f_input]]
    ref = ref[~ref.index.duplicated()]
    other_all = rs.read_mtz(refined_mtz)
    if f_refined is None:
        f_refined = _pick_data_column(other_all, prefer=f_input)
    if f_refined is None:
        return None, math.nan, {}
    other = other_all[[f_refined]].dropna()
    ops = [gemmi.Op("x,y,z")]
    try:
        ops += list(gemmi.find_twin_laws(ref.cell, ref.spacegroup, 3.0, False))
    except Exception:  # pragma: no cover - only affects unusual lattices
        pass
    ccs, by_triplet = {}, {}
    best = (None, -np.inf)
    for op in ops:
        cand = other if op.triplet() == "x,y,z" else other.apply_symop(op)
        cand = cand.hkl_to_asu()[[f_refined]]
        cand = cand[~cand.index.duplicated()]
        joined = ref.join(cand, how="inner", rsuffix="_refined")
        if len(joined) < 50:
            continue
        a = joined.iloc[:, 0].to_numpy(dtype=float)
        b = joined.iloc[:, 1].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        cc = float(np.corrcoef(a[ok], b[ok])[0, 1])
        ccs[op.triplet()] = cc
        by_triplet[op.triplet()] = op
        if cc > best[1]:
            best = (op, cc)
    chosen = _choose_frame(ccs, min_cc, min_margin)
    if chosen is None:
        return None, (best[1] if np.isfinite(best[1]) else math.nan), ccs
    return by_triplet[chosen], ccs[chosen], ccs


def normalising_variant(op, spacegroup):
    """An operator with the same effect on amplitudes as ``op`` that is safe for phases and models.

    Twin laws (``gemmi.find_twin_laws``) are pure rotations defined only up to
    the point group: ``op * g`` for any symmetry rotation ``g`` gives the same
    amplitudes. Phases and a model may only be transformed by an operator
    whose real-space action (x' = op^-1 x, see :func:`reindex_model`) maps
    every symmetry operator of the space group onto one of the space group
    (no origin shift); otherwise the transformed map and model are not
    consistent with the space group. Returns the first such operator among
    ``op * g`` (``op`` itself first), or None when every choice would need an
    origin shift (e.g. P 21 21 21 with a = b and the a<->b re-indexing).
    """
    import gemmi

    if isinstance(op, str):
        op = gemmi.Op(op)
    ops = list(spacegroup.operations())
    group = {g.wrap().triplet() for g in ops}
    for g in ops:
        rot = gemmi.Op(g.triplet())
        rot.tran = [0, 0, 0]
        cand = op * rot
        inv = cand.inverse()
        if all((inv * h * cand).wrap().triplet() in group for h in ops):
            return cand
    return None


def reindex_mtz(src, dest, op):
    """Write ``src`` with Miller indices transformed by ``op`` (phases follow)."""
    import reciprocalspaceship as rs

    ds = rs.read_mtz(src)
    out = ds.apply_symop(op).hkl_to_asu()
    if "M/ISYM" in out.columns and "M/ISYM" not in ds.columns:
        out = out.drop(columns=["M/ISYM"])
    tmp = dest + ".tmp.mtz"
    out.write_mtz(tmp)
    os.replace(tmp, dest)


def reindex_model(pdb_in, pdb_out, op, normalise=True):
    """Write a model in the setting obtained by applying ``op`` to the data.

    ``op`` acts on Miller indices (h' = h R, as rs ``apply_symop``); the
    matching transformation of fractional coordinates is x' = R^-1 x, i.e.
    ``op.inverse()`` applied to every atom (anisotropic U transformed too).
    The cell and space group are kept.

    With ``normalise`` (default) and a model with a space group, ``op`` is
    first replaced by :func:`normalising_variant` (``op`` itself whenever
    that is safe): re-indexed amplitudes are the same for every member of
    ``op``'s point-group coset, but only a member that maps the space group
    onto itself gives a model that fits them (e.g. gemmi's twin law -x,y,-z
    for I 41 needs y,x,-z). Raises ValueError when no member does (the
    change of setting needs an origin shift). Returns the operator used.
    """
    import gemmi

    if isinstance(op, str):
        op = gemmi.Op(op)
    st = gemmi.read_structure(pdb_in)
    sg = st.find_spacegroup()
    if normalise and sg is not None:
        safe = normalising_variant(op, sg)
        if safe is None:
            raise ValueError(f"re-indexing with {op.triplet()} in space group {sg.hm} needs an origin shift; "
                             "the model cannot be transformed to match the re-indexed data")
        op = safe
    inv = op.inverse()
    cell = st.cell
    rot = np.array(inv.rot, dtype=float) / gemmi.Op.DEN
    orth = np.array(cell.orth.mat.tolist())
    rc = orth @ rot @ np.linalg.inv(orth)
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    f = cell.fractionalize(atom.pos)
                    g = inv.apply_to_xyz([f.x, f.y, f.z])
                    atom.pos = cell.orthogonalize(gemmi.Fractional(*g))
                    if atom.aniso.nonzero():
                        a = atom.aniso
                        u = np.array([[a.u11, a.u12, a.u13], [a.u12, a.u22, a.u23], [a.u13, a.u23, a.u33]])
                        v = rc @ u @ rc.T
                        atom.aniso = gemmi.SMat33f(v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[0, 2], v[1, 2])
    _write_structure_pdb(st, pdb_out)
    return op


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def refine_datasets(jobs, program, out_dir, ncpu=1, eff=None, extra_args=(), timeout_min=60, log=print,
                    twin_law=None, program_info=None):
    """Refine every dataset in ``jobs`` with ``program`` and summarise.

    Parameters
    ----------
    jobs : list of dict
        One dict per dataset: ``stem`` (e.g. "0001_0", the file stem used
        downstream), ``id`` ("0001"), ``symop`` ("0", or "" without reindex),
        ``mtz`` and ``pdb`` (paths). Optional: ``f_col``/``sigf_col`` (data
        columns; auto-detected otherwise), ``free_col`` (R-free column),
        ``twin_law`` (this job's twin law, with ``twin_law`` set).
    program : {"phenix", "dimple"}
    out_dir : str
        Folder for ``runs/<stem>/``, ``phases/``, ``models/`` and
        ``refine_summary.csv`` (``<workdir>/04_refine``).
    ncpu : int
        Datasets refined at the same time (one process each, 1 thread each).
    eff : str, optional
        PHENIX parameter file (default: packaged refine_drug.eff).
    extra_args : sequence of str
        Extra command-line arguments appended for every dataset.
    timeout_min : float or None
        Per-dataset wall-clock limit; None or <= 0 disables it.
    log : callable
        Called with one progress string at a time (from the calling thread).
    twin_law : str, optional
        Twin refinement with this law (hkl notation, in the setting of the jobs' MTZs, e.g.
        "-h,-k,l"; a job's own ``twin_law`` key takes precedence for that job): phenix.refine gets
        ``xray_data.twin_law=<law>``, dimple ``--refmac-key twin``. None (default) = no twin
        refinement, exactly as before.
    program_info : dict, optional
        :func:`valdo.easy.programs.resolve` result: the executable and the environment to run
        it in (``programs.phenix`` / ``programs.ccp4`` in valdo.yaml). None = the program on PATH
        in valdo's own environment (as before).

    Returns
    -------
    pandas.DataFrame with SUMMARY_COLUMNS (+ EXTRA_COLUMNS), one row per job,
    also written to ``out_dir/refine_summary.csv``; the rows with status "ok"
    and a known R-free are also written to ``out_dir/refine_summary_ok.csv``
    (feed that one to upstream's filter stage: it cannot handle rows without
    an R-free).

    Raises RefineError for problems that affect every dataset (unknown
    program, program not on PATH, bad .eff, malformed jobs); a failing dataset
    only gets status "failed" with the reason in its row and log.
    """
    if program not in PROGRAM_EXECUTABLES:
        raise RefineError(f"unknown refinement program {program!r}; use 'phenix' or 'dimple'")
    version, where = None, None
    if program_info is not None:
        if program_info.get("program") != program:
            raise RefineError(f"program_info describes {program_info.get('program')!r}, not {program!r}")
        if program_info.get("error"):
            err = program_info["error"]
            raise RefineError(f"{err['what']}. Fix: {err['fix']}")
        exe = program_info["exe"]
        base_env = program_info.get("env")
        version = program_info.get("version")
        from valdo.easy.programs import describe
        where = describe(program_info)
    else:
        exe = shutil.which(PROGRAM_EXECUTABLES[program])
        base_env = None
        if exe is None:
            hint = ("set programs.ccp4 in valdo.yaml to your CCP4 folder, or source <ccp4>/bin/ccp4.setup-sh "
                    "(CCP4 9 provides dimple and refmac5)" if program == "dimple" else
                    "set programs.phenix in valdo.yaml to your PHENIX folder, or source <phenix>/phenix_env.sh "
                    "(PHENIX needs a licence; see phenix-online.org)")
            raise RefineError(f"{PROGRAM_EXECUTABLES[program]} is not on PATH. Fix: {hint}, "
                              "then run again.")
    extra_args = [str(a) for a in extra_args]
    if twin_law:
        from valdo.easy import twin
        if twin.is_auto(twin_law):
            raise RefineError("twin law 'auto' must be resolved first: 'valdo check' and 'valdo run' resolve it "
                              "from the reference dataset's lattice")
        try:
            twin_law = twin.normalise_text(twin_law)
        except twin.TwinLawError as exc:
            raise RefineError(f"twin law: {exc}") from None
        clash = twin.twin_in_extra_args(program, extra_args)
        if clash:
            raise RefineError(f"the twin law is given twice: phases.twin_law and '{clash}' in the extra "
                              "arguments; set it in one place: phases.twin_law")
    else:
        twin_law = None
    jobs = _check_jobs(jobs)
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    for sub in ("runs", "phases", "models"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    eff_path, eff_notes = (prepare_eff(eff, out_dir) if program == "phenix" else (None, []))
    for note in eff_notes:
        log(f"[refine] note: {note}")
    timeout_s = None if not timeout_min or timeout_min <= 0 else float(timeout_min) * 60.0
    env = dict(os.environ if base_env is None else base_env)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    ctx = _Context(program=program, exe=exe, out_dir=out_dir, eff_path=eff_path,
                   extra_args=extra_args, timeout_s=timeout_s, env=env, twin_law=twin_law,
                   version=version, where=where)

    results = {}
    total = len(jobs)
    n_workers = max(1, min(int(ncpu or 1), total)) if total else 1
    log(f"[refine] {program}: {total} dataset(s), {n_workers} at a time, "
        f"timeout {('none' if timeout_s is None else f'{timeout_min:g} min')}, output {out_dir}")
    log(f"[refine] program: {where or exe}")
    if twin_law:
        log(f"[refine] twin refinement, twin law {twin_law}"
            + (" (REFMAC finds the twin laws itself: --refmac-key twin; the law it keeps is compared with this one)"
               if program == "dimple" else ""))
    t_start = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="valdo-refine")
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        def _on_sigterm(signum, frame):
            ctx.cancelled.set()
            ctx.registry.kill_all()
            if callable(previous_handler):
                previous_handler(signum, frame)
            raise KeyboardInterrupt("terminated by SIGTERM")
        try:
            previous_handler = signal.signal(signal.SIGTERM, _on_sigterm)
        except (ValueError, OSError):  # pragma: no cover
            previous_handler = None
    try:
        futures = {pool.submit(_run_one, job, ctx): job for job in jobs}
        n_ok = n_fail = n_skip = 0
        last_report = time.time()
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            job = futures[fut]
            stem = job["stem"]
            try:
                rec = fut.result()
            except Exception as exc:  # internal error in the wrapper itself
                rec = _record(stem, ctx, job=job, status="failed",
                              reason=f"internal error in valdo-easy refine: {exc!r}")
            results[stem] = rec
            if rec["status"] == "ok":
                n_ok += 1
                n_skip += bool(rec.get("skipped"))
            else:
                n_fail += 1
            now = time.time()
            detail = (f"R-free {rec['Rf_final']:.3f}" if rec["status"] == "ok" and _finite(rec["Rf_final"])
                      else rec.get("reason") or "")
            if rec.get("skipped"):
                detail = "already done"
            log(f"[refine] {i}/{total} {stem}: {rec['status']} {detail}".rstrip())
            if now - last_report > 300 or i == total:
                log(f"[refine] progress: {i}/{total} finished ({n_ok} ok, {n_fail} failed, "
                    f"{n_skip} reused), {(now - t_start) / 60:.1f} min elapsed")
                last_report = now
                _write_summary(jobs, results, out_dir, ctx, partial=(i < total))
    except BaseException:
        ctx.cancelled.set()
        ctx.registry.kill_all()
        pool.shutdown(wait=False, cancel_futures=True)
        _write_summary(jobs, results, out_dir, ctx, partial=True)
        raise
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGTERM, previous_handler)
            except (ValueError, OSError):  # pragma: no cover
                pass
    pool.shutdown(wait=True)
    table = _write_summary(jobs, results, out_dir, ctx, partial=False)
    n_ok = int((table["status"] == "ok").sum())
    log(f"[refine] done: {n_ok}/{total} refined, {total - n_ok} failed, "
        f"{(time.time() - t_start) / 60:.1f} min; summary {os.path.join(out_dir, 'refine_summary.csv')}")
    return table


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _Registry:
    """Running subprocesses, so they can all be killed on interrupt."""

    def __init__(self):
        self._procs = set()
        self._lock = threading.Lock()

    def add(self, proc):
        with self._lock:
            self._procs.add(proc)

    def discard(self, proc):
        with self._lock:
            self._procs.discard(proc)

    def kill_all(self):
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            _kill_group(proc, grace_s=2.0)


class _Context:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.registry = _Registry()
        self.cancelled = threading.Event()


def _check_jobs(jobs):
    jobs = list(jobs)
    seen = set()
    out = []
    for n, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise RefineError(f"refine job #{n} is not a dict: {job!r}")
        missing = [k for k in ("stem", "mtz", "pdb") if not job.get(k)]
        if missing:
            raise RefineError(f"refine job #{n} lacks {', '.join(missing)}: {job!r}")
        stem = str(job["stem"])
        if stem in seen:
            raise RefineError(f"duplicate stem {stem!r} in refine jobs")
        if os.sep in stem or stem.startswith("."):
            raise RefineError(f"bad stem {stem!r} (must be a plain file stem such as 0001_0)")
        seen.add(stem)
        j = dict(job)
        j["stem"] = stem
        m = re.match(r"^(.*?)(?:_(\d+))?$", stem)
        j.setdefault("id", m.group(1))
        j.setdefault("symop", m.group(2) or "")
        j["id"] = "" if j["id"] is None else str(j["id"])
        j["symop"] = "" if j["symop"] is None else str(j["symop"])
        j["mtz"] = os.path.abspath(os.path.expanduser(str(job["mtz"])))
        j["pdb"] = os.path.abspath(os.path.expanduser(str(job["pdb"])))
        if j.get("twin_law"):
            from valdo.easy import twin
            try:
                j["twin_law"] = twin.normalise_text(j["twin_law"])
            except twin.TwinLawError as exc:
                raise RefineError(f"refine job {stem}: twin law: {exc}") from None
        out.append(j)
    return out


def _record(stem, ctx, job=None, **kw):
    job = job or {}
    rec = {
        "stem": stem,
        "file_idx": job.get("id", ""),
        "symop": job.get("symop", ""),
        "Rw_final": math.nan,
        "Rf_final": math.nan,
        "status": "failed",
        "program": ctx.program,
        "phases_mtz": "",
        "model_pdb": "",
        "log": os.path.join(ctx.out_dir, "runs", stem, "refine.log"),
        "Rw_start": math.nan,
        "Rf_start": math.nan,
        "seconds": math.nan,
        "reason": "",
        "reindex_op": "",
        "f_col": "",
        "sigf_col": "",
        "free_col": "",
        "skipped": False,
        "program_version": getattr(ctx, "version", None) or "",
        "twin_law": "",
        "twin_fraction": math.nan,
        "twin_note": "",
    }
    rec.update(kw)
    return rec


def _run_one(job, ctx):
    stem = job["stem"]
    run_dir = os.path.join(ctx.out_dir, "runs", stem)
    phases_dest = os.path.join(ctx.out_dir, "phases", f"{stem}.mtz")
    model_dest = os.path.join(ctx.out_dir, "models", f"{stem}.pdb")
    if ctx.cancelled.is_set():
        return _record(stem, ctx, job=job, status="not_run", reason="interrupted")
    t0 = time.time()
    for key, what in (("mtz", "input MTZ"), ("pdb", "model")):
        if not os.path.isfile(job[key]):
            return _finish_failed(job, ctx, run_dir, f"{what} not found: {job[key]}", t0)
    try:
        info = inspect_mtz(job["mtz"], f_col=job.get("f_col"), sigf_col=job.get("sigf_col"),
                           free_col=job.get("free_col"))
    except Exception as exc:
        return _finish_failed(job, ctx, run_dir, f"could not read the input MTZ: {exc}", t0)
    if not info["f_col"]:
        return _finish_failed(job, ctx, run_dir,
                              "the input MTZ has no amplitude column with a matching sigma column "
                              "(e.g. F-obs/SIGF-obs); give f_col/sigf_col", t0)
    fingerprint = _fingerprint(job, ctx, info, run_dir)

    previous = _read_json(os.path.join(run_dir, RESULT_FILE))
    if (previous and previous.get("status") == "ok"
            and _fingerprints_equal(previous.get("fingerprint"), fingerprint)
            and os.path.isfile(phases_dest) and os.path.isfile(model_dest)):
        rec = {k: previous.get(k, v) for k, v in _record(stem, ctx, job=job).items()}
        # the version that made this result (empty if it was not recorded), not today's
        rec.update(skipped=True, phases_mtz=phases_dest, model_pdb=model_dest,
                   program_version=previous.get("program_version") or "")
        return rec

    try:
        _reset_run_dir(run_dir, ctx.out_dir)
    except RefineError as exc:  # a folder valdo-easy did not make: leave it alone, write nothing into it
        return _record(stem, ctx, job=job, status="failed", reason=str(exc), seconds=round(time.time() - t0, 1))
    for p in (phases_dest, model_dest):
        if os.path.lexists(p):
            os.remove(p)
    run_mtz = job["mtz"]
    if ctx.program == "dimple":
        try:
            run_mtz = _write_dimple_input(job["mtz"], os.path.join(run_dir, "input.mtz"), info)
        except Exception as exc:
            return _finish_failed(job, ctx, run_dir, f"could not prepare the input MTZ for dimple: {exc}", t0,
                                  fingerprint=fingerprint)
    cmd = _build_command(ctx, job, info, run_dir, run_mtz)
    log_path = os.path.join(run_dir, "refine.log")
    with open(log_path, "w") as fh:
        fh.write(f"# valdo-easy refine ({ctx.program}) {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# program: {ctx.where or ctx.exe}\n")
        fh.write(f"# cwd: {run_dir}\n# command: {shlex.join(cmd)}\n\n")
        fh.flush()
        rc, timed_out, launch_error = _run_process(cmd, run_dir, fh, ctx)
    if ctx.cancelled.is_set():
        return _record(stem, ctx, job=job, status="not_run", reason="interrupted")
    base = dict(f_col=info["f_col"], sigf_col=info["sigf_col"], free_col=info["free_col"] or "")
    if launch_error:
        return _finish_failed(job, ctx, run_dir, f"could not start {ctx.exe}: {launch_error}", t0,
                              fingerprint=fingerprint, **base)
    if timed_out:
        return _finish_failed(job, ctx, run_dir,
                              f"timed out after {ctx.timeout_s / 60:g} min (raise the refine timeout "
                              "or check the log)", t0, fingerprint=fingerprint, **base)
    if rc != 0:
        return _finish_failed(job, ctx, run_dir,
                              f"{ctx.program} exited with code {rc}: {_error_excerpt(log_path)}", t0,
                              fingerprint=fingerprint, **base)
    try:
        r, phases_src, model_src = _collect_outputs(ctx, run_dir, stem, log_path)
    except _OutputError as exc:
        return _finish_failed(job, ctx, run_dir, str(exc), t0, fingerprint=fingerprint, **base)
    law = _twin_law_of(job, ctx)
    frame_notes = []
    try:
        op_text, frame_op = _normalise(ctx.program, job, info, phases_src, model_src, phases_dest, model_dest,
                                       run_dir=run_dir, twin=bool(law), notes=frame_notes)
    except _OutputError as exc:
        return _finish_failed(job, ctx, run_dir, str(exc), t0, fingerprint=fingerprint, **base)
    if law:
        base.update(_twin_outcome(ctx, run_dir, stem, model_src, frame_op, law))
    reason = "; ".join(frame_notes)
    if not _finite(r.get("Rf_final")):
        reason = (reason + "; " if reason else "") + "refinement finished but R-free could not be read from its output"
    if op_text:
        reason = (reason + "; " if reason else "") + (
            f"{ctx.program} re-indexed the data with {op_text} to match the model; phases and model "
            "were transformed back into the input setting")
    if ctx.program == "dimple" and re.search(r"^\s*\d+\s+phaser\s", _read_text(log_path), re.M):
        reason = (reason + "; " if reason else "") + (
            "dimple ran molecular replacement (phaser: the model did not fit after rigid-body refinement, "
            "R > 0.4); check this dataset's model and R-free")
    rec = _record(stem, ctx, job=job, status="ok", phases_mtz=phases_dest, model_pdb=model_dest,
                  seconds=round(time.time() - t0, 1), reason=reason, reindex_op=op_text or "", **base, **r)
    _write_json(os.path.join(run_dir, RESULT_FILE), dict(rec, fingerprint=fingerprint))
    return rec


def _finish_failed(job, ctx, run_dir, reason, t0, fingerprint=None, **kw):
    rec = _record(job["stem"], ctx, job=job, status="failed", reason=reason,
                  seconds=round(time.time() - t0, 1), **kw)
    try:
        os.makedirs(run_dir, exist_ok=True)
        _touch_marker(run_dir)
        _write_json(os.path.join(run_dir, RESULT_FILE), dict(rec, fingerprint=fingerprint))
        with open(os.path.join(run_dir, "reason.txt"), "w") as fh:
            fh.write(reason + "\n")
        if not os.path.isfile(rec["log"]):
            with open(rec["log"], "w") as fh:
                fh.write(f"# valdo-easy refine: {reason}\n")
    except OSError:
        pass
    return rec


class _OutputError(Exception):
    pass


def _write_dimple_input(src, dest, info):
    """Copy the data columns dimple needs into ``dest`` under standard CCP4 labels."""
    import reciprocalspaceship as rs

    ds = rs.read_mtz(src)
    labels = {info["f_col"]: "F", info["sigf_col"]: "SIGF"}
    if info["free_col"]:
        labels[info["free_col"]] = "FreeR_flag"
    out = ds[list(labels)].rename(columns=labels)
    out = out[out["F"].notna() & out["SIGF"].notna()]
    tmp = dest + ".tmp.mtz"
    out.write_mtz(tmp)
    os.replace(tmp, dest)
    return dest


def _build_command(ctx, job, info, run_dir, run_mtz=None):
    extra = list(ctx.extra_args)
    twin = _twin_law_of(job, ctx)
    if ctx.program == "phenix":
        # PHENIX 1.20 and later (checked with 1.21.2-5419 and 2.2.1-6174): the data columns are chosen
        # with miller_array.labels.name (xray_data.labels and xray_data.r_free_flags.label are accepted
        # but only fill a GUI scope, so PHENIX would stop at 'Multiple equally suitable arrays', or
        # prefer intensities over the amplitudes valdo uses); a bare twin_law= is ambiguous
        # (xray_data.twin_law / neutron_data.twin_law); a start model refined against another R-free set
        # (REMARK r_free_flags.md5.hexdigest, e.g. every PanDDA input model) stops the run unless
        # ignore_pdb_hexdigest is set, which is what dimple does without asking. '--overwrite' comes last:
        # PHENIX 1.x (Python 2.7 argparse) rejects parameters that follow an option.
        cmd = [ctx.exe, job["pdb"], job["mtz"], f"output.prefix=refine_{job['stem']}", ctx.eff_path]
        keys = [a.split("=", 1)[0].strip() for a in extra if "=" in a]

        def given(*parts):
            return any(part in key for key in keys for part in parts)

        if not given("labels"):
            cmd.append(f"miller_array.labels.name={info['f_col']},{info['sigf_col']}")
        if not given("r_free_flags.generate", "r_free_flags.label", "labels"):
            if info["free_col"] is None:
                cmd.append("xray_data.r_free_flags.generate=True")
            elif info.get("n_free_candidates", 1) > 1:
                cmd.append(f"miller_array.labels.name={info['free_col']}")
        if not given("ignore_pdb_hexdigest"):
            cmd.append("xray_data.r_free_flags.ignore_pdb_hexdigest=True")
        if twin:
            cmd.append(f"xray_data.twin_law={twin}")
        return cmd + extra + ["--overwrite"]
    run_mtz = run_mtz or job["mtz"]
    cmd = [ctx.exe, run_mtz, job["pdb"], os.path.join(run_dir, "dimple"), "--fcolumn", "F", "--sigfcolumn", "SIGF"]
    if info["free_col"] and not any(a.startswith(("--free-r-flags", "-R", "--freecolumn")) for a in extra):
        cmd += ["--free-r-flags", run_mtz, "--freecolumn", f"FreeR_flag={info['free_value']}"]
    if twin:   # REFMAC finds the twin laws itself (jelly-body and final restrained runs of dimple)
        cmd += ["--refmac-key", "twin"]
    return cmd + extra


def _run_process(cmd, cwd, fh, ctx):
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=ctx.env, start_new_session=True)
    except OSError as exc:
        return None, False, str(exc)
    ctx.registry.add(proc)
    try:
        try:
            return proc.wait(timeout=ctx.timeout_s), False, None
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return None, True, None
    finally:
        ctx.registry.discard(proc)


def _kill_group(proc, grace_s=10.0):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass


def _collect_outputs(ctx, run_dir, stem, log_path):
    """Locate the refined MTZ/model and parse R factors; raise _OutputError."""
    with open(log_path, errors="replace") as fh:
        screen = fh.read()
    if ctx.program == "phenix":
        prefix = os.path.join(run_dir, f"refine_{stem}")
        mtz = _first_existing([f"{prefix}_001.mtz"] + sorted(
            p for p in _glob(f"{prefix}_[0-9][0-9][0-9].mtz") if not p.endswith("_data.mtz"))[::-1])
        model = _first_existing([f"{prefix}_001.pdb", f"{prefix}_001.cif"]
                                + sorted(_glob(f"{prefix}_[0-9][0-9][0-9].pdb"))[::-1]
                                + sorted(_glob(f"{prefix}_[0-9][0-9][0-9].cif"))[::-1])
        log_file = _first_existing([f"{prefix}_001.log"] + sorted(_glob(f"{prefix}_[0-9][0-9][0-9].log"))[::-1])
        text = screen
        if log_file:
            with open(log_file, errors="replace") as fh:
                text = fh.read()
        r = parse_phenix_log(text)
        if not _finite(r["Rf_final"]) and model:
            r.update({k: v for k, v in parse_model_r_factors(_read_text(model)).items() if _finite(v)})
    else:
        ddir = os.path.join(run_dir, "dimple")
        mtz = _first_existing([os.path.join(ddir, "final.mtz")])
        model = _first_existing([os.path.join(ddir, "final.pdb"), os.path.join(ddir, "final.cif")])
        r = dict(Rw_start=math.nan, Rf_start=math.nan, Rw_final=math.nan, Rf_final=math.nan)
        if model:
            r.update(parse_model_r_factors(_read_text(model)))
        fallback = parse_dimple_output(screen + "\n" + _read_text(os.path.join(ddir, "dimple.log")))
        for k, v in fallback.items():
            if not _finite(r.get(k)) and _finite(v):
                r[k] = v
    if not mtz:
        raise _OutputError(f"{ctx.program} finished but wrote no refined MTZ in {run_dir} "
                           f"({_error_excerpt(log_path)})")
    if not model:
        raise _OutputError(f"{ctx.program} finished but wrote no refined model in {run_dir}")
    return r, mtz, model


def _normalise(program, job, info, phases_src, model_src, phases_dest, model_dest, run_dir=None, twin=False,
               notes=None):
    """Copy (or transform) refined outputs into phases/ and models/.

    Returns (operator triplet when the program re-indexed the data else "", the gemmi.Op used to
    bring the outputs back into the input's indexing). With ``twin`` the indexing is taken from the
    program (:func:`_twin_frame_op`) instead of the amplitudes.
    """
    import gemmi

    try:
        refined = gemmi.read_mtz_file(phases_src)
    except Exception as exc:
        raise _OutputError(f"could not read the refined MTZ {phases_src}: {exc}") from None
    types = {c.label: c.type for c in refined.columns}
    missing = [f"{a}/{b}" for a, b in MAP_COEFFICIENTS[program] if a not in types or b not in types]
    if missing:
        raise _OutputError(
            f"the refined MTZ {os.path.basename(phases_src)} lacks the map coefficients "
            f"{', '.join(missing)} (columns present: {', '.join(types)}); with a custom .eff, keep "
            "the 2mFo-DFc (2FOFCWT/PH2FOFCWT) and mFo-DFc (FOFCWT/PHFOFCWT) map_coefficients blocks")
    in_sg = info.get("spacegroup")
    if in_sg and refined.spacegroup and refined.spacegroup.hm != in_sg:
        raise _OutputError(
            f"the refined MTZ is in space group {refined.spacegroup.hm} but the input is {in_sg} "
            "(the program probably ran molecular replacement); the dataset cannot be used")
    if twin:
        op = _twin_frame_op(program, job, info, run_dir, phases_src)
        cc, ccs = None, {}
    else:
        op, cc, ccs = find_frame_op(job["mtz"], phases_src, info["f_col"])
        others = _phenix_identity(ccs) if op is None and program == "phenix" else None
        if others is not None:
            op = gemmi.Op("x,y,z")
            if others and notes is not None:
                also = ", ".join(f"{k} (CC {v:.3f})" for k, v in others.items())
                notes.append(f"the refined amplitudes also fit {also}; phenix.refine keeps the input indexing "
                             f"(x,y,z, CC {ccs['x,y,z']:.3f}), so that one was used; {TWIN_HINT}")
    if op is None:
        if ccs:
            both = sum(1 for v in ccs.values() if np.isfinite(v) and v >= FRAME_MIN_CC) > 1
            raise _OutputError(
                "could not relate the refined MTZ to the input data (best amplitude CC "
                f"{cc:.3f}; tried {', '.join(f'{k}: {v:.3f}' for k, v in ccs.items())}; need >= "
                f"{FRAME_MIN_CC} and {FRAME_MIN_MARGIN} more than any other operator)"
                + (f"; {TWIN_HINT}" if both else ""))
        op_text = ""  # nothing comparable: assume the program kept the indexing
        identity = True
    else:
        identity = op.triplet() == "x,y,z"
        if not identity:
            safe = normalising_variant(op, refined.spacegroup)
            if safe is None:
                raise _OutputError(
                    f"{program} re-indexed the data with {op.triplet()} to match the model; for space group "
                    f"{refined.spacegroup.hm} the phases and the model cannot be transformed back without an "
                    "origin shift, which valdo-easy does not do. Give a model in the same setting as the data.")
            op = safe
        op_text = "" if identity else op.triplet()
    if identity:
        if _all_in_asu(refined):
            _atomic_copy(phases_src, phases_dest)
        else:
            reindex_mtz(phases_src, phases_dest, gemmi.Op("x,y,z"))
        if model_src.endswith(".pdb"):
            _atomic_copy(model_src, model_dest)
        else:
            _write_structure_pdb(gemmi.read_structure(model_src), model_dest)
    else:
        reindex_mtz(phases_src, phases_dest, op)
        reindex_model(model_src, model_dest, op)
    return op_text, (op if op is not None else gemmi.Op("x,y,z"))


def _twin_frame_op(program, job, info, run_dir, phases_src):
    """Twin mode: the operator that maps the refined MTZ's indexing onto the input's.

    phenix.refine never re-indexes (identity). dimple (2.7, main.py) runs pointless when the data
    and model symmetry match, and refines pointless's reindexed.mtz when pointless succeeded with an
    ``alt_reindex`` list (recorded in the [pointless] section of dimple.log); otherwise it refines
    the input as given. reindexed.mtz holds exactly the input amplitudes in the new indexing, so the
    operator is the candidate under which they agree exactly (fallback: the inverse of the first,
    chosen, alt_reindex operator). A loose correlation check then catches gross mix-ups."""
    import gemmi
    op = gemmi.Op("x,y,z")
    if program == "dimple" and run_dir:
        ddir = os.path.join(run_dir, "dimple")
        log_text = _read_text(os.path.join(ddir, "dimple.log"))
        if not log_text:
            raise _OutputError("dimple wrote no dimple.log, so it is unknown whether it re-indexed the data; "
                               "the phases cannot be placed safely")
        if _dimple_used_reindexed(log_text):
            found = None
            reindexed = os.path.join(ddir, "reindexed.mtz")
            if os.path.isfile(reindexed):
                found = _exact_frame_op(os.path.join(run_dir, "input.mtz"), reindexed)
            if found is None:
                found = _dimple_log_reindex_op(os.path.join(ddir, "dimple.log"))
            if found is None:
                raise _OutputError("dimple re-indexed the data (pointless), but neither reindexed.mtz nor the "
                                   "alt_reindex record of dimple.log says how; the phases cannot be placed safely")
            op = found
    cc = _frame_cc(job["mtz"], phases_src, info["f_col"], op)
    if cc is not None and not cc >= TWIN_FRAME_MIN_CC:
        raise _OutputError(f"the refined MTZ does not match the input data (amplitude CC {cc:.3f} under "
                           f"{op.triplet()}; need >= {TWIN_FRAME_MIN_CC})")
    return op


def _dimple_used_reindexed(log_text):
    """True if dimple refined pointless's reindexed.mtz: dimple.log has a [pointless] section with a
    non-empty alt_reindex list and pointless succeeded (dimple 2.7 main.py uses reindexed.mtz exactly
    then; its workflow writes an ``exit_status:`` line into the section only for a non-zero status)."""
    section = re.search(r"^\[pointless\]\s*$(.*?)(?=^\[|\Z)", log_text, re.M | re.S)
    return bool(section and re.search(r"^alt_reindex:\s*\[\s*\{", section.group(1), re.M)
                and not re.search(r"^exit_status:", section.group(1), re.M))


def _candidate_ops(mtz_path):
    import gemmi
    ops = [gemmi.Op("x,y,z")]
    try:
        m = gemmi.read_mtz_file(mtz_path)
        ops += list(gemmi.find_twin_laws(m.cell, m.spacegroup, 3.0, False))
    except Exception:  # pragma: no cover - unusual lattices
        pass
    return ops


def _exact_frame_op(input_mtz, reindexed_mtz, column="F", tolerance=1e-3):
    """The operator under which ``reindexed_mtz`` holds exactly the amplitudes of ``input_mtz``."""
    import reciprocalspaceship as rs
    try:
        ref = rs.read_mtz(input_mtz)[[column]].dropna().hkl_to_asu()[[column]]
        other = rs.read_mtz(reindexed_mtz)[[column]].dropna()
    except Exception:
        return None
    ref = ref[~ref.index.duplicated()]
    for op in _candidate_ops(input_mtz):
        cand = other if op.triplet() == "x,y,z" else other.apply_symop(op)
        cand = cand.hkl_to_asu()[[column]]
        cand = cand[~cand.index.duplicated()]
        joined = ref.join(cand, how="inner", rsuffix="_re")
        if len(joined) < max(50, 0.9 * len(ref)):
            continue
        a = joined.iloc[:, 0].to_numpy(dtype=float)
        b = joined.iloc[:, 1].to_numpy(dtype=float)
        if np.mean(np.abs(a - b) <= tolerance * np.maximum(1.0, np.abs(a))) >= 0.99:
            return op
    return None


def _dimple_log_reindex_op(path):
    """The re-indexing dimple chose, from the alt_reindex record of dimple.log (first = chosen)."""
    import gemmi
    m = re.search(r"^alt_reindex:\s*(\[.*\])\s*$", _read_text(path), re.M)
    if not m:
        return None
    try:
        first = json.loads(m.group(1))[0]["op"]
    except (ValueError, IndexError, KeyError, TypeError):
        return None
    from valdo.easy import twin
    try:
        matrix = twin.parse_law(str(first).strip("[]"))
    except twin.TwinLawError:
        return None
    if matrix == twin.IDENTITY:
        return gemmi.Op("x,y,z")
    # pointless's operator maps the input onto reindexed.mtz; the frame operator is its inverse
    inv = twin.inverse(matrix)
    rows = [[inv[i][j] for j in range(3)] for i in range(3)]
    if any(x.denominator != 1 for row in rows for x in row):
        return None
    den = gemmi.Op.DEN
    op = gemmi.Op("x,y,z")
    op.rot = [[int(x * den) for x in row] for row in rows]
    return op


def _frame_cc(input_mtz, refined_mtz, f_input, op):
    """Amplitude CC of the refined MTZ (brought into the input indexing with ``op``) with the input."""
    import reciprocalspaceship as rs
    try:
        ref = rs.read_mtz(input_mtz)[[f_input]].dropna().hkl_to_asu()[[f_input]]
        other_all = rs.read_mtz(refined_mtz)
    except Exception:
        return None
    ref = ref[~ref.index.duplicated()]
    f_refined = _pick_data_column(other_all, prefer=f_input)
    if f_refined is None:
        return None
    other = other_all[[f_refined]].dropna()
    cand = other if op.triplet() == "x,y,z" else other.apply_symop(op)
    cand = cand.hkl_to_asu()[[f_refined]]
    cand = cand[~cand.index.duplicated()]
    joined = ref.join(cand, how="inner", rsuffix="_refined")
    if len(joined) < 50:
        return None
    a = joined.iloc[:, 0].to_numpy(dtype=float)
    b = joined.iloc[:, 1].to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 50 else None


def _twin_outcome(ctx, run_dir, stem, model_src, frame_op, law):
    """twin_law / twin_fraction / twin_note of one twin refinement (configured law ``law``), in the
    input's indexing."""
    from valdo.easy import twin
    out = {"twin_law": "", "twin_fraction": math.nan, "twin_note": ""}
    try:
        import gemmi
        sg = gemmi.read_mtz_file(os.path.join(ctx.out_dir, "phases", f"{stem}.mtz")).spacegroup
        rots = twin.rotations(sg) if sg is not None else None
    except Exception:
        rots = None
    if ctx.program == "dimple":
        ddir = os.path.join(run_dir, "dimple")
        domains = twin.parse_refmac_twin(_read_text(os.path.join(ddir, "final.pdb")),
                                         _read_text(os.path.join(ddir, "final.mmcif")))
        identity = frame_op is None or frame_op.triplet() == "x,y,z"
        law_used, fraction, note = twin.summarise_domains(domains, law, None if identity else frame_op, rots)
        out.update(twin_law=law_used, twin_fraction=math.nan if fraction is None else fraction, twin_note=note)
        return out
    # phenix.refine writes refine_<stem>_001.* (another number only if the prefix was used before)
    base = os.path.splitext(model_src)[0]
    cif_text = _read_text(base + ".cif")
    pdb_text = _read_text(model_src) if model_src.endswith(".pdb") else ""
    found = twin.parse_phenix_twin(cif_text, pdb_text, _read_text(base + ".log"))
    if found is None:
        out.update(twin_law=law, twin_note="twin fraction not found in the phenix.refine output")
        return out
    note = ""
    if rots is not None and not twin.same_law(twin.parse_law(found["law"]), twin.parse_law(law), rots):
        note = f"phenix.refine reports twin law {found['law']}, not the configured {law}"
    out.update(twin_law=law if not note else found["law"], twin_fraction=found["fraction"], twin_note=note)
    return out


def _all_in_asu(mtz):
    import reciprocalspaceship as rs

    hkl = np.array(mtz.make_miller_array())
    if len(hkl) == 0:
        return True
    return bool(np.all(rs.utils.in_asu(hkl, mtz.spacegroup)))


def _pick_data_column(ds, prefer=None):
    cols = list(ds.columns)
    dtypes = {c: str(ds[c].dtype) for c in cols}
    if prefer and prefer in cols:
        return prefer
    for c in cols:
        if dtypes[c] in ("SFAmplitude",) and c not in _NOT_DATA and ("SIG" + c) in cols:
            return c
    return None


def _fingerprint(job, ctx, info, run_dir):
    previous = _read_json(os.path.join(run_dir, RESULT_FILE)) or {}
    old = (previous.get("fingerprint") or {}).get("files", {})
    files = {}
    for key in ("mtz", "pdb"):
        files[key] = _file_digest(job[key], old.get(key))
    if ctx.eff_path:
        files["eff"] = _file_digest(ctx.eff_path, old.get("eff"))
    fp = {
        "version": FINGERPRINT_VERSION,
        "program": ctx.program,
        "extra_args": list(ctx.extra_args),
        "f_col": info["f_col"],
        "sigf_col": info["sigf_col"],
        "free_col": info["free_col"],
        "files": files,
    }
    if ctx.program == "phenix":          # the phenix.refine command line itself changed (see _build_command)
        fp["command"] = PHENIX_COMMAND_VERSION
    law = _twin_law_of(job, ctx)
    if law:                              # only then: results refined without a twin law stay valid
        fp["twin_law"] = law
    return fp


def _twin_law_of(job, ctx):
    """The twin law to refine for ``job``: its own (an indexing variant can need another law than the
    reference indexing, see twin.law_for_variant), else the one given to refine_datasets; None without
    twin refinement."""
    if not getattr(ctx, "twin_law", None):
        return None
    return job.get("twin_law") or ctx.twin_law


def _file_digest(path, previous=None):
    st = os.stat(path)
    if previous and previous.get("size") == st.st_size and previous.get("mtime_ns") == st.st_mtime_ns \
            and previous.get("path") == path:
        return previous
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()}


def _fingerprints_equal(a, b):
    def strip(fp):
        if not fp:
            return fp
        fp = dict(fp)
        fp["files"] = {k: v.get("sha256") for k, v in fp.get("files", {}).items()}
        return fp
    return strip(a) == strip(b)


def _reset_run_dir(run_dir, out_dir):
    """Empty runs/<stem>/ before a new attempt (only a folder this module made)."""
    runs = os.path.join(out_dir, "runs")
    if os.path.islink(run_dir):
        raise RefineError(f"{run_dir} is a symbolic link; refusing to use it as a run folder")
    if os.path.isdir(run_dir):
        inside = os.path.commonpath([os.path.abspath(run_dir), os.path.abspath(runs)]) == os.path.abspath(runs)
        if not inside or not os.path.isfile(os.path.join(run_dir, RUN_MARKER)):
            raise RefineError(
                f"{run_dir} exists but was not created by valdo-easy refine; move it away and run again")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    _touch_marker(run_dir)


def _touch_marker(run_dir):
    path = os.path.join(run_dir, RUN_MARKER)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write("created by valdo.easy.refine; this folder may be deleted and recreated\n")


def _write_summary(jobs, results, out_dir, ctx, partial):
    rows = []
    for job in jobs:
        rec = results.get(job["stem"])
        if rec is None:
            rec = _record(job["stem"], ctx, job=job, status="not_run",
                          reason="not run yet" if partial else "not run")
        rows.append(rec)
    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS + EXTRA_COLUMNS)
    _atomic_write_csv(table, os.path.join(out_dir, "refine_summary.csv"))
    usable = (table["status"] == "ok") & pd.to_numeric(table["Rf_final"], errors="coerce").notna()
    _atomic_write_csv(table[usable], os.path.join(out_dir, "refine_summary_ok.csv"))
    return table


def _error_excerpt(log_path, max_chars=300):
    text = _read_text(log_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    keys = ("Sorry", "Error", "ERROR", "error:", "Traceback", "Exception", "failed", "Failed",
            "No phaser solution", "Killed", "FATAL")
    hits = [ln for ln in lines if any(k in ln for k in keys)]
    pick = hits[-2:] if hits else lines[-2:]
    msg = " | ".join(pick) or "no output (see the log)"
    return msg[-max_chars:]


def _quick_output(cmd, timeout_s):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, timeout=timeout_s, text=True)
        return res.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _eff_file_name(line):
    code = line.split("#", 1)[0]
    m = re.search(r"file_name\s*=\s*\"?([^\"\s]+)\"?", code)
    if not m or m.group(1) in ("None", "none", "Auto"):
        return None
    return m.group(1)


def _write_structure_pdb(st, dest):
    tmp = dest + ".tmp.pdb"
    st.write_pdb(tmp)
    os.replace(tmp, dest)


def _atomic_copy(src, dest):
    tmp = dest + ".tmp" + os.path.splitext(dest)[1]
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _atomic_write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_json(path, data):
    _atomic_write_text(path, json.dumps(data, indent=1, default=_json_default) + "\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    return str(obj)


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _read_text(path):
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _glob(pattern):
    import glob as _glob_mod
    return _glob_mod.glob(pattern)


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
