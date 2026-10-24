"""Find and describe the input files: MTZ headers, dataset names, columns, phases, models.

Reading only MTZ headers keeps `valdo init` and `valdo check` fast on a cluster login node:
the header sits at the end of the file, so each file costs one open and two small reads, and
the files are read by a thread pool because network file systems are latency bound.
Nothing here imports torch or reciprocalspaceship.
"""

import glob
import math
import os
import re
import statistics
import struct
from concurrent.futures import ThreadPoolExecutor

from valdo.easy.ids import natural_key

# Map-coefficient conventions: key -> MTZ labels (amplitude, phase) for 2mFo-DFc and mFo-DFc.
PHASE_CONVENTIONS = {
    "phenix": {"fwt": "2FOFCWT", "phwt": "PH2FOFCWT", "delfwt": "FOFCWT", "phdelwt": "PHFOFCWT"},
    "xchem": {"fwt": "2FOFCWT", "phwt": "PHI2FOFCWT", "delfwt": "FOFCWT", "phdelwt": "PHIFOFCWT"},
    "refmac": {"fwt": "FWT", "phwt": "PHWT", "delfwt": "DELFWT", "phdelwt": "PHDELWT"},
}
RFREE_LABELS = ("R-free-flags", "FreeR_flag", "FREE", "FreeRflag", "FREER", "RFREE")
AMPLITUDE_PRIORITY = ("F-obs", "FP", "F", "FOBS", "F-obs-filtered", "F_obs", "Fobs")
# Amplitude-type (F) columns that are not observations
NOT_OBSERVED_F = {"2FOFCWT", "FOFCWT", "FWT", "DELFWT", "FC", "FC_ALL", "FC_ALL_LS", "FCALC",
                  "FMODEL", "F-MODEL", "FC_NOSCALE", "FAN", "F-OBS-SCALED", "RECONS", "DIFF",
                  "WDF", "ESF_2", "ESF_4", "ESF_6", "ESF_8", "ESF_16", "2FOFCWT_FILL",
                  "FWT_FILL", "FCALC_ALL", "FB", "F-CALC"}

MTZ_EXTENSIONS = (".mtz",)
MODEL_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")


# ---------------------------------------------------------------------------
# Paths and file lists
# ---------------------------------------------------------------------------

def expand_path(value, base_dir):
    """Expand ~ and $VARS; make relative paths absolute against base_dir (no symlink resolution)."""
    if value is None:
        return None
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if not os.path.isabs(value):
        value = os.path.join(base_dir, value)
    return os.path.abspath(value)


def has_glob(text):
    return any(c in text for c in "*?[")


def list_data_files(spec, base_dir="."):
    """MTZ files named by `spec`: a folder (its *.mtz), a glob, a .txt list, or one file.

    Returns (sorted absolute paths, note). The note explains an empty result.
    """
    path = expand_path(spec, base_dir)
    if path.endswith(".txt") and os.path.isfile(path):
        files = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    files.append(expand_path(line, os.path.dirname(path)))
        return sorted(set(files)), ""
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(MTZ_EXTENSIONS)]
        files = [f for f in files if os.path.isfile(f)]
        note = ""
        if not files:
            nested = glob.glob(os.path.join(path, "*", "*.mtz"))
            if nested:
                note = (f"{path} has no .mtz files directly inside, but its subfolders do; "
                        f"give a glob such as '{os.path.join(path, '*', os.path.basename(nested[0]))}'")
            else:
                note = f"no .mtz files in {path}"
        return sorted(files), note
    if has_glob(path):
        files = [os.path.abspath(f) for f in glob.glob(path, recursive=True) if os.path.isfile(f)]
        return sorted(set(files)), ("" if files else f"the pattern {path} matches no files")
    if os.path.isfile(path):
        return [path], ""
    return [], f"{path} does not exist"


NAME_SEPARATORS = "_-."


def strip_common_suffix(names):
    """Drop the end that every name shares, cut at a separator, if the names stay unique:
    ['PTP1B-y0001_mrflagsref_idxs', 'PTP1B-y0002_mrflagsref_idxs'] -> ['PTP1B-y0001', 'PTP1B-y0002'].
    Names that only share trailing letters or digits ('x10', 'x20') are left alone."""
    if len(names) < 2:
        return list(names)
    common = os.path.commonprefix([n[::-1] for n in names])[::-1]
    cut = next((common[i:] for i, ch in enumerate(common) if ch in NAME_SEPARATORS), "")
    if not cut:
        return list(names)
    out = [n[: len(n) - len(cut)] for n in names]
    if all(out) and len(set(out)) == len(out):
        return out
    return list(names)


def dataset_names(paths):
    """Unique, readable dataset names for the given files.

    The file stem if stems are unique, without an ending that every stem shares (see
    strip_common_suffix); otherwise the shortest unique path suffix; a stem shared by every file
    (e.g. '*/dimple.mtz') is dropped in favour of the folder names.
    """
    parts = []
    for p in paths:
        comps = os.path.normpath(p).split(os.sep)
        comps[-1] = os.path.splitext(comps[-1])[0]
        parts.append([c for c in comps if c])
    stems = [c[-1] for c in parts]
    drop_stem = len(paths) > 1 and len(set(stems)) == 1
    if not drop_stem and len(set(stems)) == len(stems):
        return strip_common_suffix(stems)
    for depth in range(1, max(len(c) for c in parts) + 1):
        names = []
        for c in parts:
            use = c[:-1] if drop_stem else c
            names.append("/".join(use[-depth:]) if use else c[-1])
        if len(set(names)) == len(names):
            return names
    return [os.path.splitext(p)[0] for p in paths]


# ---------------------------------------------------------------------------
# MTZ header reader
# ---------------------------------------------------------------------------

def read_mtz_header(path):
    """Read the header of an MTZ file without reading the reflections.

    Returns a dict: path, ok, error, columns [{label, type, dataset}], cell (6 floats),
    sg_number, sg_name (gemmi H-M name when gemmi is available), symops, nrefl, nbatch,
    d_min, d_max, datasets {id: name}, size, mtime.
    """
    info = {"path": path, "ok": False, "error": None, "columns": [], "cell": None,
            "sg_number": None, "sg_name": None, "symops": [], "nrefl": None, "nbatch": 0,
            "d_min": None, "d_max": None, "datasets": {}, "size": None, "mtime": None}
    try:
        st = os.stat(path)
        info["size"], info["mtime"] = st.st_size, st.st_mtime
        with open(path, "rb") as handle:
            first = handle.read(20)
            if len(first) < 20 or first[:4] != b"MTZ ":
                raise ValueError("not an MTZ file (no 'MTZ ' signature)")
            int_little = (first[9] >> 4) != 1  # machine stamp: 1 = big endian integers
            fmt = "<i" if int_little else ">i"
            pos = struct.unpack(fmt, first[4:8])[0]
            if pos == -1:  # large files: 64-bit header position after the stamp
                pos = struct.unpack("<q" if int_little else ">q", first[12:20])[0]
            offset = (pos - 1) * 4
            if offset <= 0 or offset >= info["size"]:
                raise ValueError("corrupt MTZ header position")
            handle.seek(offset)
            raw = handle.read(min(info["size"] - offset, 2_000_000))
        _parse_header_records(raw, info)
        info["ok"] = info["nrefl"] is not None and info["cell"] is not None
        if not info["ok"]:
            info["error"] = "incomplete MTZ header"
    except Exception as exc:  # unreadable file: reported by check, never fatal here
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _parse_header_records(raw, info):
    text = raw.decode("ascii", errors="replace")
    min_1_d2 = max_1_d2 = None
    sg_name_raw = None
    for i in range(0, len(text) - 3, 80):
        rec = text[i:i + 80]
        key = rec[:4].upper()
        if rec.startswith("END") and not rec.startswith("ENDO"):
            break
        tokens = rec.split()
        if not tokens:
            continue
        try:
            if key == "NCOL":
                info["nrefl"] = int(tokens[2])
                info["nbatch"] = int(tokens[3]) if len(tokens) > 3 else 0
            elif key == "CELL":
                info["cell"] = tuple(float(x) for x in tokens[1:7])
            elif key == "SYMI":  # SYMINF nsym nsymp lattice number 'name' pointgroup
                info["sg_number"] = int(tokens[4])
                m = re.search(r"'([^']*)'", rec)
                sg_name_raw = m.group(1).strip() if m else (tokens[5] if len(tokens) > 5 else None)
            elif key == "SYMM":
                info["symops"].append(rec[4:].strip().replace(" ", ""))
            elif key == "RESO":
                min_1_d2, max_1_d2 = float(tokens[1]), float(tokens[2])
            elif key == "COLU":  # COLUMN label type min max dataset
                info["columns"].append({"label": tokens[1], "type": tokens[2],
                                        "dataset": int(tokens[5]) if len(tokens) > 5 else 0})
            elif key == "DATA":  # DATASET id name
                if len(tokens) >= 3:
                    info["datasets"][int(tokens[1])] = tokens[2]
        except (ValueError, IndexError):
            continue
    if max_1_d2 and max_1_d2 > 0:
        info["d_min"] = 1.0 / math.sqrt(max_1_d2)
    if min_1_d2 and min_1_d2 > 0:
        info["d_max"] = 1.0 / math.sqrt(min_1_d2)
    info["sg_name"] = canonical_spacegroup(info["symops"], sg_name_raw, info["sg_number"])


def canonical_spacegroup(symops, name, number):
    """gemmi's H-M name for the header's space group (from the ops, else name, else number)."""
    try:
        import gemmi
    except ImportError:
        return name
    try:
        if symops:
            sg = gemmi.find_spacegroup_by_ops(gemmi.GroupOps([gemmi.Op(s) for s in symops]))
            if sg is not None:
                return sg.hm
    except Exception:
        pass
    for finder, arg in ((getattr(gemmi, "find_spacegroup_by_name", None), name),
                        (getattr(gemmi, "find_spacegroup_by_number", None), number)):
        try:
            if finder and arg is not None:
                sg = finder(arg)
                if sg is not None:
                    return sg.hm
        except Exception:
            pass
    return name


def scan_headers(paths, threads=16):
    """Read many MTZ headers concurrently; result is in the order of `paths`."""
    if not paths:
        return []
    workers = max(1, min(threads, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read_mtz_header, paths))


def column_types(header):
    return {c["label"]: c["type"] for c in header["columns"]}


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def amplitude_candidates(header):
    """(F, SIGF) label pairs in one header: an F column with a Q column named SIG+label."""
    types = column_types(header)
    pairs = []
    for label, typ in types.items():
        if typ != "F" or label.upper() in NOT_OBSERVED_F:
            continue
        for sig in ("SIG" + label, "SIG" + label.upper(), "SIG_" + label):
            if types.get(sig) == "Q":
                pairs.append((label, sig))
                break
    return pairs


def detect_amplitude_columns(headers):
    """Choose the observed amplitude/sigma pair shared by most files.

    Returns dict(amplitude, sigma, n_files, alternatives, intensity_only).
    amplitude is None if no pair was found.
    """
    counts = {}
    intensity_only = 0
    for h in headers:
        if not h["ok"]:
            continue
        pairs = amplitude_candidates(h)
        if not pairs and any(t == "J" for t in column_types(h).values()):
            intensity_only += 1
        for pair in pairs:
            counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return {"amplitude": None, "sigma": None, "n_files": 0, "alternatives": [],
                "intensity_only": intensity_only}

    def rank(pair):
        prio = AMPLITUDE_PRIORITY.index(pair[0]) if pair[0] in AMPLITUDE_PRIORITY else len(AMPLITUDE_PRIORITY)
        return (-counts[pair], prio, pair[0])

    ordered = sorted(counts, key=rank)
    best = ordered[0]
    return {"amplitude": best[0], "sigma": best[1], "n_files": counts[best],
            "alternatives": [f"{a},{s}" for a, s in ordered[1:]], "intensity_only": intensity_only}


def detect_phase_columns(headers):
    """Find map coefficients (2mFo-DFc and mFo-DFc, amplitude + phase) in the headers.

    Returns dict(convention, fwt, phwt, delfwt, phdelwt, n_files) for the convention found in
    the most files (phase columns required; amplitudes recorded if present), or None.
    """
    best = None
    for name, cols in PHASE_CONVENTIONS.items():
        n = 0
        for h in headers:
            if not h["ok"]:
                continue
            types = column_types(h)
            if types.get(cols["phwt"]) == "P" and types.get(cols["phdelwt"]) == "P":
                n += 1
        if n and (best is None or n > best["n_files"]):
            best = dict(cols, convention=name, n_files=n)
    return best


def has_phase_columns(header, columns):
    types = column_types(header)
    return types.get(columns.get("phwt")) == "P" and types.get(columns.get("phdelwt")) == "P"


def detect_rfree_column(headers):
    """Most common R-free flag column label (type I), or None."""
    counts = {}
    for h in headers:
        if not h["ok"]:
            continue
        types = column_types(h)
        for label in RFREE_LABELS:
            if types.get(label) == "I":
                counts[label] = counts.get(label, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], -RFREE_LABELS.index(k)))


# ---------------------------------------------------------------------------
# Cell, space group, resolution
# ---------------------------------------------------------------------------

def summarize_headers(headers, cell_tolerance=0.03, angle_tolerance=3.0):
    """Majority space group, median cell, outliers and resolution range of readable headers."""
    good = [h for h in headers if h["ok"]]
    out = {"n_files": len(headers), "n_readable": len(good), "space_group": None,
           "space_group_counts": {}, "median_cell": None, "cell_outliers": [],
           "sg_mismatch": [], "d_min_range": None, "d_min_median": None, "unmerged": []}
    if not good:
        return out
    for h in good:
        out["space_group_counts"][h["sg_name"]] = out["space_group_counts"].get(h["sg_name"], 0) + 1
    out["space_group"] = max(out["space_group_counts"], key=out["space_group_counts"].get)
    same = [h for h in good if h["sg_name"] == out["space_group"]]
    out["sg_mismatch"] = [h["path"] for h in good if h["sg_name"] != out["space_group"]]
    out["unmerged"] = [h["path"] for h in good if h["nbatch"]]
    med = tuple(statistics.median(h["cell"][i] for h in same) for i in range(6))
    out["median_cell"] = med
    for h in same:
        dev_len = max(abs(h["cell"][i] - med[i]) / med[i] for i in range(3))
        dev_ang = max(abs(h["cell"][i] - med[i]) for i in range(3, 6))
        if dev_len > cell_tolerance or dev_ang > angle_tolerance:
            out["cell_outliers"].append(h["path"])
    dmins = [h["d_min"] for h in good if h["d_min"]]
    if dmins:
        out["d_min_range"] = (min(dmins), max(dmins))
        out["d_min_median"] = statistics.median(dmins)
    return out


def format_cell(cell):
    if not cell:
        return ""
    return " ".join(f"{x:.2f}" for x in cell)


def hkl_triplet(op):
    """Reindexing operator as rs applies it to Miller indices (h' = h . R), e.g. '-h,-k,l'."""
    rot, den = op.rot, op.DEN
    out = []
    for j in range(3):
        terms = ""
        for i, letter in enumerate("hkl"):
            c = rot[i][j] / den
            if c == 0:
                continue
            sign = "-" if c < 0 else ("+" if terms else "")
            mag = abs(c)
            coef = "" if mag == 1 else (f"{mag:g}" if mag != int(mag) else str(int(mag)))
            terms += f"{sign}{coef}{letter}"
        out.append(terms or "0")
    return ",".join(out)


def alternative_indexing(cell, space_group):
    """Alternative indexing operators exactly as upstream reindex finds them.

    upstream: gemmi.find_twin_laws(reference.cell, reference_asu.spacegroup, max_obliq=3,
    all_ops=False). Returns (list of gemmi.Op, list of hkl strings) without the identity.
    """
    import gemmi
    ops = gemmi.find_twin_laws(gemmi.UnitCell(*cell), gemmi.SpaceGroup(space_group), 3, False)
    return list(ops), [hkl_triplet(op) for op in ops]


# ---------------------------------------------------------------------------
# Matching models / phase files / bound models to dataset names
# ---------------------------------------------------------------------------

def _boundary_find(name, text):
    """True if `name` occurs in `text` with no letter or digit directly before or after it."""
    start = 0
    while True:
        i = text.find(name, start)
        if i < 0:
            return False
        before = text[i - 1] if i > 0 else ""
        after = text[i + len(name)] if i + len(name) < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        start = i + 1


def match_files_to_names(spec, names, base_dir=".", extensions=MODEL_EXTENSIONS):
    """Match files to dataset names.

    `spec` is a pattern containing '{name}' (formatted per dataset; may also contain glob
    characters), a folder, or a glob. Returns (matches {name: path}, problems [str]).
    Names are matched with a boundary rule against the file stem, then its folder name;
    the longest matching name wins; two files for one name is reported as a problem.
    """
    matches, problems = {}, []
    if spec is None:
        return matches, problems
    if "{name}" in str(spec):
        for name in names:
            path = expand_path(str(spec).replace("{name}", name), base_dir)
            if has_glob(path):
                found = sorted(glob.glob(path))
                if len(found) == 1:
                    matches[name] = os.path.abspath(found[0])
                elif len(found) > 1:
                    problems.append(f"{name}: {len(found)} files match {path}")
            elif os.path.isfile(path):
                matches[name] = path
        return matches, problems
    path = expand_path(spec, base_dir)
    if os.path.isdir(path):
        candidates = [os.path.join(path, f) for f in os.listdir(path)]
    elif has_glob(path):
        candidates = glob.glob(path, recursive=True)
    elif os.path.isfile(path):
        candidates = [path]
    else:
        problems.append(f"{path} does not exist")
        return matches, problems
    candidates = sorted(os.path.abspath(c) for c in candidates
                        if os.path.isfile(c) and c.lower().endswith(tuple(extensions)))
    by_length = sorted(names, key=len, reverse=True)
    claimed = {}
    for cand in candidates:
        stem = os.path.splitext(os.path.basename(cand))[0]
        parent = os.path.basename(os.path.dirname(cand))
        hit = next((n for n in by_length if _boundary_find(n, stem)), None)
        if hit is None:
            hit = next((n for n in by_length if _boundary_find(n, parent)), None)
        if hit is not None:
            claimed.setdefault(hit, []).append(cand)
    for name, files in claimed.items():
        if len(files) == 1:
            matches[name] = files[0]
        else:
            problems.append(f"{name}: {len(files)} files match ({', '.join(os.path.basename(f) for f in files[:3])}"
                            f"{', ...' if len(files) > 3 else ''}); use a pattern with {{name}}")
    return matches, problems


def sort_names(names):
    return sorted(names, key=natural_key)
