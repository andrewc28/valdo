"""Canonical dataset IDs and the dataset table (<workdir>/datasets.csv).

Every input MTZ gets a zero-padded numeric ID (0001, 0002, ...; wider if there are more
than 9999 files) in natural-sort order of the dataset names. Upstream stages only ever see
these IDs, which avoids upstream's name-parsing traps (first-4-characters phase lookup,
'PTP1B-y0049' parsed as id 1, ...). datasets.csv keeps the mapping and every dataset's fate.
"""

import csv
import os
import re
import tempfile

# Column order of datasets.csv. The first eleven are the documented contract.
DATASET_COLUMNS = [
    "id", "original_name", "source_mtz", "model", "phases_source", "variant",
    "reindex_op", "status", "dropped_stage", "dropped_reason", "flags",
    "phase_file", "bound_model", "space_group", "cell", "d_min", "n_reflections",
]

STEM_RE = re.compile(r"^(\d+)(?:_(\d+))?$")


def natural_key(text):
    """Sort key that compares digit runs as numbers ('x2' < 'x10'); ties by the plain string."""
    parts = re.split(r"(\d+)", text)
    key = []
    for i, part in enumerate(parts):
        if i % 2:
            key.append((1, int(part), ""))
        else:
            key.append((0, 0, part.lower()))
    return (tuple(key), text)


def id_width(n):
    """Width of zero-padded IDs for n datasets (4, or more if n > 9999)."""
    return max(4, len(str(n)))


def assign_ids(names):
    """Return {name: id} with IDs 0001.. in natural-sort order of the names (names must be unique)."""
    if len(set(names)) != len(names):
        raise ValueError("dataset names must be unique")
    ordered = sorted(names, key=natural_key)
    width = id_width(len(ordered))
    return {name: str(i + 1).zfill(width) for i, name in enumerate(ordered)}


def parse_stem(stem):
    """'0042_1' -> ('0042', '1'); '0042' -> ('0042', None); anything else -> (None, None)."""
    m = STEM_RE.match(stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def stem_of(path):
    """File stem without directory and extension ('/x/0042_1.mtz' -> '0042_1')."""
    return os.path.splitext(os.path.basename(path))[0]


def split_flags(text):
    return [f for f in (text or "").split(";") if f]


def join_flags(flags):
    seen = []
    for f in flags:
        if f and f not in seen:
            seen.append(f)
    return ";".join(seen)


def write_datasets_csv(rows, path):
    """Write rows (list of dicts) with DATASET_COLUMNS, atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".datasets.", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in DATASET_COLUMNS})
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_datasets_csv(path):
    """Read datasets.csv as a list of dicts of strings (IDs keep their zero padding)."""
    with open(path, newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
