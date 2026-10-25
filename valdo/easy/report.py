"""The valdo-easy report: what VALDO found, which datasets made it, how to look.

:func:`build_report` reads a finished (or partly finished) valdo-easy work
folder and writes ``<workdir>/report/``:

``report.html``
    One self-contained page (inline CSS, plots embedded as base64 PNG, no
    scripts, no network): run summary, dataset funnel, top hits, how to open
    them in Coot or PyMOL, QC plots, optional validation against known bound
    models, dropped datasets, stage timings, settings, caveats.
``report.md`` / ``report.json``
    The same content as text and as machine-readable JSON.
``blobs.csv``
    Every blob kept after tagging (not near the focal residue, not marked
    duplicate by upstream, which only catches a symmetry copy next in peak order
    and within 1 A), ranked by ``score`` (upstream's ROC/AUC evaluation ranks
    blobs by ``score``: PTP1B_pipeline/plot_auc.py), with dataset ID, original
    name and ``symmetry_copy_of`` (the rank of a stronger blob of the same
    dataset that this one is a symmetry copy of; such copies are left out of the
    top hits).
``datasets.csv``
    One row per dataset: original name, fate (status, stage and reason when
    dropped), flags, number of kept blobs, rank/score/peak z of its best blob.
``top_hits.csv`` and ``hits/``
    The top ``top_n`` blobs with the nearest residue (gemmi NeighborSearch on
    the dataset's model, symmetry mates included), the most positive and most
    negative difference-map value within the blob (and a sign: positive,
    negative, mixed, weak), whether the blob overlaps the ligand of a known
    bound model (``on_known_ligand``; empty for datasets without one, the
    overlap rule of the validation section), and per hit a boxed CCP4
    difference map (signed, unblurred, from the same coefficients and phases
    the blob search used), a 2mFo-DFc-style context map when the phase source
    has such amplitudes, and a copy of the model.
``view_top_hits.py`` / ``view_top_hits.pml``
    Coot and PyMOL scripts that open model + maps centred on each hit (paths
    relative to ``report/``; both scripts find that folder when started from
    the work folder too).

Nothing here imports torch; matplotlib is not needed either (the QC plots are
the PNG files the upstream stages wrote).
"""

import base64
import datetime
import glob
import html
import json
import math
import os
import re
import shutil

import numpy as np
import pandas as pd

from valdo.easy import maps

REPORT_SCHEMA = "valdo-easy-report/1"
RANK_KEYS = ("score", "peakz", "peak", "volume")
DEFAULT_RANK = "score"
HITS_MARKER = ".valdo_report_hits"
BOX_RADIUS_A = 12.0
DIFF_LEVEL_SIGMA = 3.0
DIFF_DOMINANCE = 1.5      # diff_sign: one side "dominates" when it is this many times stronger
CONTEXT_LEVEL_SIGMA = 1.0
# Two kept blobs of one dataset whose centroids are symmetry copies closer than this (or half the
# smaller blob radius, if larger) are one blob found twice; the weaker one is left out of the top hits.
COPY_MIN_A = 1.5

STAGE_ORDER = [
    "standardize", "reindex", "scale", "refine", "filter", "preprocess", "train",
    "reconstruct", "rescale", "add_phases_and_blobs", "tag_blobs", "report",
]
STAGE_DIRS = {
    "standardize": "01_standardize", "reindex": "02_reindex", "scale": "03_scale",
    "refine": "04_refine", "filter": "05_filter", "preprocess": "06_preprocess",
    "train": "07_train", "reconstruct": "08_reconstruct", "rescale": "09_rescale",
    "add_phases_and_blobs": "10_phases_blobs", "tag_blobs": "11_tag", "report": "report",
}
FUNNEL_LABELS = {
    "input": "Datasets found",
    "input_checks": "Passed the input checks (readable, same crystal form)",
    "standardize": "Standardized (copied under their IDs)",
    "reindex": "Indexed consistently with the reference",
    "scale": "Scaled to the reference",
    "refine": "Refined (phases available)",
    "filter": "Passed the quality filter",
    "preprocess": "In the VAE's input table",
    "train": "VAE trained (on a training/validation split of these)",
    "reconstruct": "Reconstructed by the VAE",
    "rescale": "Difference amplitudes computed",
    "add_phases_and_blobs": "Phased and searched for blobs",
    "tag_blobs": "Blobs tagged",
}

# HETATM residues that are not ligands by default (waters, ions, buffers,
# cryoprotectants, precipitants, the fragment solvent). PTP1B bound-state models
# contain HOH, TRS (Tris buffer) and LIG.
NOT_LIGANDS = frozenset("""
HOH WAT DOD H2O SOL
NA K LI RB CS MG CA SR BA MN FE FE2 CO NI CU CU1 ZN CD HG PB AL
CL BR IOD F
SO4 SUL PO4 PI NO3 NH4 SCN AZI CO3 BCT
ACT ACY FMT TRS EPE MES BTB B3P CIT FLC TAR TLA MLI SIN MLA IMD
GOL EDO PEG PGE PG4 PG5 PG6 1PE 2PE P6G 12P 15P XPE MPD MRD PDO BU3 IPA IPH EOH MOH
DMS DMF DMSO ACN ACE BME DTT TCE GSH
""".split())

_STEM_RE = re.compile(r"^(\d+)(?:_(\d+))?$")
_LIST_COLUMNS = ("fractional", "all_possible_frac", "all_possible_cart")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_report(workdir, top_n=20, rank_by=None, known_bound_dir=None, ligand_resnames=None,
                 write_maps=True):
    """Write ``<workdir>/report/`` and return a JSON-serialisable summary.

    Parameters
    ----------
    workdir : str
        The valdo-easy work folder (with datasets.csv, valdo_state.json,
        configs/, 10_phases_blobs/, 11_tag/ ...).
    top_n : int
        Number of top-ranked blobs to describe in detail (maps, residues).
    rank_by : {"score", "peakz", "peak", "volume"}, optional
        Ranking of the kept blobs; default "score" (what upstream's ROC/AUC
        evaluation ranks by).
    known_bound_dir : str, optional
        Folder of known bound-state models (any file names containing the
        dataset name or ID). Default: the ``bound_model`` column of
        datasets.csv, else ``<workdir>/inputs/bound_models``.
    ligand_resnames : iterable of str, optional
        Residue names that count as ligand in the bound models. Default: any
        HETATM residue that is not water, an ion or a common buffer/solvent
        (:data:`NOT_LIGANDS`).
    write_maps : bool
        Write per-hit CCP4 maps, model copies and viewer scripts.

    Returns
    -------
    dict with keys schema, ok, workdir, report_dir, files, counts, rank_by,
    funnel, refinement (or None), top_hits, validation (or None), warnings.
    """
    workdir = os.path.abspath(os.path.expanduser(workdir))
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"work folder not found: {workdir}")
    rank_by = rank_by or DEFAULT_RANK
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {', '.join(RANK_KEYS)} (got {rank_by!r})")
    top_n = max(0, int(top_n or 0))
    report_dir = os.path.join(workdir, "report")
    os.makedirs(report_dir, exist_ok=True)
    warnings = []

    ctx = _load_context(workdir, warnings)
    datasets = ctx["datasets"]
    kept = ctx["blobs_kept"]
    ranked = _rank_blobs(kept, rank_by) if kept is not None else None
    if ranked is not None:
        ranked = _attach_names(ranked, datasets)
        ranked = _mark_symmetry_copies(ranked, ctx)

    ds_table = _dataset_table(datasets, ranked)
    funnel = _funnel(datasets, ctx["state"], ranked)
    hits_dir = os.path.join(report_dir, "hits")
    if write_maps:
        _reset_hits_dir(hits_dir, workdir)
    bound = _bound_models(ctx, known_bound_dir, warnings)
    validation = None
    if bound:  # before the top hits: it marks which blobs overlap a known ligand (ctx["on_ligand"])
        validation = _validate_known_bound(ranked, ctx, bound, ligand_resnames, warnings)
    top_hits = []
    copies_skipped = 0
    if ranked is not None and len(ranked) and top_n:
        distinct = ranked[ranked["symmetry_copy_of"].isna()].head(top_n)
        if len(distinct):
            last = int(distinct["rank"].max())
            copies_skipped = int(((ranked["rank"] <= last) & ranked["symmetry_copy_of"].notna()).sum())
        top_hits = _describe_top_hits(distinct, ctx, hits_dir if write_maps else None, warnings)

    qc = _qc_plots(ctx)
    stages = _stage_rows(ctx["state"])
    settings = _settings(ctx)
    counts = _counts(datasets, ctx, ranked)
    refinement = _refinement(ctx, datasets)

    files = {
        "html": os.path.join(report_dir, "report.html"),
        "markdown": os.path.join(report_dir, "report.md"),
        "json": os.path.join(report_dir, "report.json"),
        "datasets_csv": os.path.join(report_dir, "datasets.csv"),
        "blobs_csv": os.path.join(report_dir, "blobs.csv"),
        "top_hits_csv": os.path.join(report_dir, "top_hits.csv"),
    }
    if write_maps:
        files.update(coot_script=os.path.join(report_dir, "view_top_hits.py"),
                     pymol_script=os.path.join(report_dir, "view_top_hits.pml"),
                     hits_dir=hits_dir)

    _write_csv(ds_table, files["datasets_csv"])
    _write_csv(_blobs_csv_table(ranked), files["blobs_csv"])
    _write_csv(_top_hits_table(top_hits), files["top_hits_csv"])
    if write_maps:
        _write_coot_script(top_hits, files["coot_script"])
        _write_pymol_script(top_hits, files["pymol_script"])

    result = {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "valdo_version": _valdo_version(),
        "workdir": workdir,
        "report_dir": report_dir,
        "files": files,
        "counts": counts,
        "rank_by": rank_by,
        "funnel": funnel,
        "refinement": refinement,
        "top_hits": [_public_hit(h) for h in top_hits],
        "top_hits_symmetry_copies_left_out": copies_skipped,
        "validation": validation,
        "stages": stages,
        "settings": settings,
        "qc_plots": [{k: v for k, v in q.items() if k != "data"} for q in qc],
        "dropped": _dropped_summary(datasets),
        "warnings": warnings,
    }
    _write_text(files["json"], json.dumps(_jsonable(result), indent=1) + "\n")
    _write_text(files["markdown"], _render_markdown(result))
    _write_text(files["html"], _render_html(result, qc, ranked))
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_context(workdir, warnings):
    cfg = {name: _read_yaml(os.path.join(workdir, "configs", f"{name}.yaml"))
           for name in ("add_phases_and_blobs", "tag_blobs", "reindex", "scale", "train", "filter")}
    apb, tag = cfg["add_phases_and_blobs"], cfg["tag_blobs"]
    ctx = {
        "workdir": workdir,
        "configs": cfg,
        "state": _read_json(os.path.join(workdir, "valdo_state.json")) or {},
        "phased_dir": apb.get("output_folder") or os.path.join(workdir, "10_phases_blobs", "mtz"),
        "diff_col": apb.get("blob_diff_col") or "WDF",
        "phase_col": apb.get("phase_col") or "PH2FOFCWT",
        "phasing_path": apb.get("phasing_path"),
        "phase_2fofc_in": apb.get("phase_2FOFC_col_in"),
        "model_folder": apb.get("model_folder") or tag.get("model_folder")
        or os.path.join(workdir, "inputs", "models"),
        "focal_tag": tag.get("focal_tag_name"),
        "tag_dir": tag.get("output_folder") or os.path.join(workdir, "11_tag"),
    }
    ds_path = os.path.join(workdir, "datasets.csv")
    if os.path.isfile(ds_path):
        datasets = pd.read_csv(ds_path, dtype=str, keep_default_na=False)
    else:
        datasets = None
        warnings.append("datasets.csv not found: dataset names and fates are unknown")
    ctx["datasets"] = datasets

    tagged = os.path.join(ctx["tag_dir"], "blob_stats_tagged.pkl")
    filtered = os.path.join(ctx["tag_dir"], "filtered_blob_stats_tagged.pkl")
    ctx["blobs_all"] = ctx["blobs_kept"] = None
    ctx["blob_source"] = None
    if os.path.isfile(filtered):
        ctx["blobs_kept"] = pd.read_pickle(filtered)
        ctx["blob_source"] = filtered
        if os.path.isfile(tagged):
            ctx["blobs_all"] = pd.read_pickle(tagged)
    elif os.path.isfile(tagged):
        allb = pd.read_pickle(tagged)
        ctx["blobs_all"] = allb
        keep = pd.Series(True, index=allb.index)
        if ctx["focal_tag"] and ctx["focal_tag"] in allb:
            keep &= allb[ctx["focal_tag"]] == 0
        if "duplicate" in allb:
            keep &= allb["duplicate"] == 0
        ctx["blobs_kept"] = allb[keep]
        ctx["blob_source"] = tagged
    else:
        prefix = apb.get("prefix") or ""
        raw = os.path.join(apb.get("blob_output_folder") or os.path.join(workdir, "10_phases_blobs", "blobs"),
                           prefix + "blob_stats.pkl")
        if os.path.isfile(raw):
            ctx["blobs_all"] = ctx["blobs_kept"] = pd.read_pickle(raw)
            ctx["blob_source"] = raw
            warnings.append("tag_blobs has not run: blobs near the focal residue and symmetry "
                            "duplicates are still in the list")
        else:
            warnings.append("no blob table found (add_phases_and_blobs/tag_blobs have not finished): "
                            "the report has no hits yet")
    return ctx


def _read_yaml(path):
    try:
        import yaml
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ImportError, ValueError):
        return {}
    except Exception:  # yaml errors
        return {}


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _split_stem(sample):
    m = _STEM_RE.match(str(sample))
    if m:
        return m.group(1), m.group(2) or ""
    return str(sample), ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _rank_blobs(kept, rank_by):
    df = kept.copy()
    df["blob_index"] = df.index
    secondary = "peakz" if rank_by != "peakz" else "score"
    keys = [rank_by] + ([secondary] if secondary in df else []) + ["sample", "blob_index"]
    asc = [False] + ([False] if secondary in df else []) + [True, True]
    df = df.sort_values(keys, ascending=asc, kind="mergesort").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    ids = df["sample"].map(lambda s: _split_stem(s)[0])
    symops = df["sample"].map(lambda s: _split_stem(s)[1])
    df.insert(1, "id", ids)
    df.insert(2, "symop", symops)
    df["rank_in_dataset"] = df.groupby("id").cumcount() + 1
    return df


def _attach_names(ranked, datasets):
    names, flags, status = {}, {}, {}
    if datasets is not None and "id" in datasets:
        names = dict(zip(datasets["id"], datasets.get("original_name", datasets["id"])))
        if "flags" in datasets:
            flags = dict(zip(datasets["id"], datasets["flags"]))
        if "status" in datasets:
            status = dict(zip(datasets["id"], datasets["status"]))
    ranked = ranked.copy()
    ranked.insert(3, "original_name", ranked["id"].map(lambda i: names.get(i, "")))
    ranked["flags"] = ranked["id"].map(lambda i: flags.get(i, ""))
    ranked["dataset_status"] = ranked["id"].map(lambda i: status.get(i, ""))
    return ranked


def _sample_symmetry(ctx, sample):
    """(orthogonalisation matrix, fractionalisation matrix, [(R, t)]) of a sample's phased MTZ, or None."""
    from valdo.easy import scan
    h = scan.read_mtz_header(os.path.join(ctx["phased_dir"], f"{sample}.mtz"))
    if not h.get("ok") or not h.get("cell"):
        return None
    try:
        import gemmi
        cell = gemmi.UnitCell(*h["cell"])
        if h.get("symops"):
            ops = [gemmi.Op(s) for s in h["symops"]]
        elif h.get("sg_name"):
            ops = list(gemmi.SpaceGroup(h["sg_name"]).operations())
        else:
            return None
    except Exception:
        return None
    orth = np.array(cell.orth.mat.tolist(), dtype=float)
    frac = np.array(cell.frac.mat.tolist(), dtype=float)
    return orth, frac, [(np.array(op.rot, dtype=float) / op.DEN, np.array(op.tran, dtype=float) / op.DEN)
                        for op in ops]


_SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


def _symmetry_distance(fa, fb, sym):
    """Shortest distance (A) between fractional point fa and any symmetry/lattice image of fb."""
    orth, _, ops = sym
    best = np.inf
    for rot, tran in ops:
        d = rot @ fb + tran - fa
        d = d - np.round(d) + _SHIFTS
        best = min(best, float(np.sqrt(((d @ orth.T) ** 2).sum(axis=1)).min()))
    return best


def _mark_symmetry_copies(ranked, ctx):
    """Column symmetry_copy_of: the rank of a stronger kept blob of the same dataset (variant) whose
    centroid is a symmetry copy of this blob's (space group and lattice translations), closer than
    max(COPY_MIN_A, half the smaller blob radius); empty otherwise. Upstream's mark_duplicates only
    compares blobs next to each other in peak order, with a 1 A cut-off, so some copies of one blob
    stay in the kept list. Nothing is removed here: the top-hit table skips the copies, blobs.csv
    marks them, and the ranks (and the validation) stay those of upstream's list."""
    ranked = ranked.copy()
    copy_of = pd.array([pd.NA] * len(ranked), dtype="Int64")
    if len(ranked) and {"cenx", "ceny", "cenz"} <= set(ranked.columns):
        position = {idx: i for i, idx in enumerate(ranked.index)}
        for sample, grp in ranked.groupby("sample", sort=False):
            if len(grp) < 2:
                continue
            sym = _sample_symmetry(ctx, sample)
            if sym is None:
                continue
            frac = sym[1]
            distinct = []
            for idx, row in grp.sort_values("rank").iterrows():
                f = frac @ np.array([row["cenx"], row["ceny"], row["cenz"]], dtype=float)
                radius = float(row["radius"]) if "radius" in row.index and pd.notna(row["radius"]) else 0.0
                for rank_i, f_i, r_i in distinct:
                    if _symmetry_distance(f_i, f, sym) < max(COPY_MIN_A, 0.5 * min(radius, r_i)):
                        copy_of[position[idx]] = rank_i
                        break
                else:
                    distinct.append((int(row["rank"]), f, radius))
    ranked["symmetry_copy_of"] = copy_of
    return ranked


def _blobs_csv_table(ranked):
    cols = ["rank", "id", "original_name", "sample", "symop", "rank_in_dataset", "score", "peakz",
            "peak", "volume", "radius", "cenx", "ceny", "cenz"]
    if ranked is None:
        return pd.DataFrame(columns=cols + ["flags"])
    extra = [c for c in ranked.columns if c not in cols and c not in _LIST_COLUMNS
             and c not in ("blob_index", "flags", "dataset_status")]
    out = ranked[[c for c in cols if c in ranked] + extra + ["flags"]].copy()
    if "fractional" in ranked:
        frac = ranked["fractional"].map(lambda f: list(np.asarray(f, dtype=float)) if f is not None
                                        and np.size(f) == 3 else [np.nan] * 3)
        out["frac_x"] = [f[0] for f in frac]
        out["frac_y"] = [f[1] for f in frac]
        out["frac_z"] = [f[2] for f in frac]
    return out


def _dataset_table(datasets, ranked):
    base_cols = ["id", "original_name", "status", "dropped_stage", "dropped_reason", "flags", "variant"]
    if datasets is not None:   # twin refinement: the kept variant's twin law and fraction
        base_cols += [c for c in ("twin_law", "twin_fraction", "twin_note")
                      if c in datasets and (datasets[c].astype(str) != "").any()]
    if datasets is not None:
        table = datasets[[c for c in base_cols if c in datasets]].copy()
    else:
        ids = sorted(set(ranked["id"])) if ranked is not None else []
        table = pd.DataFrame({"id": ids, "original_name": ids, "status": "ok"})
    for c in base_cols:
        if c not in table:
            table[c] = ""
    n_blobs, best = {}, {}
    if ranked is not None and len(ranked):
        n_blobs = ranked.groupby("id").size().to_dict()
        top = ranked[ranked["rank_in_dataset"] == 1].set_index("id")
        best = top[["rank", "score", "peakz", "sample"]].to_dict("index")
    table["n_blobs"] = table["id"].map(lambda i: int(n_blobs.get(i, 0)))
    table["top_blob_rank"] = table["id"].map(lambda i: best.get(i, {}).get("rank", np.nan)).astype("Int64")
    table["top_blob_score"] = table["id"].map(lambda i: best.get(i, {}).get("score", np.nan))
    table["top_blob_peakz"] = table["id"].map(lambda i: best.get(i, {}).get("peakz", np.nan))
    table["top_blob_sample"] = table["id"].map(lambda i: best.get(i, {}).get("sample", ""))
    return table[base_cols + ["n_blobs", "top_blob_rank", "top_blob_score", "top_blob_peakz", "top_blob_sample"]]


def _top_hits_table(top_hits):
    cols = ["rank", "id", "original_name", "sample", "blob", "score", "peakz", "volume", "nearest_residue",
            "nearest_atom", "nearest_distance_A", "nearest_water_A", "diff_max_sigma", "diff_min_sigma",
            "diff_at_centroid_sigma", "diff_sign", "diff_peak_sigma", "on_known_ligand", "flags",
            "display_x", "display_y", "display_z", "diff_map", "context_map", "model", "notes"]
    rows = [{c: h.get(c, "") for c in cols} for h in top_hits]
    return pd.DataFrame(rows, columns=cols)


def _counts(datasets, ctx, ranked):
    out = {
        "datasets": int(len(datasets)) if datasets is not None else None,
        "datasets_ok": None,
        "datasets_dropped": None,
        "blobs_found": int(len(ctx["blobs_all"])) if ctx["blobs_all"] is not None else None,
        "blobs_kept": int(len(ranked)) if ranked is not None else 0,
        "datasets_with_blobs": int(ranked["id"].nunique()) if ranked is not None and len(ranked) else 0,
        "blob_table": ctx["blob_source"],
    }
    if datasets is not None and "status" in datasets:
        out["datasets_ok"] = int((datasets["status"] == "ok").sum())
        out["datasets_dropped"] = int((datasets["status"] == "dropped").sum())
    return out


def _funnel(datasets, state, ranked):
    if datasets is None or "id" not in datasets:
        return []
    n0 = len(datasets)
    # "skipped" = not applicable to this project (e.g. refine with phases from the data files)
    stages_run = {r["stage"] for r in _stage_rows(state)
                  if r.get("status") not in (None, "", "pending", "skipped")}
    dropped_at = datasets["dropped_stage"].value_counts().to_dict() if "dropped_stage" in datasets else {}
    rows = [{"stage": "input", "label": FUNNEL_LABELS["input"], "count": n0, "dropped": 0}]
    remaining = n0
    # files excluded before the pipeline started (unreadable, other crystal form): dropped_stage "input"
    lost = int(dropped_at.pop("input", 0))
    if lost:
        remaining -= lost
        rows.append({"stage": "input_checks", "label": FUNNEL_LABELS["input_checks"], "count": remaining,
                     "dropped": lost})
    for stage in STAGE_ORDER:
        if stage == "report":
            continue
        if stage not in stages_run and stage not in dropped_at:
            continue
        lost = int(dropped_at.get(stage, 0))
        remaining -= lost
        rows.append({"stage": stage, "label": FUNNEL_LABELS.get(stage, stage), "count": remaining,
                     "dropped": lost})
    other = {k: v for k, v in dropped_at.items() if k and k not in STAGE_ORDER}
    for stage, lost in sorted(other.items()):
        remaining -= int(lost)
        rows.append({"stage": stage, "label": stage, "count": remaining, "dropped": int(lost)})
    if ranked is not None:
        rows.append({"stage": "hits", "label": "With at least one candidate blob",
                     "count": int(ranked["id"].nunique()) if len(ranked) else 0, "dropped": None})
    return rows


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def _common_reason(reasons):
    """One text for reasons that differ only in numbers: differing numbers become '...'."""
    first = reasons[0]
    if all(r == first for r in reasons):
        return first
    parts = [_NUMBER.split(r) for r in reasons]
    numbers = [_NUMBER.findall(r) for r in reasons]
    text = parts[0][0]
    for k, value in enumerate(numbers[0]):
        same = all(len(n) > k and n[k] == value for n in numbers)
        text += (value if same else "...") + parts[0][k + 1]
    return text


def _varying_numbers(reasons):
    """For each reason, the numbers that differ within the group (in order); [] when all agree."""
    numbers = [_NUMBER.findall(r) for r in reasons]
    if len(reasons) < 2:
        return [[] for _ in reasons]
    width = min(len(n) for n in numbers)
    varying = [k for k in range(width) if len({n[k] for n in numbers}) > 1]
    return [[n[k] for k in varying] for n in numbers]


def _dropped_summary(datasets):
    if datasets is None or "status" not in datasets:
        return []
    d = datasets[datasets["status"] == "dropped"].copy()
    if not len(d):
        return []
    for col in ("dropped_stage", "dropped_reason", "original_name"):
        if col not in d:
            d[col] = ""
    # Reasons that differ only in a measured value ("CC ... 0.349 is below min_cc 0.55") form one
    # group; numbers that differ inside a group are shown as "..." (thresholds stay).
    d["_reason_key"] = d["dropped_reason"].astype(str).map(lambda t: _NUMBER.sub("#", t))
    out = []
    for (stage, _key), grp in d.groupby(["dropped_stage", "_reason_key"], sort=False):
        reasons = grp["dropped_reason"].astype(str).tolist()
        values = _varying_numbers(reasons)   # the numbers shown as "..." in the grouped reason
        names = []
        for i, n, vals in zip(grp["id"], grp.get("original_name", grp["id"]), values):
            label = f"{i} ({n})" if n and n != i else i
            names.append(f"{label}: {', '.join(vals)}" if vals else label)
        out.append({"stage": stage, "reason": _common_reason(reasons),
                    "count": int(len(grp)), "examples": names[:8]})
    out.sort(key=lambda r: (STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 99, -r["count"]))
    return out


def _stage_rows(state):
    if not isinstance(state, dict):
        return []
    stages = state.get("stages")
    if isinstance(stages, dict):
        items = list(stages.items())
    elif isinstance(stages, list):
        items = [(s.get("name") or s.get("stage"), s) for s in stages if isinstance(s, dict)]
    else:
        return []
    rows = []
    for name, info in items:
        if not isinstance(info, dict):
            continue
        secs = None
        for key in ("seconds", "duration_s", "elapsed_s", "wall_s", "duration"):
            if isinstance(info.get(key), (int, float)):
                secs = float(info[key])
                break
        if secs is None:
            secs = _seconds_between(info.get("started") or info.get("start"),
                                    info.get("finished") or info.get("end"))
        counts = info.get("counts") if isinstance(info.get("counts"), dict) else {}
        n_in = _first(counts, ("in", "n_in", "datasets_in", "input")) or _first(info, ("n_in",))
        n_out = _first(counts, ("out", "n_out", "datasets_out", "output")) or _first(info, ("n_out",))
        status = info.get("status")
        if name == "report" and status == "running":
            status = "this report"   # the report stage is still running while it writes this table
        rows.append({"stage": name, "status": status, "seconds": secs,
                     "n_in": n_in, "n_out": n_out})
    rows.sort(key=lambda r: STAGE_ORDER.index(r["stage"]) if r["stage"] in STAGE_ORDER else 99)
    return rows


def _refinement(ctx, datasets):
    """Which program refined the datasets (and with which twin law), from the refine stage's record."""
    st = ((ctx.get("state") or {}).get("stages") or {}).get("refine") or {}
    if st.get("status") != "done":
        return None
    out = st.get("outputs") or {}
    prog = out.get("program") or {}
    res = {"program": prog.get("program"), "version": prog.get("version"),
           "location": prog.get("root") if prog.get("source") == "valdo.yaml" else prog.get("exe"),
           "source": prog.get("source"), "setting": prog.get("setting"), "twin_law": out.get("twin_law"),
           "twin": None, "versions": {}}
    if ctx.get("workdir"):
        from valdo.easy.check import recorded_versions
        res["versions"] = recorded_versions(ctx["workdir"])   # {version: refined variants}
    if out.get("twin_law") and datasets is not None and "twin_fraction" in datasets:
        ok = datasets[datasets["status"] == "ok"]
        all_vals = pd.to_numeric(ok["twin_fraction"], errors="coerce")
        vals = all_vals.dropna()
        notes = ok["twin_note"].astype(str) if "twin_note" in ok else pd.Series([], dtype=str)
        res["twin"] = {"n": int(len(vals)), "n_missing": int(all_vals.isna().sum()),
                       "median": round(float(vals.median()), 4) if len(vals) else None,
                       "min": round(float(vals.min()), 4) if len(vals) else None,
                       "max": round(float(vals.max()), 4) if len(vals) else None,
                       "n_untwinned": int((vals == 0).sum()),
                       "n_above_half": int((vals > 0.5).sum()),
                       "n_with_note": int((notes != "").sum())}
    return res


def _refinement_lines(ref):
    """Plain-text lines for the report's refinement section."""
    if not ref:
        return []
    name = {"phenix": "phenix.refine (PHENIX)", "dimple": "dimple (CCP4)"}.get(ref.get("program"), ref.get("program"))
    where = ref.get("location") or "?"
    origin = (f"{ref.get('setting')} in valdo.yaml" if ref.get("source") == "valdo.yaml" else
              ("found on PATH" if ref.get("source") else "recorded before valdo-easy noted the program"))
    lines = [f"Program: {name}" + (f" {ref['version']}" if ref.get("version") else "") + f" at {where} ({origin})"]
    versions = ref.get("versions") or {}
    if len(versions) > 1 or (versions and ref.get("version") and set(versions) != {ref["version"]}):
        lines.append("Refined variants by program version: "
                     + ", ".join(f"{v}: {n}" for v, n in sorted(versions.items()))
                     + " (datasets refined by an earlier run were not refined again; 'valdo check' says how to "
                       "refine them all with one version).")
    if ref.get("twin_law"):
        t = ref.get("twin") or {}
        text = f"Twin refinement with twin law {ref['twin_law']}"
        if t.get("n"):
            text += (f": twin fraction median {t['median']:.3f} (range {t['min']:.3f}-{t['max']:.3f}) over "
                     f"{t['n']} datasets")
            if t.get("n_untwinned"):
                from valdo.easy.twin import zero_fraction_text
                text += f"; {zero_fraction_text(ref.get('program'), t['n_untwinned'])}"
            if t.get("n_above_half"):
                text += (f"; {t['n_above_half']} above 0.5 (the model and phases follow the smaller twin "
                         "domain there)")
        if t.get("n_missing"):
            text += (f"{';' if t.get('n') else ':'} twin fraction not found for {t['n_missing']} "
                     f"dataset{'s' if t['n_missing'] != 1 else ''} (datasets.csv twin_note says why)")
        lines.append(text + ". Per dataset: datasets.csv (twin_law, twin_fraction, twin_note).")
    else:
        lines.append("No twin refinement (phases.twin_law is null).")
    return lines


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _seconds_between(a, b):
    def parse(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            try:
                return datetime.datetime.fromisoformat(x).timestamp()
            except ValueError:
                return None
        return None
    ta, tb = parse(a), parse(b)
    return tb - ta if ta is not None and tb is not None and tb >= ta else None


def _settings(ctx):
    cfg = ctx["configs"]
    picks = [
        ("add_phases_and_blobs", ["blob_diff_col", "phase_col", "cutoff", "radius_in_A", "sigdF_pct",
                                  "absdF_pct", "phase_2FOFC_col_in", "phase_FOFC_col_in"]),
        ("tag_blobs", ["focal_seqid", "focal_tag_name", "focal_radius"]),
        ("reindex", ["wcorr", "cc_min_dif"]),
        ("scale", ["when_opt"]),
        ("filter", ["max_rfree", "min_cc"]),
        ("train", ["latent_dim", "n_hidden_layers", "n_hidden_size", "epochs", "batch_size",
                   "learning_rate", "stdof", "include_errors", "train_fraction", "random_seed",
                   "activation"]),
    ]
    out = []
    for stage, keys in picks:
        c = cfg.get(stage) or {}
        for k in keys:
            if k in c:
                out.append({"stage": stage, "key": k, "value": c[k]})
    return out


# ---------------------------------------------------------------------------
# Top hits: residues, maps, models
# ---------------------------------------------------------------------------

def _reset_hits_dir(hits_dir, workdir):
    """Empty report/hits/ if this module made it (marker file), else create it."""
    if os.path.islink(hits_dir):
        raise RuntimeError(f"{hits_dir} is a symbolic link; refusing to write report maps there")
    if os.path.isdir(hits_dir):
        inside = os.path.commonpath([hits_dir, workdir]) == workdir
        if inside and os.path.isfile(os.path.join(hits_dir, HITS_MARKER)):
            shutil.rmtree(hits_dir)
        else:
            raise RuntimeError(f"{hits_dir} exists but was not written by valdo-easy report; "
                               "move it away and run the report again")
    os.makedirs(hits_dir)
    with open(os.path.join(hits_dir, HITS_MARKER), "w") as fh:
        fh.write("written by valdo.easy.report; rewritten on every report\n")


def _describe_top_hits(top, ctx, hits_dir, warnings):
    import gemmi

    hits = []
    order = list(dict.fromkeys(top["sample"]))  # samples in rank order, each once
    rows_by_sample = {s: top[top["sample"] == s] for s in order}
    results = {}
    for sample in order:
        rows = rows_by_sample[sample]
        notes_common = []
        phased = os.path.join(ctx["phased_dir"], f"{sample}.mtz")
        model_path = maps.locate_model(sample, ctx["model_folder"])
        structure = None
        if model_path:
            try:
                structure = gemmi.read_structure(model_path)
            except Exception as exc:
                notes_common.append(f"model unreadable: {exc}")
        else:
            notes_common.append("no model found")
        diff_grid = context_grid = None
        context_info = None
        if os.path.isfile(phased):
            try:
                diff_grid = maps.map_from_mtz(phased, ctx["diff_col"], ctx["phase_col"])
            except Exception as exc:
                notes_common.append(f"difference map failed: {exc}")
            context_info = _context_source(phased, sample, ctx)
            if context_info and hits_dir:
                try:
                    context_grid = maps.map_from_mtz(context_info[0], context_info[1], context_info[2])
                except Exception as exc:
                    notes_common.append(f"context map failed: {exc}")
                    context_info = None
        else:
            notes_common.append(f"phased MTZ not found: {phased}")
        model_copy = ""
        if hits_dir and model_path:
            model_copy = os.path.join(hits_dir, f"{sample}.pdb")
            shutil.copyfile(model_path, model_copy)
        for _, row in rows.iterrows():
            hit = _hit_base(row)
            notes = list(notes_common)
            centroid = gemmi.Position(float(row["cenx"]), float(row["ceny"]), float(row["cenz"]))
            display = centroid
            if structure is not None:
                try:
                    near = maps.nearest_residue(structure, centroid)
                    if near:
                        hit.update(nearest_residue=near["label"], nearest_atom=near["atom"],
                                   nearest_distance_A=round(near["dist"], 2))
                        display = near["display_pos"]
                    else:
                        hit.update(nearest_residue="none within 12 A")
                    water = maps.nearest_water_distance(structure, centroid)
                    hit["nearest_water_A"] = round(water, 2) if water is not None else ""
                except Exception as exc:  # an odd model must not stop the report
                    notes.append(f"nearest-residue search failed: {exc}")
            hit.update(display_x=round(display.x, 3), display_y=round(display.y, 3),
                       display_z=round(display.z, 3))
            if diff_grid is not None:
                vals = maps.values_near(diff_grid, centroid, max(float(row.get("radius", 1.5)), 1.0))
                # both extremes: a blob often holds strong positive and negative density, so the
                # largest-magnitude value alone (diff_peak_sigma) can make a ligand look negative
                hit["diff_peak_sigma"] = round(vals["max_abs"], 2)
                hit["diff_max_sigma"] = round(vals["max"], 2)
                hit["diff_min_sigma"] = round(vals["min"], 2)
                hit["diff_at_centroid_sigma"] = round(vals["at_center"], 2)
                hit["diff_sign"] = diff_sign(vals["max"], vals["min"])
            on_ligand = ctx.get("on_ligand", {})
            if int(row["rank"]) in on_ligand:  # same overlap rule as the validation section
                hit["on_known_ligand"] = bool(on_ligand[int(row["rank"])])
            elif "ligand" in row.index and pd.notna(row["ligand"]) and int(row["ligand"]) == 1:
                hit["on_known_ligand"] = True  # upstream's tag (residue LIG of a bound model)
            if hits_dir:
                tag = f"hit{int(row['rank']):02d}_{sample}"
                if diff_grid is not None:
                    path = os.path.join(hits_dir, f"{tag}_diff.ccp4")
                    maps.write_ccp4(diff_grid, path, center=display, radius=BOX_RADIUS_A)
                    hit["diff_map"] = os.path.relpath(path, os.path.dirname(hits_dir))
                if context_grid is not None:
                    path = os.path.join(hits_dir, f"{tag}_2fofc.ccp4")
                    maps.write_ccp4(context_grid, path, center=display, radius=BOX_RADIUS_A)
                    hit["context_map"] = os.path.relpath(path, os.path.dirname(hits_dir))
                    hit["context_source"] = f"{os.path.basename(context_info[0])} {context_info[1]}/{context_info[2]}"
                elif diff_grid is not None:
                    notes.append("no 2mFo-DFc coefficients in the phase source: context map not written")
                if model_copy:
                    hit["model"] = os.path.relpath(model_copy, os.path.dirname(hits_dir))
            hit["notes"] = "; ".join(dict.fromkeys(notes))
            results[int(row["rank"])] = hit
        del diff_grid, context_grid
    for r in sorted(results):
        hits.append(results[r])
    if any("no model found" in h["notes"] for h in hits):
        warnings.append("some top hits have no model in the model folder: nearest residues are missing")
    return hits


def _hit_base(row):
    def num(key, nd=3):
        v = row.get(key)
        try:
            return round(float(v), nd)
        except (TypeError, ValueError):
            return ""
    return {
        "rank": int(row["rank"]),
        "id": row["id"],
        "original_name": row.get("original_name", ""),
        "sample": row["sample"],
        "rank_in_dataset": int(row.get("rank_in_dataset", 0)),
        "blob": f"{row['sample']} #{int(row.get('rank_in_dataset', 0))}",
        "score": num("score", 2),
        "peakz": num("peakz", 2),
        "peak": num("peak", 3),
        "volume": num("volume", 1),
        "radius": num("radius", 2),
        "cenx": num("cenx"), "ceny": num("ceny"), "cenz": num("cenz"),
        "flags": row.get("flags", ""),
        "nearest_residue": "", "nearest_atom": "", "nearest_distance_A": "", "nearest_water_A": "",
        "diff_peak_sigma": "", "diff_max_sigma": "", "diff_min_sigma": "", "diff_at_centroid_sigma": "",
        "diff_sign": "", "on_known_ligand": None,
        "diff_map": "", "context_map": "", "model": "", "notes": "",
    }


def diff_sign(vmax, vmin, level=DIFF_LEVEL_SIGMA, dominance=DIFF_DOMINANCE):
    """What the difference map shows within the blob: 'positive' (density appeared) or 'negative'
    (density left) when only that side reaches `level` sigma, or when both do and it is at least
    `dominance` times stronger than the other; 'mixed' when both reach the level and neither
    dominates; 'weak' when neither reaches it. (A bound ligand usually comes with weaker negative
    density of the waters it displaced: on the PTP1B screen every top hit reached -3 sigma
    somewhere within its radius, so 'reaches the level on both sides' alone said nothing.)"""
    pos, neg = vmax >= level, vmin <= -level
    if pos and neg:
        if vmax >= dominance * -vmin:
            return "positive"
        if -vmin >= dominance * vmax:
            return "negative"
        return "mixed"
    return "positive" if pos else ("negative" if neg else "weak")


def _context_source(phased_mtz, sample, ctx):
    """(path, F, PHI) of 2mFo-DFc-style coefficients for this dataset, or None.

    Looks first in the phase file upstream's add_phases used (same lookup rule),
    then in the phase_file recorded in datasets.csv.
    """
    candidates = []
    if ctx.get("phasing_path"):
        path, handler = maps.locate_phase_file(phased_mtz, ctx["phasing_path"])
        if path and handler > 0:
            candidates.append(path)
    ds = ctx.get("datasets")
    ident, symop = _split_stem(sample)
    if ds is not None and "phase_file" in ds and symop in ("", "0"):
        # only for datasets kept in their original indexing (operator 0 = identity):
        # an original phase file of a re-indexed dataset is in the other setting
        match = ds.loc[ds["id"] == ident, "phase_file"]
        candidates += [p for p in match if p and os.path.isfile(p)]
    for path in dict.fromkeys(candidates):
        try:
            cols = maps.mtz_column_types(path)
        except Exception:
            continue
        pairs = maps.find_coefficient_pairs(cols, "2fofc", preferred_phase=ctx.get("phase_2fofc_in"))
        if pairs:
            return path, pairs[0][0], pairs[0][1]
    return None


def _public_hit(hit):
    return {k: v for k, v in hit.items()}


# ---------------------------------------------------------------------------
# Known-bound validation (descriptive only)
# ---------------------------------------------------------------------------

def _bound_models(ctx, known_bound_dir, warnings):
    """Map dataset ID -> bound model path."""
    ds = ctx["datasets"]
    ids = list(ds["id"]) if ds is not None and "id" in ds else []
    out = {}
    if known_bound_dir:
        folder = os.path.abspath(os.path.expanduser(known_bound_dir))
        if not os.path.isdir(folder):
            warnings.append(f"known bound models folder not found: {folder}")
            return {}
        files = sorted(glob.glob(os.path.join(folder, "*.pdb")) + glob.glob(os.path.join(folder, "*.cif")))
        names = {}
        if ds is not None and "original_name" in ds:
            names = {i: n for i, n in zip(ds["id"], ds["original_name"]) if n}
        width = max([len(i) for i in ids] or [4])
        for f in files:
            base = os.path.basename(f)
            stem = os.path.splitext(base)[0]
            ident = None
            if stem in ids:
                ident = stem
            else:
                hits = [(len(n), i) for i, n in names.items() if n in base]
                if hits:
                    ident = max(hits)[1]
                else:
                    m = re.match(r"[a-zA-Z]*(\d+)", stem)
                    if m and m.group(1).zfill(width) in ids:
                        ident = m.group(1).zfill(width)
            if ident and ident not in out:
                out[ident] = f
        if files and not out:
            warnings.append(f"none of the {len(files)} files in {folder} could be matched to a dataset")
        return out
    if ds is not None and "bound_model" in ds:
        out = {i: p for i, p in zip(ds["id"], ds["bound_model"]) if p and os.path.isfile(p)}
        if out:
            return out
    farm = os.path.join(ctx["workdir"], "inputs", "bound_models")
    for f in sorted(glob.glob(os.path.join(farm, "*.pdb"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if not ids or stem in ids:
            out[stem] = f
    return out


def _validate_known_bound(ranked, ctx, bound, ligand_resnames, warnings):
    import gemmi

    fixed = {r.strip().upper() for r in ligand_resnames} if ligand_resnames else None
    ds = ctx["datasets"]
    status = dict(zip(ds["id"], ds["status"])) if ds is not None and "status" in ds else {}
    names = dict(zip(ds["id"], ds["original_name"])) if ds is not None and "original_name" in ds else {}
    per_dataset = []
    labels = np.zeros(len(ranked), dtype=int) if ranked is not None else np.zeros(0, dtype=int)
    resnames_seen = set()
    on_ligand = ctx.setdefault("on_ligand", {})  # rank -> blob overlaps a ligand of a known bound model
    for ident, path in sorted(bound.items()):
        row = {"id": ident, "original_name": names.get(ident, ""), "bound_model": path,
               "analysed": status.get(ident, "ok") == "ok", "ligand_residues": "",
               "n_blobs": 0, "first_ligand_blob_rank_in_dataset": None,
               "first_ligand_blob_global_rank": None, "closest_blob_distance_A": None, "note": ""}
        try:
            st = gemmi.read_structure(path)
        except Exception as exc:
            row["note"] = f"unreadable model: {exc}"
            per_dataset.append(row)
            continue
        model = st[0]
        lig_names = set()
        for chain in model:
            for res in chain:
                name = res.name.upper()
                if fixed is not None:
                    if name in fixed:
                        lig_names.add(name)
                elif res.het_flag == "H" and name not in NOT_LIGANDS:
                    lig_names.add(name)
        row["ligand_residues"] = " ".join(sorted(lig_names))
        resnames_seen |= lig_names
        if not lig_names:
            row["note"] = "no ligand residue in this model"
            per_dataset.append(row)
            continue
        if ranked is None or not len(ranked):
            per_dataset.append(row)
            continue
        sel = ranked.index[ranked["id"] == ident]
        row["n_blobs"] = int(len(sel))
        for chain in model:  # keep only ligand residues: the search below then sees ligand atoms only
            for i in reversed(range(len(chain))):
                if chain[i].name.upper() not in lig_names:
                    del chain[i]
        ns = gemmi.NeighborSearch(model, st.cell, 10.0).populate()
        closest = None
        for idx in sel:
            b = ranked.loc[idx]
            pos = gemmi.Position(float(b["cenx"]), float(b["ceny"]), float(b["cenz"]))
            dmin = None
            for mark in ns.find_atoms(pos, "\0", radius=10.0):
                cra = mark.to_cra(model)
                if cra.residue.name.upper() not in lig_names:
                    continue
                d = st.cell.find_nearest_pbc_image(pos, cra.atom.pos, mark.image_idx).dist()
                dmin = d if dmin is None else min(dmin, d)
            if dmin is not None:
                closest = dmin if closest is None else min(closest, dmin)
            on_ligand[int(b["rank"])] = dmin is not None and dmin <= float(b["radius"])
            if dmin is not None and dmin <= float(b["radius"]):
                labels[ranked.index.get_loc(idx)] = 1
                if row["first_ligand_blob_rank_in_dataset"] is None:
                    row["first_ligand_blob_rank_in_dataset"] = int(b["rank_in_dataset"])
                    row["first_ligand_blob_global_rank"] = int(b["rank"])
        row["closest_blob_distance_A"] = round(closest, 2) if closest is not None else None
        per_dataset.append(row)

    analysed = [r for r in per_dataset if r["analysed"] and r["ligand_residues"]]
    found = [r for r in analysed if r["first_ligand_blob_rank_in_dataset"] is not None]
    ranks = [r["first_ligand_blob_rank_in_dataset"] for r in found]
    summary = {
        "description": ("Descriptive only: for each dataset with a known bound model, the rank (within "
                        "that dataset, and in the whole hit list) of the first kept blob whose sphere "
                        "(radius (3V/4pi)^(1/3) around the centroid, symmetry included) contains a ligand "
                        "atom - the same overlap rule as upstream tag_blobs."),
        "ligand_resnames": sorted(fixed) if fixed else "auto: " + (" ".join(sorted(resnames_seen)) or "none found"),
        "n_bound_models": len(bound),
        "n_analysed": len(analysed),
        "n_with_ligand_blob": len(found),
        "n_top1": int(sum(1 for r in ranks if r == 1)),
        "median_rank_in_dataset": float(np.median(ranks)) if ranks else None,
        "per_dataset": per_dataset,
        "auc_score": None,
        "n_positive_blobs": int(labels.sum()),
    }
    if ranked is not None and len(ranked) and 0 < labels.sum() < len(labels):
        summary["auc_score"] = round(_roc_auc(labels, ranked["score"].to_numpy(dtype=float)), 4)
        if "ligand" in ranked and ranked["ligand"].astype(float).sum() > 0:
            agree = int((ranked["ligand"].astype(int).to_numpy() == labels).sum())
            summary["agreement_with_upstream_ligand_tags"] = f"{agree}/{len(labels)} blobs"
    if not analysed:
        warnings.append("known bound models were given but none matches an analysed dataset with a ligand")
    return summary


def _roc_auc(labels, scores):
    """ROC AUC as upstream computes it (sklearn roc_curve + auc), with a numpy fallback."""
    try:
        from sklearn import metrics
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        return float(metrics.auc(fpr, tpr))
    except ImportError:  # pragma: no cover
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores))
        sorted_scores = scores[order]
        i = 0
        while i < len(scores):
            j = i
            while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        pos = labels == 1
        n_pos, n_neg = pos.sum(), (~pos).sum()
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

_QC_CAPTIONS = [
    ("reindex_cc_diff_histogram", "Reindexing: CC difference",
     "CC(alternative indexing) - CC(identity) per dataset, against the reference. Two clearly separated "
     "groups mean the indexing ambiguity was resolved; datasets near 0 are ambiguous (both variants are "
     "kept until the filter stage)."),
    ("reindex_cc_diff_scatter", "Reindexing: CC difference vs best CC",
     "Low best CC (left) means the dataset correlates poorly with the reference under every operator."),
    ("scaling_end_corr_histogram", "Scaling: correlation with the reference",
     "Correlation with the reference after scaling. The filter stage drops datasets below min_cc."),
    ("scaling_LS_scatter", "Scaling: loss before vs after",
     "Least-squares loss before and after scaling; points above the diagonal got worse."),
    ("vae_loss_curves", "VAE training curves",
     "Training and validation loss. Validation loss should level off; a rising validation curve means "
     "over-fitting, NaN or spikes mean the training failed."),
]


def _qc_plots(ctx):
    workdir, cfg = ctx["workdir"], ctx["configs"]
    folders = [
        (cfg["reindex"].get("output_folder") or os.path.join(workdir, STAGE_DIRS["reindex"])),
        (cfg["scale"].get("output_folder") or os.path.join(workdir, STAGE_DIRS["scale"])),
        (os.path.dirname(cfg["train"].get("output_path") or "") or os.path.join(workdir, STAGE_DIRS["train"])),
    ]
    out = []
    seen = set()
    for folder in folders:
        for png in sorted(glob.glob(os.path.join(folder, "*.png"))):
            if png in seen:
                continue
            seen.add(png)
            base = os.path.basename(png)
            title, caption = base, ""
            for key, t, c in _QC_CAPTIONS:
                if key in base:
                    title, caption = t, c
                    break
            try:
                with open(png, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode("ascii")
            except OSError:
                continue
            out.append({"title": title, "caption": caption, "path": png, "data": data})
    order = {t: i for i, (_, t, _) in enumerate(_QC_CAPTIONS)}
    out.sort(key=lambda q: order.get(q["title"], 99))
    return out


# ---------------------------------------------------------------------------
# Viewer scripts
# ---------------------------------------------------------------------------

def _write_coot_script(hits, path):
    entries = []
    for h in hits:
        if not h.get("diff_map"):
            continue
        entries.append({
            "rank": h["rank"], "sample": h["sample"], "id": h["id"], "name": h.get("original_name", ""),
            "model": h.get("model") or None, "diff": h["diff_map"], "context": h.get("context_map") or None,
            "center": [h["display_x"], h["display_y"], h["display_z"]],
            "score": h["score"], "peakz": h["peakz"], "residue": h.get("nearest_residue", ""),
        })
    text = _COOT_TEMPLATE.replace("@@HITS@@", json.dumps(entries, indent=1)) \
        .replace("@@DIFF@@", repr(DIFF_LEVEL_SIGMA)).replace("@@CONTEXT@@", repr(CONTEXT_LEVEL_SIGMA)) \
        .replace("@@REPORT@@", repr(os.path.dirname(os.path.abspath(path))))
    _write_text(path, text)


_COOT_TEMPLATE = '''# VALDO top hits for Coot (0.9 and 1.x), written by valdo-easy.
#
#   cd <workdir>/report
#   coot --script view_top_hits.py
#
# Opens hit 1 (model, signed difference map, context map) centred on the blob.
# Then, in Coot's Python console (Calculate > Scripting > Python):
#   valdo_next()   valdo_prev()   valdo_go(5)
# The difference map is in sigma units of the whole cell: green = +@@DIFF@@ sigma
# (density that appeared), red = -@@DIFF@@ sigma (density that went away). The
# blob finder looks at |difference|, so check which colour the blob is.
import json
import os

HITS = json.loads(r"""@@HITS@@""")
DIFF_LEVEL = @@DIFF@@
CONTEXT_LEVEL = @@CONTEXT@@

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
try:
    import coot as _coot
except ImportError:
    _coot = None


def _fn(name):
    if _coot is not None and hasattr(_coot, name):
        return getattr(_coot, name)
    import __main__
    if hasattr(__main__, name):
        return getattr(__main__, name)
    return globals()[name]


# where the hit files may be: next to this script, the current folder, its report/
# subfolder (Coot started from the work folder), the folder the report was written to
_ROOTS = [_HERE, os.getcwd(), os.path.join(os.getcwd(), "report"), @@REPORT@@]


def _path(rel):
    for root in _ROOTS:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    print("VALDO: %s not found; start Coot from the report folder (cd <workdir>/report)" % rel)
    return rel


_state = {"index": None, "mols": []}


def valdo_go(rank):
    """Show hit number `rank` (1 = best)."""
    idx = [i for i, h in enumerate(HITS) if h["rank"] == int(rank)]
    if not idx:
        print("VALDO: no hit with rank %s (ranks: %s)" % (rank, [h["rank"] for h in HITS]))
        return
    for imol in _state["mols"]:
        try:
            _fn("close_molecule")(imol)
        except Exception:
            pass
    _state["mols"] = []
    hit = HITS[idx[0]]
    if hit["model"]:
        _state["mols"].append(_fn("read_pdb")(_path(hit["model"])))
    imol = _fn("handle_read_ccp4_map")(_path(hit["diff"]), 1)
    _fn("set_contour_level_absolute")(imol, DIFF_LEVEL)
    _state["mols"].append(imol)
    if hit["context"]:
        imol = _fn("handle_read_ccp4_map")(_path(hit["context"]), 0)
        _fn("set_contour_level_absolute")(imol, CONTEXT_LEVEL)
        _state["mols"].append(imol)
    x, y, z = hit["center"]
    _fn("set_rotation_centre")(x, y, z)
    _state["index"] = idx[0]
    print("VALDO hit %d: %s %s score %s peak z %s near %s" % (
        hit["rank"], hit["sample"], hit["name"], hit["score"], hit["peakz"], hit["residue"]))


def valdo_next():
    i = 0 if _state["index"] is None else min(_state["index"] + 1, len(HITS) - 1)
    valdo_go(HITS[i]["rank"])


def valdo_prev():
    i = 0 if _state["index"] is None else max(_state["index"] - 1, 0)
    valdo_go(HITS[i]["rank"])


try:  # a VALDO menu where Coot's Python GUI helpers exist (Coot 0.9)
    import coot_gui
    _menu = coot_gui.coot_menubar_menu("VALDO")
    coot_gui.add_simple_coot_menu_menuitem(_menu, "Next hit", lambda func: valdo_next())
    coot_gui.add_simple_coot_menu_menuitem(_menu, "Previous hit", lambda func: valdo_prev())
except Exception:
    pass

if HITS:
    valdo_go(HITS[0]["rank"])
else:
    print("VALDO: no hits with maps in this report")
'''


def _write_pymol_script(hits, path):
    lines = [
        "# VALDO top hits for PyMOL, written by valdo-easy.",
        "#",
        "#   cd <workdir>/report",
        "#   pymol view_top_hits.pml",
        "#",
        "# Every hit is a scene: PageDown / PageUp step through them (or type: scene hit02).",
        f"# Difference map: green mesh = +{DIFF_LEVEL_SIGMA:g} sigma, red = -{DIFF_LEVEL_SIGMA:g} sigma "
        "(whole-cell sigma);",
        f"# blue mesh = 2mFo-DFc-style map at {CONTEXT_LEVEL_SIGMA:g} sigma (when available).",
        "# The paths below are relative to the report folder; the python block finds it when PyMOL",
        "# was started elsewhere (the work folder, or anywhere while the report is still where it was written).",
        "python",
        "import os as _os",
        f"for _d in ('.', 'report', {os.path.dirname(os.path.abspath(path))!r}):",
        f"    if _os.path.isfile(_os.path.join(_d, 'hits', {HITS_MARKER!r})):",
        "        _os.chdir(_d)",
        "        break",
        "else:",
        "    print('VALDO: hits/ folder not found. Open this script from the report folder: '",
        "          'cd <workdir>/report; pymol view_top_hits.pml')",
        "python end",
        "set normalize_ccp4_maps, 0",
        "set mesh_width, 0.7",
        "set sphere_scale, 0.35",
    ]
    models_loaded = set()
    shown = []
    for h in hits:
        if not h.get("diff_map"):
            continue
        tag = f"hit{h['rank']:02d}"
        model_obj = ""
        if h.get("model"):
            model_obj = "m_" + re.sub(r"[^A-Za-z0-9_]", "_", h["sample"])
            if model_obj not in models_loaded:
                lines.append(f"load {h['model']}, {model_obj}")
                models_loaded.add(model_obj)
        lines += [
            f"load {h['diff_map']}, {tag}_diff",
            f"isomesh {tag}_pos, {tag}_diff, {DIFF_LEVEL_SIGMA:g}",
            f"isomesh {tag}_neg, {tag}_diff, {-DIFF_LEVEL_SIGMA:g}",
            f"color green, {tag}_pos",
            f"color red, {tag}_neg",
            f"disable {tag}_diff",
        ]
        if h.get("context_map"):
            lines += [
                f"load {h['context_map']}, {tag}_ctx",
                f"isomesh {tag}_2fofc, {tag}_ctx, {CONTEXT_LEVEL_SIGMA:g}",
                f"color slate, {tag}_2fofc",
                f"disable {tag}_ctx",
            ]
        lines += [
            f"pseudoatom {tag}_centre, pos=[{h['display_x']}, {h['display_y']}, {h['display_z']}]",
            f"show spheres, {tag}_centre",
            f"color yellow, {tag}_centre",
        ]
        shown.append((tag, model_obj, h))
    if models_loaded:
        lines.append("hide everything, m_*")
        lines.append("show lines, m_*")
        lines.append("show sticks, m_* and not polymer and not solvent")
    for tag, model_obj, h in shown:
        msg = re.sub(r"[,;\"]", " ", f"VALDO hit {h['rank']}: {h['sample']} {h.get('original_name', '')} "
                                       f"score {h['score']} peak z {h['peakz']} {h.get('nearest_residue', '')}")
        lines += [
            "disable all",
            f"enable {tag}_*",
            f"disable {tag}_diff",
        ]
        if h.get("context_map"):
            lines.append(f"disable {tag}_ctx")
        if model_obj:
            lines.append(f"enable {model_obj}")
        lines += [
            f"center {tag}_centre",
            f"zoom {tag}_centre, 8",
            f"scene {tag}, store, message={msg.strip()}",
        ]
    if shown:
        lines.append(f"scene {shown[0][0]}, recall")
    _write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v, nd=2):
    if v is None or v == "":
        return ""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{v:.{nd}f}"
    return str(v)


def _fmt_secs(s):
    if s is None:
        return ""
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


TOP_HITS_TEXT = (
    "Blobs kept after tagging (not within the focal-residue radius; upstream removes a symmetry copy only "
    "when it is next in peak order and within 1 A, so the table below also leaves out the remaining copies), "
    "ranked by score = gemmi blob score on the blurred |difference| map (what upstream's ROC/AUC evaluation "
    "ranks by); peak z = blob peak in sigma; Blob = dataset variant #rank within that dataset. Diff. max / min "
    "= the most positive and the most negative value of the unblurred difference map within the blob's "
    "radius (sigma units): positive = density appeared, negative = something left. A blob often holds both; "
    "Sign says which side dominates: positive or negative when only that side reaches {level} sigma or it is "
    "at least {dominance} times stronger than the other, mixed when both reach {level} sigma and neither "
    "dominates, weak when neither does. A bound ligand is usually positive (often with weaker negative density "
    "of displaced water nearby). The full list is in blobs.csv.")


def _top_hits_text():
    return TOP_HITS_TEXT.format(level=f"{DIFF_LEVEL_SIGMA:g}", dominance=f"{DIFF_DOMINANCE:g}")


def _yes_no(flag):
    """'yes' / 'no' for True / False; '' when unknown (no bound model for that dataset)."""
    return {True: "yes", False: "no"}.get(flag, "")


def _extremes(h):
    if h.get("diff_max_sigma") in (None, "") or h.get("diff_min_sigma") in (None, ""):
        return _fmt(h.get("diff_peak_sigma"))
    return f"{float(h['diff_max_sigma']):+.1f} / {float(h['diff_min_sigma']):+.1f}"


CAVEATS = [
    "A VALDO blob means \"the data differ from what the VAE expects for an apo crystal, here\": something "
    "changed. It is not proof of a bound ligand. Open the map and judge the shape.",
    "The blob search is sign-blind (it looks at |difference|). Green (positive) difference density is new "
    "density; red (negative) density is something that left (e.g. a moved side chain or water).",
    "Blobs at the surface next to waters, alternative conformations, or crystal contacts are common false "
    "positives; so are blobs in datasets flagged in the table.",
    "Ranks compare blobs across datasets with different resolution and quality; a weaker blob in a "
    "high-quality dataset can be more real than a strong one in a poor dataset.",
]


def _render_markdown(r):
    c = r["counts"]
    out = ["# VALDO report", "",
           f"Work folder: `{r['workdir']}`  ",
           f"Generated {r['generated']} by valdo {r['valdo_version']}", ""]
    out += ["## Summary", ""]
    if c.get("datasets") is not None:
        out.append(f"- Datasets: {c['datasets']} found, {c.get('datasets_ok')} analysed, "
                   f"{c.get('datasets_dropped')} dropped (see datasets.csv)")
    out.append(f"- Candidate blobs (after tagging): {c['blobs_kept']} in {c['datasets_with_blobs']} datasets"
               + (f" ({c['blobs_found']} before removing focal-residue blobs and duplicates)"
                  if c.get("blobs_found") is not None else ""))
    out.append(f"- Ranked by `{r['rank_by']}`")
    out.append("")
    if r["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    out += ["## Read this first", ""] + [f"- {cv}" for cv in CAVEATS] + [""]
    if r.get("refinement"):
        out += ["## Refinement", ""] + [f"- {line}" for line in _refinement_lines(r["refinement"])] + [""]
    if r["funnel"]:
        out += ["## Dataset funnel", "", "| Stage | Datasets | Dropped here |", "|---|---:|---:|"]
        for f in r["funnel"]:
            out.append(f"| {f['label']} | {f['count']} | {'' if f['dropped'] in (None, 0) else f['dropped']} |")
        out.append("")
    if r["top_hits"]:
        known = bool(r.get("validation"))
        out += ["## Top hits", "", _top_hits_text(), ""]
        if r.get("top_hits_symmetry_copies_left_out"):
            out += [f"{r['top_hits_symmetry_copies_left_out']} lower-ranked symmetry copies of blobs listed here "
                    "are left out (column `symmetry_copy_of` in blobs.csv), so ranks can skip numbers.", ""]
        out += ["| Rank | ID | Original name | Blob | Score | Peak z | Volume (A^3) | Nearest residue | "
                "Dist (A) | Diff. max / min (sigma) | Sign |" + (" Known ligand |" if known else "") + " Flags |",
                "|---:|---|---|---|---:|---:|---:|---|---:|---:|---|" + ("---|" if known else "") + "---|"]
        for h in r["top_hits"]:
            lig = f" {_yes_no(h.get('on_known_ligand'))} |" if known else ""
            out.append(f"| {h['rank']} | {h['id']} | {h['original_name']} | {h.get('blob', h['sample'])} | "
                       f"{_fmt(h['score'], 1)} | {_fmt(h['peakz'])} | {_fmt(h['volume'], 1)} | {h['nearest_residue']} | "
                       f"{_fmt(h['nearest_distance_A'])} | {_extremes(h)} | {h.get('diff_sign', '')} |{lig} "
                       f"{h['flags']} |")
        out += ["", "Open them: `cd report && coot --script view_top_hits.py` or "
                "`cd report && pymol view_top_hits.pml`.", ""]
    v = r.get("validation")
    if v:
        out += ["## Known bound models (descriptive only)", "", v["description"], "",
                f"- Bound models: {v['n_bound_models']}; analysed with a ligand: {v['n_analysed']}; "
                f"with a blob on the ligand: {v['n_with_ligand_blob']}; ligand blob ranked first in its "
                f"dataset: {v['n_top1']}",
                f"- Ligand residue names: {v['ligand_resnames']}"]
        if v.get("auc_score") is not None:
            out.append(f"- ROC AUC of blob score (ligand-overlapping blobs vs all others): {v['auc_score']}")
        out.append("")
    if r["dropped"]:
        out += ["## Dropped datasets", "", "| Stage | Reason | Count | Examples |", "|---|---|---:|---|"]
        for d in r["dropped"]:
            out.append(f"| {d['stage']} | {d['reason']} | {d['count']} | {', '.join(d['examples'])} |")
        out.append("")
    if r["stages"]:
        out += ["## Stages", "", "| Stage | Status | Time | In | Out |", "|---|---|---:|---:|---:|"]
        for s in r["stages"]:
            out.append(f"| {s['stage']} | {s['status'] or ''} | {_fmt_secs(s['seconds'])} | "
                       f"{'' if s['n_in'] is None else s['n_in']} | {'' if s['n_out'] is None else s['n_out']} |")
        out.append("")
    out += ["## Files", ""] + [f"- {k}: `{os.path.relpath(p, r['report_dir'])}`" for k, p in r["files"].items()]
    return "\n".join(out) + "\n"


_CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e2dd;
--accent:#2a78d6;--bar:#2a78d6;--warn-bg:#fff6e0;--warn-line:#eda100;--note-bg:#eef4fc;--zebra:#f6f6f3;
--plot-bg:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;
--card:#222220;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--bar:#3987e5;
--warn-bg:#3a2f12;--warn-line:#c98500;--note-bg:#1f2b3a;--zebra:#262624;--plot-bg:#f4f4f2}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#222220;--ink:#f0efec;--ink2:#c3c2b7;
--line:#3a3a37;--accent:#3987e5;--bar:#3987e5;--warn-bg:#3a2f12;--warn-line:#c98500;--note-bg:#1f2b3a;
--zebra:#262624;--plot-bg:#f4f4f2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-bottom:6px;
border-bottom:1px solid var(--line)}h3{font-size:16px;margin:18px 0 6px}
p,li{color:var(--ink)}.muted{color:var(--ink2)}code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
background:var(--zebra);padding:1px 4px;border-radius:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .v{font-size:26px;font-weight:650}.tile .k{color:var(--ink2);font-size:13px}
.callout{background:var(--warn-bg);border-left:4px solid var(--warn-line);border-radius:6px;padding:10px 14px;
margin:12px 0}.note{background:var(--note-bg);border-left:4px solid var(--accent);border-radius:6px;
padding:10px 14px;margin:12px 0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}
td.wrap{white-space:normal;min-width:180px}
th{color:var(--ink2);font-weight:600;background:var(--card);position:sticky;top:0}
tr:nth-child(even) td{background:var(--zebra)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.funnel{display:grid;grid-template-columns:minmax(150px,300px) 1fr;gap:6px 12px;align-items:center}
.funnel .lab{font-size:14px}.funnel .track{position:relative;height:22px}
.funnel .bar{height:22px;background:var(--bar);border-radius:0 4px 4px 0;min-width:2px}
.funnel .cnt{font-size:13px;color:var(--ink2);margin-left:8px;white-space:nowrap}
.funnel .row{display:flex;align-items:center}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
figure img{width:100%;height:auto;background:var(--plot-bg);border-radius:6px}
figcaption{font-size:13px;color:var(--ink2);margin-top:6px}
.flag{display:inline-block;font-size:12px;padding:0 6px;border-radius:9px;border:1px solid var(--warn-line);
margin-right:3px}
footer{margin-top:40px;color:var(--ink2);font-size:13px}
"""


def _render_html(r, qc, ranked):
    e = html.escape
    c = r["counts"]
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>VALDO report</title>", f"<style>{_CSS}</style></head><body><main>"]
    parts.append("<h1>VALDO report</h1>")
    parts.append(f"<div class='muted'>Work folder <code>{e(r['workdir'])}</code> &middot; generated "
                 f"{e(r['generated'])} &middot; valdo {e(str(r['valdo_version']))}</div>")
    tiles = []
    if c.get("datasets") is not None:
        tiles.append((c["datasets"], "datasets found"))
        tiles.append((c.get("datasets_ok"), "datasets analysed"))
    tiles.append((c["blobs_kept"], "candidate blobs"))
    tiles.append((c["datasets_with_blobs"], "datasets with a blob"))
    if r["top_hits"]:
        tiles.append((_fmt(r["top_hits"][0]["score"], 1), f"best blob {r['rank_by']}"))
    parts.append("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='v'>{e(str(v if v is not None else '-'))}</div>"
        f"<div class='k'>{e(k)}</div></div>" for v, k in tiles) + "</div>")
    if r["warnings"]:
        parts.append("<div class='callout'><b>Warnings</b><ul>" + "".join(
            f"<li>{e(w)}</li>" for w in r["warnings"]) + "</ul></div>")
    parts.append("<div class='note'><b>Read this first.</b><ul>" + "".join(
        f"<li>{e(cv)}</li>" for cv in CAVEATS) + "</ul></div>")
    if r.get("refinement"):
        parts.append("<h2>Refinement</h2><ul>" + "".join(
            f"<li>{e(line)}</li>" for line in _refinement_lines(r["refinement"])) + "</ul>")

    if r["funnel"]:
        n0 = max(f["count"] for f in r["funnel"]) or 1
        parts.append("<h2>Dataset funnel</h2><p class='muted'>How many datasets reached each stage. "
                     "datasets.csv says which ones were dropped, where and why.</p><div class='funnel'>")
        for f in r["funnel"]:
            width = 100.0 * f["count"] / n0
            drop = f" (&minus;{f['dropped']})" if f.get("dropped") else ""
            parts.append(f"<div class='lab'>{e(f['label'])}</div><div class='row'>"
                         f"<div class='bar' style='width:{width:.1f}%' title='{e(f['label'])}: {f['count']}'>"
                         f"</div><span class='cnt'>{f['count']}{drop}</span></div>")
        parts.append("</div>")

    parts.append(f"<h2>Top hits</h2><p class='muted'>{e(_top_hits_text())} "
                 f"Ranked by <code>{e(r['rank_by'])}</code>.</p>")
    if r.get("top_hits_symmetry_copies_left_out"):
        parts.append(f"<p class='muted'>{r['top_hits_symmetry_copies_left_out']} lower-ranked symmetry copies of "
                     "blobs listed here are left out (column <code>symmetry_copy_of</code> in "
                     "<code>blobs.csv</code>), so ranks can skip numbers.</p>")
    if r["top_hits"]:
        known = bool(r.get("validation"))
        head = ["Rank", "ID", "Original name", "Blob", "Score", "Peak z", "Volume (&Aring;&sup3;)",
                "Nearest residue", "Dist (&Aring;)", "Diff. max / min (&sigma;)", "Sign"] + \
            (["Known ligand"] if known else []) + ["Flags", "Maps"]
        num = {0, 4, 5, 6, 8, 9}
        rows = []
        for h in r["top_hits"]:
            flags = "".join(f"<span class='flag'>{e(f)}</span>" for f in str(h.get("flags", "")).split(";") if f)
            maps_cell = " ".join(f"<a href='{e(p)}'>{e(lbl)}</a>" for lbl, p in
                                 (("diff", h.get("diff_map")), ("2fofc", h.get("context_map")),
                                  ("model", h.get("model"))) if p)
            cells = [h["rank"], e(h["id"]), e(str(h["original_name"])), e(str(h.get("blob", h["sample"]))),
                     _fmt(h["score"], 1), _fmt(h["peakz"]), _fmt(h["volume"], 1), e(str(h["nearest_residue"])),
                     _fmt(h["nearest_distance_A"]), e(_extremes(h)), e(str(h.get("diff_sign", "")))] + \
                ([_yes_no(h.get("on_known_ligand"))] if known else []) + [flags, maps_cell]
            rows.append("<tr>" + "".join(
                f"<td class='num'>{v}</td>" if i in num else f"<td>{v}</td>" for i, v in enumerate(cells)) + "</tr>")
        parts.append("<div class='scroll'><table><thead><tr>" + "".join(
            f"<th class='num'>{h}</th>" if i in num else f"<th>{h}</th>" for i, h in enumerate(head))
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
        parts.append(
            "<h3>Look at the hits</h3><p>From the <code>report</code> folder (the scripts use paths relative "
            "to it):</p><ul>"
            "<li>Coot: <code>coot --script view_top_hits.py</code>, then <code>valdo_next()</code>, "
            "<code>valdo_prev()</code> or <code>valdo_go(5)</code> in the Python console "
            "(Calculate &rsaquo; Scripting &rsaquo; Python).</li>"
            "<li>PyMOL: <code>pymol view_top_hits.pml</code>; each hit is a scene, step with "
            "PageDown/PageUp.</li></ul>"
            f"<p class='muted'>Maps are boxed ({BOX_RADIUS_A:g} &Aring; around the blob) and in sigma units of the "
            f"whole cell: difference map at &plusmn;{DIFF_LEVEL_SIGMA:g}&sigma; (green +, red &minus;), context "
            f"2mFo-DFc-style map at {CONTEXT_LEVEL_SIGMA:g}&sigma; when the phase source has such "
            "coefficients. Map centres are moved to the symmetry copy of the blob next to the model.</p>")
    else:
        parts.append("<p>No candidate blobs.</p>")

    v = r.get("validation")
    if v:
        parts.append("<h2>Known bound models <span class='muted'>(descriptive only)</span></h2>")
        parts.append(f"<p class='muted'>{e(v['description'])}</p>")
        auc = (f" ROC AUC of the blob score, ligand-overlapping blobs vs all other kept blobs: "
               f"<b>{v['auc_score']}</b>." if v.get("auc_score") is not None else "")
        parts.append(f"<p>{v['n_with_ligand_blob']} of {v['n_analysed']} analysed bound datasets have a kept "
                     f"blob on the ligand; in {v['n_top1']} it is the dataset's top blob.{auc} Ligand residue "
                     f"names: <code>{e(str(v['ligand_resnames']))}</code>.</p>")
        head = ["ID", "Original name", "Ligand", "Blobs", "First ligand blob (in dataset)",
                "First ligand blob (global rank)", "Closest blob (&Aring;)", "Note"]
        rows = []
        for d in v["per_dataset"]:
            cells = [e(d["id"]), e(str(d["original_name"])), e(d["ligand_residues"]), d["n_blobs"],
                     "" if d["first_ligand_blob_rank_in_dataset"] is None else d["first_ligand_blob_rank_in_dataset"],
                     "" if d["first_ligand_blob_global_rank"] is None else d["first_ligand_blob_global_rank"],
                     _fmt(d["closest_blob_distance_A"]),
                     e(d["note"] or ("" if d["analysed"] else "not analysed (dropped)"))]
            rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
        parts.append("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head)
                     + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    if qc:
        parts.append("<h2>Quality-control plots</h2><div class='plots'>")
        for q in qc:
            parts.append(f"<figure><img alt='{e(q['title'])}' src='data:image/png;base64,{q['data']}'>"
                         f"<figcaption><b>{e(q['title'])}.</b> {e(q['caption'])} "
                         f"<span class='muted'>({e(os.path.basename(q['path']))})</span></figcaption></figure>")
        parts.append("</div>")

    if r["dropped"]:
        parts.append("<h2>Dropped datasets</h2><div class='scroll'><table><thead><tr><th>Stage</th>"
                     "<th>Reason</th><th class='num'>Count</th><th>Examples</th></tr></thead><tbody>")
        for d in r["dropped"]:
            parts.append(f"<tr><td>{e(str(d['stage']))}</td><td class='wrap'>{e(str(d['reason']))}</td>"
                         f"<td class='num'>{d['count']}</td><td class='wrap'>{e(', '.join(d['examples']))}</td></tr>")
        parts.append("</tbody></table></div>")

    if r["stages"]:
        parts.append("<h2>Stages</h2><div class='scroll'><table><thead><tr><th>Stage</th><th>Status</th>"
                     "<th class='num'>Time</th><th class='num'>In</th><th class='num'>Out</th></tr></thead><tbody>")
        for s in r["stages"]:
            parts.append(f"<tr><td>{e(str(s['stage']))}</td><td>{e(str(s['status'] or ''))}</td>"
                         f"<td class='num'>{_fmt_secs(s['seconds'])}</td>"
                         f"<td class='num'>{'' if s['n_in'] is None else s['n_in']}</td>"
                         f"<td class='num'>{'' if s['n_out'] is None else s['n_out']}</td></tr>")
        parts.append("</tbody></table></div>")

    if r["settings"]:
        parts.append("<h2>Key settings</h2><div class='scroll'><table><thead><tr><th>Stage</th><th>Key</th>"
                     "<th>Value</th></tr></thead><tbody>")
        for s in r["settings"]:
            parts.append(f"<tr><td>{e(s['stage'])}</td><td><code>{e(s['key'])}</code></td>"
                         f"<td>{e(str(s['value']))}</td></tr>")
        parts.append("</tbody></table></div><p class='muted'>All settings: <code>configs/&lt;stage&gt;.yaml"
                     "</code> (each stage can be rerun with <code>valdo.pipeline &lt;stage&gt; "
                     "configs/&lt;stage&gt;.yaml</code>).</p>")

    parts.append("<h2>Files</h2><ul>" + "".join(
        f"<li><b>{e(k)}</b>: <a href='{e(os.path.relpath(p, r['report_dir']))}'>"
        f"{e(os.path.relpath(p, r['report_dir']))}</a></li>" for k, p in r["files"].items()) + "</ul>")
    parts.append("<footer>VALDO (Hekstra lab) with the valdo-easy layer. A blob is a lead, not a result.</footer>")
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _valdo_version():
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "VERSION")) as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


def _write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)
