"""The stage table of `valdo run`: generated upstream configs, pre-hooks and post-conditions.

Each stage is described by a `Stage`: when it applies, the config dict it gives the
unchanged upstream stage function (`valdo.commandline.pipeline.STAGE_REGISTRY`, or agent
B's refine/report functions), a pre-hook that prepares inputs (file lists, symlink farms,
phase files), and a post-hook that checks the outputs and records what happened to every
dataset (drops with a reason, flags, values for datasets.csv). The runner (runner.py) does
the bookkeeping: fingerprints, cleaning, logs, state.

Hooks receive a context `ctx` (runner.Context) with: project, inputs (check.Inputs), workdir,
ncpu, state, path(rel), dir(rel), alive(), outputs(stage), reindexed, ops, ops_hkl,
reference_id.
"""

import copy
import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from valdo.easy import scan
from valdo.easy.ids import parse_stem

F, SIGF = "F-obs", "SIGF-obs"
F_SCALED, SIGF_SCALED = "F-obs-scaled", "SIGF-obs-scaled"
PHASE_OUT_2FOFC, PHASE_OUT_FOFC = "PH2FOFCWT", "PHFOFCWT"
IDENTITY = "x,y,z"
LOW_COVERAGE = 0.9
# Model-map check (phases from the data files or the user's refined files): mean 2mFo-DFc density
# (sigma units) at the model's protein atoms. On 293 PTP1B datasets the 286 own models gave
# 1.6-3.8 (median 3.3); the apo model where it sat at another origin/indexing -0.03 to 0.03.
MODEL_FIT_MIN = 1.0
NO_FOCAL_SEQID = -9999
# Version of the phase/model preparation in add_phases_and_blobs (part of its fingerprint, so work
# folders prepared by an older rule are redone). 2: phases are moved with the same space-group-safe
# operator as the models (refine.normalising_variant), not with the raw twin law.
PHASE_PREP_VERSION = 2


class StageError(Exception):
    """The stage cannot finish: what went wrong, likely cause, how to fix (exit code 1)."""

    def __init__(self, what, cause="", fix=""):
        super().__init__(what)
        self.what, self.cause, self.fix = what, cause, fix


class Result:
    """What one stage run did to the datasets, plus outputs for later stages."""

    def __init__(self):
        self.dropped = {}     # id -> reason
        self.flags = {}       # id -> [flag]
        self.values = {}      # id -> {datasets.csv column: value}
        self.outputs = {}     # JSON-serialisable facts later stages and `status` use
        self.messages = []    # warnings shown on the console
        self.n_in = None
        self.n_out = None

    def drop(self, ident, reason):
        self.dropped.setdefault(ident, reason)

    def flag(self, ident, flag):
        flags = self.flags.setdefault(ident, [])
        if flag not in flags:
            flags.append(flag)

    def set(self, ident, **values):
        self.values.setdefault(ident, {}).update(values)

    def contrib(self):
        return {"dropped": self.dropped, "flags": self.flags, "values": self.values}


class Stage:
    def __init__(self, number, name, folder, upstream, runs_when, description, build,
                 pre=None, execute=None, post=None, extras=None, applicable=None,
                 clean=True, owns=()):
        self.number, self.name, self.folder = number, name, folder
        self.upstream, self.runs_when, self.description = upstream, runs_when, description
        self.build, self.pre, self.execute, self.post = build, pre, execute, post
        self.extras = extras or (lambda ctx: {})
        self.applicable = applicable or (lambda ctx: (True, ""))
        self.clean = clean            # empty the stage folder before running
        self.owns = tuple(owns)       # extra folders (inside the workdir) this stage rebuilds

    def describe(self):
        return {"name": self.name, "number": self.number, "folder": self.folder,
                "upstream": self.upstream, "runs_when": self.runs_when,
                "description": self.description}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _write_list(path, files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("".join(f"{f}\n" for f in files))


def read_list(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def _symlink(src, dest):
    if os.path.lexists(dest):
        os.unlink(dest)
    os.symlink(src, dest)


def _stat(path):
    try:
        st = os.stat(path)
        return [path, st.st_size, st.st_mtime_ns]
    except OSError:
        return [path, None, None]


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def limit_torch_threads(ncpu):
    """torch starts one thread per core of the machine; on a shared node that is far more than the
    CPUs the job was given. (OMP_NUM_THREADS, set by `valdo run`, only helps if torch is imported later.)"""
    try:
        import torch
        torch.set_num_threads(max(1, int(ncpu or 1)))
    except Exception:  # no torch: the stage itself will say so
        pass


def run_upstream(name):
    """Execute hook that calls the unchanged upstream stage function on the config."""
    def execute(ctx, cfg, res):
        from valdo.commandline.config import validate_config
        from valdo.commandline.pipeline import STAGE_REGISTRY
        conf = validate_config(name, copy.deepcopy(cfg))
        conf["_force"] = True   # the runner emptied the folder; never reuse stale outputs
        if name in ("train", "reconstruct"):
            limit_torch_threads(ctx.ncpu)
        STAGE_REGISTRY[name](conf)
    return execute


def variant_op(ctx, stem):
    """gemmi triplet of the reindexing operator that produced variant `stem`."""
    _, symop = parse_stem(stem)
    if not ctx.reindexed or symop is None:
        return IDENTITY
    return ctx.ops[int(symop)]


def _is_fallback_model(ctx, row):
    """True if the row's model is the shared apo model (assumed to be in the reference frame)."""
    flags = row.get("flags", "")
    apo = ctx.project["models"]["apo"]
    return bool(apo) and row.get("model") == apo and (
        "fallback_model" in flags or not ctx.project["models"]["per_dataset"])


def _pool_map(func, jobs, ncpu, processes=True):
    if not jobs:
        return []
    workers = max(1, min(int(ncpu or 1), len(jobs)))
    if workers == 1:
        return [func(j) for j in jobs]
    executor = ProcessPoolExecutor if processes else ThreadPoolExecutor
    with executor(max_workers=workers) as pool:
        return list(pool.map(func, jobs, chunksize=max(1, len(jobs) // (workers * 4)) if processes else 1))


# ---------------------------------------------------------------------------
# 1. standardize
# ---------------------------------------------------------------------------

def build_standardize(ctx):
    return {"source_path": ctx.dir("inputs/mtz"), "destination_path": ctx.dir("01_standardize"),
            "mtz_file_pattern": r".*/(\d+)\.mtz$", "expcolumns": [F, SIGF], "ncpu": ctx.ncpu}


def extras_standardize(ctx):
    headers = ctx.inputs.header_by_path()
    files = []
    for row in ctx.alive():
        h = headers.get(row["source_mtz"], {})
        files.append([row["id"], row["source_mtz"], h.get("size"), h.get("mtime")])
    return {"files": files, "rename": ctx.inputs.rename}


def _copy_renamed(job):
    src, dest, rename = job
    try:
        import gemmi
        mtz = gemmi.read_mtz_file(src)
        for col in mtz.columns:
            if col.label in rename:
                col.label = rename[col.label]
        mtz.write_to_file(dest)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def pre_standardize(ctx, cfg, res):
    farm = ctx.path("inputs/mtz")
    os.makedirs(farm, exist_ok=True)
    rows = ctx.alive()
    res.n_in = len(rows)
    if not ctx.inputs.rename:
        for row in rows:
            _symlink(row["source_mtz"], os.path.join(farm, f"{row['id']}.mtz"))
        return
    print(f"Copying {len(rows)} MTZ files with renamed columns {ctx.inputs.rename} into {farm}")
    jobs = [(r["source_mtz"], os.path.join(farm, f"{r['id']}.mtz"), ctx.inputs.rename) for r in rows]
    for row, err in zip(rows, _pool_map(_copy_renamed, jobs, ctx.ncpu, processes=False)):
        if err:
            res.drop(row["id"], f"could not copy the input file ({err})")


def post_standardize(ctx, cfg, res):
    rows = [r for r in ctx.alive() if r["id"] not in res.dropped]
    paths = [os.path.join(cfg["destination_path"], f"{r['id']}.mtz") for r in rows]
    headers = scan.scan_headers(paths, threads=16)
    for row, path, h in zip(rows, paths, headers):
        if not os.path.isfile(path):
            res.drop(row["id"], "standardize could not read this file (see logs/standardize.log)")
        elif not h["ok"]:
            res.drop(row["id"], f"standardized file is unreadable ({h['error']})")
        elif not h["nrefl"]:
            res.drop(row["id"], "no reflection has both F-obs and SIGF-obs")
    if ctx.reference_id in res.dropped:
        raise StageError(f"The reference dataset ({ctx.reference_id}) could not be standardized: "
                         f"{res.dropped[ctx.reference_id]}.", "Every dataset is reindexed and scaled against it.",
                         "Set 'reference' in valdo.yaml to another dataset and run again.")
    res.n_out = len(rows) - sum(1 for r in rows if r["id"] in res.dropped)


# ---------------------------------------------------------------------------
# 2. reindex
# ---------------------------------------------------------------------------

def applicable_reindex(ctx):
    if ctx.reindexed:
        return True, ctx.inputs.reindex["why"]
    return False, ctx.inputs.reindex["why"] if ctx.inputs.reindex else "reindex is off"


def build_reindex(ctx):
    adv = ctx.project["advanced"]["reindex"]
    return {"input_files": ctx.path("configs/reindex_files.txt"),
            "reference_file": os.path.join(ctx.dir("01_standardize"), f"{ctx.reference_id}.mtz"),
            "output_folder": ctx.dir("02_reindex"), "columns": [F, SIGF],
            "wcorr": adv["wcorr"], "cc_min_dif": adv["cc_min_dif"], "ncpu": ctx.ncpu}


def extras_reindex(ctx):
    return {"ops": ctx.ops}


def pre_reindex(ctx, cfg, res):
    rows = ctx.alive()
    res.n_in = len(rows)
    _write_list(cfg["input_files"], [os.path.join(ctx.dir("01_standardize"), f"{r['id']}.mtz") for r in rows])


def post_reindex(ctx, cfg, res):
    import pandas as pd
    record_path = os.path.join(cfg["output_folder"], "reindex_record.pkl")
    if not os.path.isfile(record_path):
        raise StageError("reindex wrote no reindex_record.pkl.",
                         "Upstream found no alternative indexing for the reference, although valdo expected some.",
                         "Set reindex: false in valdo.yaml and run again.")
    record = pd.read_pickle(record_path)
    cc_cols = sorted([c for c in record.columns if c.startswith("CC_symop")], key=lambda c: int(c[8:]))
    if len(cc_cols) != len(ctx.ops):
        raise StageError(f"reindex tried {len(cc_cols)} operators, valdo expected {len(ctx.ops)} ({', '.join(ctx.ops_hkl)}).",
                         "The reference lattice was analysed differently (internal inconsistency).",
                         "Report this; as a workaround set reindex: false.")
    record["file_idx"] = record["file_idx"].astype(str)
    by_id = {r["file_idx"]: r for r in record.to_dict("records")}
    best, variants, ccs = {}, {}, {}
    rows = ctx.alive()
    for row in rows:
        ident = row["id"]
        rec = by_id.get(ident)
        found = [f"{ident}_{i}" for i in range(len(ctx.ops))
                 if os.path.isfile(os.path.join(cfg["output_folder"], f"{ident}_{i}.mtz"))]
        if rec is None or not found:
            res.drop(ident, "reindex failed for this dataset (see logs/reindex.log)")
            continue
        b = int(rec["best_symop"])
        best[ident] = b
        variants[ident] = found
        ccs[ident] = [None if not math.isfinite(float(rec[c])) else round(float(rec[c]), 4) for c in cc_cols]
        if b != 0:
            res.flag(ident, "reindexed")
        if int(rec["num_duplicates"]) > 1:
            res.flag(ident, "ambiguous_indexing")
    if ctx.reference_id not in best:
        raise StageError("The reference dataset could not be reindexed.", "",
                         "Set 'reference' in valdo.yaml to another dataset and run again.")
    res.outputs = {"best_symop": best, "variants": variants, "cc": ccs}
    n_flipped = sum(1 for b in best.values() if b)
    n_amb = sum(1 for f in res.flags.values() if "ambiguous_indexing" in f)
    if n_amb:
        res.messages.append(f"{n_amb} dataset(s) have ambiguous indexing (CCs within cc_min_dif); "
                            "the best one is used after filtering")
    res.outputs["n_reindexed"] = n_flipped
    res.n_in, res.n_out = len(rows), len(rows) - len(res.dropped)


# ---------------------------------------------------------------------------
# 3. scale
# ---------------------------------------------------------------------------

def _scale_inputs(ctx):
    rows = ctx.alive()
    if ctx.reindexed:
        out = ctx.outputs("reindex")
        files = [os.path.join(ctx.dir("02_reindex"), f"{s}.mtz") for r in rows for s in out["variants"][r["id"]]]
        ref = os.path.join(ctx.dir("02_reindex"), f"{ctx.reference_id}_{out['best_symop'][ctx.reference_id]}.mtz")
    else:
        files = [os.path.join(ctx.dir("01_standardize"), f"{r['id']}.mtz") for r in rows]
        ref = os.path.join(ctx.dir("01_standardize"), f"{ctx.reference_id}.mtz")
    return rows, files, ref


def build_scale(ctx):
    _, _, ref = _scale_inputs(ctx)
    return {"file_list": ctx.path("configs/scale_files.txt"), "reference_mtz": ref, "columns": [F, SIGF],
            "output_folder": ctx.dir("03_scale"), "prefix": "",
            "when_opt": ctx.project["advanced"]["scale"]["when_opt"], "ncpu": ctx.ncpu}


def pre_scale(ctx, cfg, res):
    rows, files, _ = _scale_inputs(ctx)
    res.n_in = len(rows)
    _write_list(cfg["file_list"], files)


def _all_nan(path, column=F_SCALED):
    try:
        import gemmi
        import numpy as np
        col = gemmi.read_mtz_file(path).column_with_label(column)
        if col is None:
            return None
        return not np.isfinite(np.array(col, copy=False)).any()
    except Exception:
        return None


def post_scale(ctx, cfg, res):
    import pandas as pd
    metrics_path = os.path.join(cfg["output_folder"], "scaling_metrics.pkl")
    if not os.path.isfile(metrics_path):
        raise StageError("scale wrote no scaling_metrics.pkl.", "Scaling stopped before the end.",
                         "Read logs/scale.log.")
    metrics = pd.read_pickle(metrics_path)
    end_corr = {str(f): (None if not math.isfinite(float(c)) else round(float(c), 4))
                for f, c in zip(metrics["file"], metrics["end_corr"])}
    rows, files, _ = _scale_inputs(ctx)
    scaled = {}
    for path in files:
        out = os.path.join(cfg["output_folder"], os.path.basename(path))
        if os.path.isfile(out):
            scaled[_stem(path)] = out
    with ThreadPoolExecutor(max_workers=8) as pool:
        nan_flags = dict(zip(scaled, pool.map(_all_nan, scaled.values())))
    diverged = sorted(s for s, bad in nan_flags.items() if bad)
    by_id = {}
    for path in files:
        by_id.setdefault(_stem(path).split("_")[0], []).append(_stem(path))
    stems = {}
    for row in rows:
        ident = row["id"]
        mine = [s for s in by_id.get(ident, []) if s in scaled]
        if not mine:
            res.drop(ident, "scaling failed for this dataset (see logs/scale.log)")
            continue
        stems[ident] = mine
        if any(s in diverged for s in mine):
            res.flag(ident, "scaling_diverged")
    if diverged:
        res.messages.append(f"{len(diverged)} scaled file(s) have no finite F-obs-scaled (scaling diverged); "
                            "filter removes them")
    res.outputs = {"stems": stems, "end_corr": end_corr, "diverged": diverged}
    res.n_in, res.n_out = len(rows), len(rows) - len(res.dropped)


# ---------------------------------------------------------------------------
# 4. refine (agent B: valdo.easy.refine)
# ---------------------------------------------------------------------------

def applicable_refine(ctx):
    if ctx.project["phases"]["source"] == "refine":
        return True, f"phases.source is refine ({ctx.project['phases']['program']})"
    return False, f"phases.source is {ctx.project['phases']['source']}"


def build_refine(ctx):
    r = ctx.project["advanced"]["refine"]
    return {"program": ctx.project["phases"]["program"], "out_dir": ctx.dir("04_refine"),
            "jobs": ctx.path("configs/refine_jobs.json"), "eff": r["eff"],
            "extra_args": list(r["extra_args"]), "timeout_min": r["timeout_min"], "ncpu": ctx.ncpu}


def _refine_jobs(ctx):
    jobs, no_model = [], []
    stems = ctx.outputs("scale")["stems"]
    for row in ctx.alive():
        if not row["model"]:
            no_model.append(row["id"])
            continue
        for stem in stems.get(row["id"], []):
            op = variant_op(ctx, stem)
            transform = op != IDENTITY and not _is_fallback_model(ctx, row)
            pdb = ctx.path(f"inputs/refine_models/{stem}.pdb") if transform else row["model"]
            _, symop = parse_stem(stem)
            jobs.append({"stem": stem, "id": row["id"], "symop": symop or "",
                         "mtz": os.path.join(ctx.dir("03_scale"), f"{stem}.mtz"), "pdb": pdb,
                         "model_source": row["model"], "op": op if transform else IDENTITY})
    return jobs, no_model


def extras_refine(ctx):
    jobs, _ = _refine_jobs(ctx)
    eff = ctx.project["advanced"]["refine"]["eff"]
    return {"models": [_stat(j["model_source"]) + [j["op"]] for j in jobs], "eff": _stat(eff) if eff else None}


def _transform_model(job):
    src, dest, op = job
    try:
        if op == IDENTITY:
            _symlink(src, dest)
            return None
        from valdo.easy.refine import reindex_model
        reindex_model(src, dest, op)
        return None
    except Exception as exc:
        try:
            _symlink(src, dest)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}"


def pre_refine(ctx, cfg, res):
    import json
    jobs, no_model = _refine_jobs(ctx)
    res.n_in = len(ctx.alive())
    for ident in no_model:
        res.drop(ident, "no model to refine (set models.apo or models.per_dataset)")
    todo = [(j["model_source"], j["pdb"], j["op"]) for j in jobs if j["op"] != IDENTITY]
    if todo:
        os.makedirs(ctx.path("inputs/refine_models"), exist_ok=True)
        for job, err in zip(todo, _pool_map(_transform_model, todo, ctx.ncpu)):
            if err:
                ident = _stem(job[1]).split("_")[0]
                res.flag(ident, "model_frame_mismatch")
                print(f"warning: could not transform {job[0]} with {job[2]}: {err}")
    with open(cfg["jobs"], "w") as handle:
        json.dump(jobs, handle, indent=1)
    ctx.scratch["refine_jobs"] = [{k: j[k] for k in ("stem", "id", "symop", "mtz", "pdb")} for j in jobs]


def execute_refine(ctx, cfg, res):
    from valdo.easy import refine
    refine.refine_datasets(ctx.scratch["refine_jobs"], cfg["program"], cfg["out_dir"], ncpu=cfg["ncpu"],
                           eff=cfg["eff"], extra_args=cfg["extra_args"], timeout_min=cfg["timeout_min"],
                           log=lambda text: print(text, flush=True))


def post_refine(ctx, cfg, res):
    import pandas as pd
    path = os.path.join(cfg["out_dir"], "refine_summary.csv")
    if not os.path.isfile(path):
        raise StageError("refine wrote no refine_summary.csv.", "Refinement stopped before the end.",
                         "Read logs/refine.log.")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rfree, status, reason = {}, {}, {}
    for rec in table.to_dict("records"):
        stem = rec["stem"]
        status[stem] = rec.get("status", "")
        reason[stem] = rec.get("reason", "") or rec.get("status", "")
        try:
            value = float(rec.get("Rf_final", ""))
            rfree[stem] = value if math.isfinite(value) else None
        except ValueError:
            rfree[stem] = None
    for row in ctx.alive():
        if row["id"] in res.dropped:
            continue
        stems = ctx.outputs("scale")["stems"].get(row["id"], [])
        ok = [s for s in stems if status.get(s) == "ok" and rfree.get(s) is not None]
        if not ok:
            why = "; ".join(sorted({reason.get(s, "not run") for s in stems})) or "not run"
            res.drop(row["id"], f"refinement failed ({why}; see 04_refine/runs/)")
    res.outputs = {"rfree": rfree, "status": status}
    res.n_out = res.n_in - len(res.dropped)


# ---------------------------------------------------------------------------
# 5. filter
# ---------------------------------------------------------------------------

def build_filter(ctx):
    f = ctx.project["advanced"]["filter"]
    return {"refine_summary": ctx.path("configs/filter_input.csv"), "reindex_record": None,
            "scaled_dir": ctx.dir("03_scale"),
            "metrics": os.path.join(ctx.dir("03_scale"), "scaling_metrics.pkl"),
            "max_rfree": f["max_rfree"], "min_cc": f["min_cc"],
            "output": os.path.join(ctx.dir("05_filter"), "kept_files.txt")}


def extras_filter(ctx):
    extras = {"rule": "valdo-easy/1: one variant per dataset (lowest refined R-free, else best reindex CC)"}
    gone = _stems_dropped_after_scale(ctx)
    if gone:   # only when datasets left the pipeline between scale and filter (refine)
        extras["excluded"] = gone
    return extras


def _stems_dropped_after_scale(ctx):
    """Scaled variants (03_scale/<stem>.mtz) of datasets that left the pipeline after the scale
    stage: refinement failed for every variant, or there was no model to refine. Upstream filter
    keeps every scaled file that has no row in its refinement summary, so each of them needs an
    explicit drop row (otherwise they would be used to train the VAE)."""
    alive = {r["id"] for r in ctx.alive()}
    stems = ctx.outputs("scale").get("stems") or {}
    return sorted(s for ident, mine in stems.items() if ident not in alive for s in mine)


def _filter_choice(ctx):
    """{id: (chosen stem, {stem: (Rf_final, Rf_refined, note)})} for every alive dataset."""
    stems = ctx.outputs("scale")["stems"]
    refine_out = ctx.outputs("refine") if ctx.stage_applicable("refine") else None
    best_symop = ctx.outputs("reindex")["best_symop"] if ctx.reindexed else {}
    choice = {}
    for row in ctx.alive():
        ident = row["id"]
        mine = stems.get(ident, [])
        if not mine:
            continue
        table = {}
        if refine_out is not None:
            rf = {s: refine_out["rfree"].get(s) for s in mine
                  if refine_out["status"].get(s) == "ok" and refine_out["rfree"].get(s) is not None}
            chosen = min(rf, key=lambda s: (rf[s], s)) if rf else None
            for s in mine:
                if s == chosen:
                    table[s] = (rf[s], rf[s], "lowest refined R-free")
                elif s in rf:
                    table[s] = (99.0, rf[s], f"higher R-free than {chosen}")
                else:
                    table[s] = (99.0, "", "refinement failed")
        else:
            wanted = f"{ident}_{best_symop[ident]}" if ctx.reindexed else ident
            chosen = wanted if wanted in mine else None
            if chosen is None:  # best variant was not scaled: take the next best by CC
                cc = ctx.outputs("reindex").get("cc", {}).get(ident, [])
                ranked = sorted(mine, key=lambda s: -(cc[int(parse_stem(s)[1])] or -2) if cc else 0)
                chosen = ranked[0]
            for s in mine:
                table[s] = (0.0, "", "best indexing (CC to reference)") if s == chosen else \
                    (99.0, "", "alternative indexing (lower CC to reference)")
        choice[ident] = (chosen, table)
    return choice


def pre_filter(ctx, cfg, res):
    import csv
    choice = _filter_choice(ctx)
    res.n_in = len(ctx.alive())
    os.makedirs(os.path.dirname(cfg["refine_summary"]), exist_ok=True)
    with open(cfg["refine_summary"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_idx", "symop", "Rf_final", "Rf_refined", "selected", "note"])
        for ident, (chosen, table) in choice.items():
            for stem, (rf_final, rf_refined, note) in sorted(table.items()):
                _, symop = parse_stem(stem)
                writer.writerow([ident, symop if symop is not None else "", rf_final, rf_refined,
                                 1 if stem == chosen else 0, note])
        # Every other scaled file belongs to a dataset that is no longer in the pipeline (e.g. its
        # refinement failed): give it a row that upstream's R-free rule drops.
        covered = {s for _, table in choice.values() for s in table}
        on_disk = {_stem(p) for p in glob.glob(os.path.join(cfg["scaled_dir"], "*.mtz"))}
        why = {r["id"]: f"{r['dropped_stage']}: {r['dropped_reason']}" for r in ctx.rows() if r["status"] != "ok"}
        for stem in sorted((set(_stems_dropped_after_scale(ctx)) | on_disk) - covered):
            ident, symop = parse_stem(stem)
            if ident is None:
                continue
            writer.writerow([ident, symop if symop is not None else "", 99.0, "", 0,
                             f"dataset no longer in the pipeline ({why.get(ident, 'dropped')})"])
    ctx.scratch["filter_choice"] = choice


def post_filter(ctx, cfg, res):
    if not os.path.isfile(cfg["output"]):
        raise StageError("filter wrote no list of kept files.", "", "Read logs/filter.log.")
    kept = {_stem(p) for p in read_list(cfg["output"])}
    scale_out = ctx.outputs("scale")
    f = ctx.project["advanced"]["filter"]
    choice = ctx.scratch.get("filter_choice") or _filter_choice(ctx)
    kept_map = {}
    for row in ctx.alive():
        ident = row["id"]
        mine = scale_out["stems"].get(ident, [])
        chosen, table = choice.get(ident, (None, {}))
        survivors = [s for s in mine if s in kept]
        if survivors:
            stem = chosen if chosen in survivors else survivors[0]
            kept_map[ident] = stem
            res.set(ident, variant=stem,
                    reindex_op=ctx.ops_hkl[int(parse_stem(stem)[1])] if ctx.reindexed else "")
            continue
        if chosen is None:
            res.drop(ident, "refinement failed for every indexing variant")
            continue
        rf = table[chosen][0]
        cc = scale_out["end_corr"].get(chosen)
        if ctx.stage_applicable("refine") and rf > f["max_rfree"]:
            res.drop(ident, f"R-free {rf:.3f} after refinement is above max_rfree {f['max_rfree']}")
        elif chosen in scale_out["diverged"]:
            res.drop(ident, "scaling diverged (no finite F-obs-scaled)")
        elif f["min_cc"] > 0 and (cc is None or cc < f["min_cc"]):
            res.drop(ident, f"CC to the reference after scaling {('n/a' if cc is None else f'{cc:.3f}')} "
                            f"is below min_cc {f['min_cc']}")
        else:
            res.drop(ident, "removed by filter (see logs/filter.log)")
    stray = sorted(kept - set(kept_map.values()))
    if stray:   # preprocess, train and rescale read kept_files.txt: nothing else may be in it
        raise StageError(f"filter kept {len(stray)} file(s) that are not the chosen variant of a dataset in the "
                         f"pipeline: {', '.join(stray[:8])}.",
                         "An internal inconsistency in valdo-easy (they would be used to train the VAE).",
                         "Please report it with logs/filter.log and configs/filter_input.csv.")
    res.outputs = {"kept": kept_map}
    res.n_out = res.n_in - len(res.dropped)


# ---------------------------------------------------------------------------
# 6. preprocess
# ---------------------------------------------------------------------------

def build_preprocess(ctx):
    d = ctx.dir("06_preprocess")
    return {"file_list": os.path.join(ctx.dir("05_filter"), "kept_files.txt"), "output_folder": d,
            "intersection_path": d + "intersection.pkl", "union_path": d + "union.pkl",
            "sigF_path": d + "sigF.pkl", "amplitude_col": F_SCALED, "error_col": SIGF_SCALED,
            "include_errors": True, "prefix": ""}


def pre_count(ctx, cfg, res):
    res.n_in = len(ctx.alive())


def post_preprocess(ctx, cfg, res):
    import numpy as np
    import pandas as pd
    d = cfg["output_folder"]
    for name in ("vae_input.npy", "vae_output.npy", "vae_sigF.npy"):
        if not os.path.isfile(os.path.join(d, name)):
            raise StageError(f"preprocess did not write {name}.", "", "Read logs/preprocess.log.")
    inter = pd.read_pickle(cfg["intersection_path"])
    union = pd.read_pickle(cfg["union_path"])
    x = np.load(os.path.join(d, "vae_input.npy"), mmap_mode="r")
    y = np.load(os.path.join(d, "vae_output.npy"), mmap_mode="r")
    names = [str(c) for c in inter.columns]
    union_names = [str(c)[2:] for c in union.columns]
    if names != union_names or x.shape[0] != len(names) or y.shape[0] != len(names):
        raise StageError("The VAE input and output rows do not belong to the same datasets "
                         f"(intersection {len(names)}, union {len(union_names)}, input {x.shape[0]}, output {y.shape[0]}).",
                         "A file could be read by one preprocessing step but not by the other.",
                         "Run 'valdo check --deep', remove unreadable files from 'data', and run again.")
    if x.shape[1] == 0:
        raise StageError("No reflection is common to all datasets (empty intersection).",
                         "A dataset with very low resolution or completeness, or one indexed differently, "
                         "leaves no common reflections.",
                         "Look at d_min in datasets.csv and remove outliers from 'data'.")
    bad_rows = [names[i] for i in np.where(~np.isfinite(np.asarray(x)).all(axis=1))[0]]
    if bad_rows:
        raise StageError(f"{len(bad_rows)} dataset(s) have missing scaled amplitudes in the VAE input: "
                         f"{', '.join(bad_rows[:8])}.", "Partly diverged scaling.",
                         "Remove these datasets from 'data' and run again.")
    present = {_stem(n) for n in names}
    stray = sorted(present - {row["variant"] for row in ctx.alive()})
    if stray:
        raise StageError(f"The VAE input contains {len(stray)} dataset(s) that are not in the pipeline: "
                         f"{', '.join(stray[:8])}.",
                         "An internal inconsistency in valdo-easy (05_filter/kept_files.txt lists files of "
                         "datasets that were dropped).",
                         "Please report it with logs/filter.log and logs/preprocess.log.")
    for row in ctx.alive():
        if row["variant"] not in present:
            res.drop(row["id"], "could not be read by preprocess (see logs/preprocess.log)")
    res.outputs = {"n_samples": int(x.shape[0]), "n_intersection": int(x.shape[1]),
                   "n_union": int(y.shape[1]), "samples": [_stem(n) for n in names]}
    if x.shape[1] < 0.1 * y.shape[1]:
        res.messages.append(f"{x.shape[1]} of {y.shape[1]} reflections are common to all datasets (the VAE input). "
                            "The lowest-resolution or least complete datasets set this number; it is normal for "
                            "screens with a range of resolutions (the validated PTP1B run used 2,775 of 77,821)")
    res.n_out = res.n_in - len(res.dropped)


# ---------------------------------------------------------------------------
# 7. train, 8. reconstruct
# ---------------------------------------------------------------------------

def build_train(ctx):
    d = ctx.dir("06_preprocess")
    t = ctx.project["advanced"]["train"]
    cfg = {"vae_input_path": d + "vae_input.npy", "vae_output_path": d + "vae_output.npy",
           "vae_sigF_path": d + "vae_sigF.npy", "output_path": ctx.dir("07_train") + "trained_vae.pkl"}
    for key in ("latent_dim", "n_hidden_layers", "n_hidden_size", "epochs", "train_fraction", "batch_size",
                "learning_rate", "w_kl", "eps", "stdof", "include_errors", "random_seed", "activation"):
        cfg[key] = copy.deepcopy(t[key])
    return cfg


def post_train(ctx, cfg, res):
    import pickle
    if not os.path.isfile(cfg["output_path"]):
        raise StageError("train did not write the model.", "", "Read logs/train.log.")
    try:
        with open(cfg["output_path"], "rb") as handle:
            losses = pickle.load(handle).get("loss_train") or []
        last = [float(v) for v in losses[-1]] if losses else []
    except Exception as exc:  # e.g. a GPU-saved model on a CPU-only node
        res.messages.append(f"could not read the training losses back ({type(exc).__name__})")
        last = []
    if last and not all(math.isfinite(v) for v in last):
        raise StageError("VAE training ended with NaN/infinite losses.",
                         "A dataset with extreme values, or a learning rate that is too high.",
                         "Lower advanced.train.learning_rate (e.g. 0.0005) and run 'valdo run --from train'.")
    res.outputs = {"final_loss": last}
    res.n_out = res.n_in


def build_reconstruct(ctx):
    r = ctx.project["advanced"]["reconstruct"]
    return {"vae_path": ctx.dir("07_train") + "trained_vae.pkl",
            "vae_input_path": ctx.dir("06_preprocess") + "vae_input.npy",
            "output_path": ctx.dir("08_reconstruct") + "recons.npy",
            "ml_recon": r["ml_recon"], "repeats": r["repeats"]}


def post_reconstruct(ctx, cfg, res):
    import numpy as np
    if not os.path.isfile(cfg["output_path"]):
        raise StageError("reconstruct did not write recons.npy.", "", "Read logs/reconstruct.log.")
    recons = np.load(cfg["output_path"], mmap_mode="r")
    n = ctx.outputs("preprocess")["n_samples"]
    rows = recons.shape[1] if cfg["repeats"] > 1 else recons.shape[0]
    if rows != n:
        raise StageError(f"reconstruct returned {rows} rows for {n} datasets.", "",
                         "Run 'valdo run --from train'.")
    if not np.isfinite(np.asarray(recons)).all():
        raise StageError("The VAE reconstructions contain NaN.", "Training diverged.",
                         "Lower advanced.train.learning_rate and run 'valdo run --from train'.")
    res.n_out = res.n_in


# ---------------------------------------------------------------------------
# 9. rescale
# ---------------------------------------------------------------------------

def build_rescale(ctx):
    p, r = ctx.dir("06_preprocess"), ctx.dir("08_reconstruct")
    return {"recons_path": r + "recons.npy", "intersection_path": p + "intersection.pkl",
            "union_path": p + "union.pkl", "input_files": os.path.join(ctx.dir("05_filter"), "kept_files.txt"),
            "info_folder": p, "output_folder": ctx.dir("09_rescale"), "amplitude_col": F_SCALED,
            "ncpu": ctx.ncpu}


def _nonfinite(path):
    try:
        import gemmi
        import numpy as np
        mtz = gemmi.read_mtz_file(path)
        bad = 0
        for label in ("diff", SIGF_SCALED):
            col = mtz.column_with_label(label)
            if col is None:
                return f"no {label} column"
            bad += int((~np.isfinite(np.array(col, copy=False))).sum())
        return bad
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def post_rescale(ctx, cfg, res):
    rows = ctx.alive()
    paths = [os.path.join(cfg["output_folder"], f"{r['variant']}.mtz") for r in rows]
    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(lambda p: _nonfinite(p) if os.path.isfile(p) else "missing", paths))
    for row, check in zip(rows, checks):
        if check == "missing":
            res.drop(row["id"], "rescale wrote no file for this dataset (see logs/rescale.log)")
        elif isinstance(check, str):
            res.drop(row["id"], f"rescaled file unreadable ({check})")
        elif check:
            res.drop(row["id"], f"{check} non-finite difference amplitudes or sigmas "
                                "(the weighting step would empty this dataset)")
    res.n_out = res.n_in - len(res.dropped)


# ---------------------------------------------------------------------------
# 10. add_phases_and_blobs
# ---------------------------------------------------------------------------

def _phase_columns(ctx):
    """(columns to copy into the phase files, phwt, phdelwt, rfree label) for the configured source."""
    p = ctx.project["phases"]
    if p["source"] == "refine":
        from valdo.easy.refine import phase_columns
        phwt, phdelwt = phase_columns(p["program"])
        return None, phwt, phdelwt, None
    c = p["columns"]
    cols = [x for x in (c["fwt"], c["phwt"], c["delfwt"], c["phdelwt"], p["rfree"]) if x]
    return cols, c["phwt"], c["phdelwt"], p["rfree"]


def _phase_dirs(ctx):
    if ctx.project["phases"]["source"] == "refine":
        return ctx.dir("04_refine/models"), ctx.dir("04_refine/phases")
    return ctx.dir("inputs/models"), ctx.dir("inputs/phases")


def build_phases(ctx):
    b = ctx.project["advanced"]["blobs"]
    _, phwt, phdelwt, rfree = _phase_columns(ctx)
    models, phases = _phase_dirs(ctx)
    out = ctx.dir("10_phases_blobs")
    return {"file_list": ctx.path("configs/add_phases_and_blobs_files.txt"), "phasing_path": phases,
            "output_folder": out + "mtz/", "blob_output_folder": out + "blobs/", "model_folder": models,
            "phase_2FOFC_col_in": phwt, "phase_FOFC_col_in": phdelwt,
            "phase_2FOFC_col_out": PHASE_OUT_2FOFC, "phase_FOFC_col_out": PHASE_OUT_FOFC,
            "rfree_label_in": rfree, "sigF_col": SIGF_SCALED, "diff_col": "diff",
            "sigdF_pct": b["sigdF_pct"], "absdF_pct": b["absdF_pct"], "F_col": F_SCALED, "recons_col": "recons",
            "extrapolate_factors": list(b["extrapolate_factors"]), "blob_diff_col": "WDF",
            "phase_col": PHASE_OUT_2FOFC, "cutoff": b["cutoff"], "radius_in_A": b["radius_in_A"],
            "prefix": "", "ncpu": ctx.ncpu}


def _phase_jobs(ctx, cfg):
    cols, phwt, phdelwt, _ = _phase_columns(ctx)
    refine = ctx.project["phases"]["source"] == "refine"
    jobs, missing = [], []
    for row in ctx.alive():
        stem = row["variant"]
        op = variant_op(ctx, stem)
        job = {"id": row["id"], "stem": stem, "phwt": phwt, "phdelwt": phdelwt,
               "data": os.path.join(ctx.dir("09_rescale"), f"{stem}.mtz")}
        if refine:
            job.update(model_src=None, model_dest=None, model_op=IDENTITY, phase_src=None, phase_dest=None,
                       phase_ready=os.path.join(cfg["phasing_path"], f"{stem}.mtz"),
                       model_ready=os.path.join(cfg["model_folder"], f"{stem}.pdb"))
        else:
            if not row["model"]:
                missing.append((row["id"], "no model for blob detection (set models.apo or models.per_dataset)"))
                continue
            if not row["phase_file"]:
                missing.append((row["id"], "no phases for this dataset (no map coefficients in its file, "
                                           "or no refined MTZ matched)"))
                continue
            model_op = IDENTITY if _is_fallback_model(ctx, row) else op
            job.update(model_src=row["model"], model_dest=os.path.join(cfg["model_folder"], f"{stem}.pdb"),
                       model_op=model_op, phase_src=row["phase_file"],
                       phase_dest=os.path.join(cfg["phasing_path"], f"{stem}.mtz"),
                       columns=cols, op=op, transform=ctx.reindexed, phase_ready=None, model_ready=None,
                       fwt=ctx.project["phases"]["columns"]["fwt"])
        jobs.append(job)
    return jobs, missing


def extras_phases(ctx):
    cfg = build_phases(ctx)
    jobs, missing = _phase_jobs(ctx, cfg)
    cols, _, _, _ = _phase_columns(ctx)
    c = ctx.project["phases"]["columns"]
    return {"models": [_stat(j["model_src"]) + [j["model_op"]] for j in jobs if j.get("model_src")],
            "phases": [_stat(j["phase_src"]) + [j.get("op")] for j in jobs if j.get("phase_src")],
            "missing": missing,
            # copied into inputs/phases/ and used for the model-map check and the report's context maps
            "phase_columns": {"copied": cols, "fwt": c["fwt"], "delfwt": c["delfwt"]},
            "prep": PHASE_PREP_VERSION}


def model_map_fit(mtz_path, f_col, phi_col, pdb_path):
    """Mean 2mFo-DFc density (sigma units of the whole cell) at the model's protein atoms
    (no waters, HETATM or hydrogens): ~2-4 when the model is the one the phases came from,
    ~0 when it sits elsewhere (another origin, symmetry setting or indexing). None if not computable."""
    import gemmi
    import numpy as np
    mtz = gemmi.read_mtz_file(mtz_path)
    if mtz.column_with_label(f_col) is None or mtz.column_with_label(phi_col) is None:
        return None
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=3)
    grid.normalize()
    st = gemmi.read_structure(pdb_path)
    vals = [grid.interpolate_value(atom.pos) for chain in st[0] for res in chain
            if not res.is_water() and res.het_flag != "H" for atom in res if atom.element.name != "H"]
    return round(float(np.mean(vals)), 3) if vals else None


def phase_operator(op, spacegroup):
    """The operator that brings a phase file (and the model it came from) into the setting that
    upstream reindex gave the amplitudes with the twin law `op`.

    Amplitudes are the same for every member of op's point-group coset, but phases and models
    must be moved by one that maps the space group onto itself (refine.normalising_variant;
    e.g. gemmi's twin law -x,y,-z for I 41 needs y,x,-z), and phases and model must be moved by
    the same one. Returns op itself when that is safe (P 31 2 1 and the other common cases), the
    safe variant otherwise, or None when every choice would need an origin shift."""
    import gemmi
    from valdo.easy.refine import normalising_variant
    op = gemmi.Op(op) if isinstance(op, str) else op
    if op.triplet() == IDENTITY or spacegroup is None:
        return op
    return normalising_variant(op, spacegroup)


def prepare_dataset(job):
    """Model symlink/transform, phase file in the variant's frame, phase coverage (worker process)."""
    out = {"id": job["id"], "stem": job["stem"], "drop": None, "flags": [], "coverage": None, "note": None,
           "model_fit": None}
    try:
        import gemmi
        import numpy as np
        import reciprocalspaceship as rs
        ds = None
        model_op = job.get("model_op")
        if job.get("phase_dest"):
            ds = rs.read_mtz(job["phase_src"])
            lacking = [c for c in (job["phwt"], job["phdelwt"]) if c not in ds.columns]
            if lacking:
                out["drop"] = f"phase file has no column(s) {', '.join(lacking)}"
                return out
            if job["transform"] and job["op"] != IDENTITY:
                safe = phase_operator(job["op"], ds.spacegroup)
                if safe is None:
                    out["drop"] = (f"reindexed with {job['op']}, but in {ds.spacegroup.hm} its phases (and model) "
                                   "can only be brought into that setting with an origin shift, which valdo-easy "
                                   "does not do; use phases.source: refine for this screen")
                    return out
                out["phase_op"] = safe.triplet()
                if model_op == job["op"]:    # the dataset's own model: moved exactly like its phases
                    model_op = safe.triplet()
        if job.get("model_dest"):
            err = _transform_model((job["model_src"], job["model_dest"], model_op))
            if err:
                out["flags"].append("model_frame_mismatch")
                out["note"] = f"model could not be transformed ({err}); using it untransformed"
        elif job.get("model_ready") and not os.path.isfile(job["model_ready"]):
            out["drop"] = "refinement wrote no model for this dataset"
            return out
        if job.get("phase_dest"):
            ds = ds[[c for c in job["columns"] if c in ds.columns]]
            if job["transform"]:
                # the reindexing upstream applied to the amplitudes; phases follow, moved by the
                # space-group-safe member of the operator's coset (the same amplitudes; see phase_operator)
                ds = ds.apply_symop(gemmi.Op(out.get("phase_op") or job["op"])).hkl_to_asu()
                if "M/ISYM" in ds.columns:
                    ds = ds.drop(columns=["M/ISYM"])
            ds = ds[~ds.index.duplicated()]
            tmp = job["phase_dest"] + ".tmp.mtz"
            ds.write_mtz(tmp)
            os.replace(tmp, job["phase_dest"])
            phased = ds
        else:
            if not os.path.isfile(job["phase_ready"]):
                out["drop"] = "refinement wrote no phases for this dataset"
                return out
            phased = rs.read_mtz(job["phase_ready"])
            if job["phwt"] not in phased.columns or job["phdelwt"] not in phased.columns:
                out["drop"] = f"refined phase file lacks {job['phwt']}/{job['phdelwt']}"
                return out
        if job.get("phase_dest") and job.get("fwt") and job.get("model_dest"):
            # does the model sit in the density of the phases it is used with? (flag only)
            try:
                fit = model_map_fit(job["phase_dest"], job["fwt"], job["phwt"], job["model_dest"])
            except Exception:
                fit = None
            out["model_fit"] = fit
            if fit is not None and fit < MODEL_FIT_MIN:
                out["flags"].append("model_map_mismatch")
        ph = phased[[job["phwt"], job["phdelwt"]]].to_numpy(dtype=float)
        good = phased.index[np.isfinite(ph).all(axis=1)]
        data = rs.read_mtz(job["data"])
        n = len(data.index)
        coverage = float(data.index.isin(good).sum()) / n if n else 0.0
        out["coverage"] = round(coverage, 4)
        if coverage == 0.0:
            out["drop"] = ("no phase matches any reflection of this dataset "
                           "(phase file from another crystal form or indexing?)")
        elif coverage < LOW_COVERAGE:
            out["flags"].append("low_phase_coverage")
    except Exception as exc:
        out["drop"] = f"could not prepare phases/model ({type(exc).__name__}: {exc})"
    return out


def pre_phases(ctx, cfg, res):
    rows = ctx.alive()
    res.n_in = len(rows)
    jobs, missing = _phase_jobs(ctx, cfg)
    for ident, reason in missing:
        res.drop(ident, reason)
    if ctx.project["phases"]["source"] != "refine":
        os.makedirs(cfg["model_folder"], exist_ok=True)
        os.makedirs(cfg["phasing_path"], exist_ok=True)
    print(f"Preparing models and phases for {len(jobs)} datasets ({cfg['model_folder']}, {cfg['phasing_path']})")
    coverage, fit = {}, {}
    for result in _pool_map(prepare_dataset, jobs, ctx.ncpu):
        if result["note"]:
            print(f"{result['stem']}: {result['note']}")
        for flag in result["flags"]:
            res.flag(result["id"], flag)
        coverage[result["stem"]] = result["coverage"]
        if result.get("model_fit") is not None:
            fit[result["stem"]] = result["model_fit"]
        if result["drop"]:
            res.drop(result["id"], result["drop"])
    keep = [r for r in rows if r["id"] not in res.dropped]
    _write_list(cfg["file_list"], [os.path.join(ctx.dir("09_rescale"), f"{r['variant']}.mtz") for r in keep])
    res.outputs["phase_coverage"] = coverage
    res.outputs["model_fit"] = fit
    bad_fit = sorted(s for s, v in fit.items() if v < MODEL_FIT_MIN)
    if bad_fit:
        for stem in bad_fit:
            print(f"{stem}: model does not fit the 2mFo-DFc map of its phases "
                  f"(mean density {fit[stem]:.2f} sigma at its atoms)")
        names = {r["id"]: r["original_name"] for r in rows}
        listed = [f"{names.get(parse_stem(s)[0], s)} ({parse_stem(s)[0] or s})" for s in bad_fit[:5]]
        more = f" and {len(bad_fit) - 5} more" if len(bad_fit) > 5 else ""
        res.messages.append(f"{len(bad_fit)} dataset(s) whose model does not sit in the density of their phases "
                            f"(mean 2mFo-DFc below {MODEL_FIT_MIN} sigma at the model's atoms; flag "
                            f"model_map_mismatch): {', '.join(listed)}{more}. Their blobs are found as usual, but "
                            "focal-residue tagging, nearest residues and viewer models are unreliable. Give these "
                            "datasets the model their phases came from")
    low = sum(1 for f in res.flags.values() if "low_phase_coverage" in f)
    if low:
        res.messages.append(f"{low} dataset(s) have phases for less than {LOW_COVERAGE:.0%} of their reflections "
                            "(flag low_phase_coverage)")
    if not keep:
        raise StageError("No dataset has both a model and phases.",
                         "; ".join(sorted(set(res.dropped.values())))[:400],
                         "Check models and phases in valdo.yaml with 'valdo check'.")


def post_phases(ctx, cfg, res):
    import pandas as pd
    stats = os.path.join(cfg["blob_output_folder"], "blob_stats.pkl")
    if not os.path.isfile(stats):
        raise StageError("add_phases_and_blobs wrote no blob_stats.pkl.", "",
                         "Read logs/add_phases_and_blobs.log.")
    blobs = pd.read_pickle(stats)
    per_sample = blobs["sample"].astype(str).value_counts().to_dict() if len(blobs) else {}
    for row in ctx.alive():
        if row["id"] in res.dropped:
            continue
        if not os.path.isfile(os.path.join(cfg["output_folder"], f"{row['variant']}.mtz")):
            res.drop(row["id"], "phases could not be added (see logs/add_phases_and_blobs.log)")
    res.outputs.update({"n_blobs": int(len(blobs)),
                        "datasets_with_blobs": int(sum(1 for v in per_sample.values() if v))})
    res.n_out = res.n_in - len(res.dropped)


# ---------------------------------------------------------------------------
# 11. tag_blobs
# ---------------------------------------------------------------------------

def _bound_jobs(ctx):
    jobs = []
    for row in ctx.alive():
        if row.get("bound_model"):
            jobs.append((row["bound_model"], ctx.path(f"inputs/bound_models/{row['id']}.pdb"),
                         variant_op(ctx, row["variant"])))
    return jobs


def build_tag(ctx):
    f = ctx.project["focal_residue"]
    models, _ = _phase_dirs(ctx)
    cfg = {"blob_stats_path": ctx.dir("10_phases_blobs") + "blobs/blob_stats.pkl", "model_folder": models,
           "mtz_folder": ctx.dir("10_phases_blobs") + "mtz/",
           "focal_seqid": f["seqid"] if f["seqid"] is not None else NO_FOCAL_SEQID,
           "focal_tag_name": f["tag"], "focal_radius": f["radius"], "output_folder": ctx.dir("11_tag"),
           "ncpu": ctx.ncpu}
    if _bound_jobs(ctx):
        cfg["bound_models_folder"] = ctx.dir("inputs/bound_models")
    return cfg


def extras_tag(ctx):
    return {"bound": [_stat(src) + [op] for src, _, op in _bound_jobs(ctx)]}


def pre_tag(ctx, cfg, res):
    res.n_in = len(ctx.alive())
    jobs = _bound_jobs(ctx)
    if jobs:
        os.makedirs(ctx.path("inputs/bound_models"), exist_ok=True)
        for (src, dest, op), err in zip(jobs, _pool_map(_transform_model, jobs, ctx.ncpu)):
            if err:
                print(f"warning: bound model {src} could not be transformed with {op}: {err}")


def post_tag(ctx, cfg, res):
    import pandas as pd
    out = cfg["output_folder"]
    for name in ("blob_stats_tagged.pkl", "filtered_blob_stats_tagged.pkl"):
        if not os.path.isfile(os.path.join(out, name)):
            raise StageError(f"tag_blobs did not write {name}.", "", "Read logs/tag_blobs.log.")
    tagged = pd.read_pickle(os.path.join(out, "blob_stats_tagged.pkl"))
    kept = pd.read_pickle(os.path.join(out, "filtered_blob_stats_tagged.pkl"))
    res.outputs = {"n_blobs": int(len(tagged)), "n_kept": int(len(kept))}
    res.n_out = res.n_in


# ---------------------------------------------------------------------------
# 12. report (agent B: valdo.easy.report)
# ---------------------------------------------------------------------------

def build_report(ctx):
    r = ctx.project["advanced"]["report"]
    bound = ctx.path("inputs/bound_models")
    return {"workdir": ctx.workdir, "top_n": r["top_n"], "write_maps": r["maps"],
            "known_bound_dir": bound if _bound_jobs(ctx) else None,
            "ligand_resnames": list(ctx.project["validation"]["ligand_resnames"] or [])}


def execute_report(ctx, cfg, res):
    from valdo.easy import report
    result = report.build_report(cfg["workdir"], top_n=cfg["top_n"], rank_by=None,
                                 known_bound_dir=cfg["known_bound_dir"],
                                 ligand_resnames=cfg["ligand_resnames"] or None, write_maps=cfg["write_maps"])
    ctx.scratch["report"] = result


def post_report(ctx, cfg, res):
    html = os.path.join(ctx.workdir, "report", "report.html")
    if not os.path.isfile(html):
        raise StageError("The report was not written.", "", "Read logs/report.log.")
    result = ctx.scratch.get("report") or {}
    res.outputs = {"report_html": html, "counts": result.get("counts"),
                   "n_top_hits": len(result.get("top_hits") or []), "warnings": result.get("warnings") or []}
    res.n_out = res.n_in


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

STAGES = [
    Stage(1, "standardize", "01_standardize", "standardize", "always",
          "Copy the data under canonical IDs (0001.mtz, ...), dropping reflections without F/SIGF.",
          build_standardize, pre_standardize, run_upstream("standardize"), post_standardize,
          extras_standardize, owns=("inputs/mtz",)),
    Stage(2, "reindex", "02_reindex", "reindex", "the lattice allows alternative indexing (reindex: auto) or reindex: true",
          "Bring every dataset into the reference's indexing (all close variants are kept until filter).",
          build_reindex, pre_reindex, run_upstream("reindex"), post_reindex, extras_reindex,
          applicable=applicable_reindex),
    Stage(3, "scale", "03_scale", "scale", "always",
          "Scale every dataset (variant) to the reference.",
          build_scale, pre_scale, run_upstream("scale"), post_scale),
    Stage(4, "refine", "04_refine", "valdo.easy.refine.refine_datasets", "phases.source: refine",
          "Refine every variant with phenix or dimple to get phases and R-free.",
          build_refine, pre_refine, execute_refine, post_refine, extras_refine,
          applicable=applicable_refine, clean=False, owns=("inputs/refine_models",)),
    Stage(5, "filter", "05_filter", "filter", "always",
          "Keep one variant per dataset; drop high R-free (refine only), diverged scaling and low CC.",
          build_filter, pre_filter, run_upstream("filter"), post_filter, extras_filter),
    Stage(6, "preprocess", "06_preprocess", "preprocess", "always",
          "Build the VAE arrays (common reflections in, all reflections out).",
          build_preprocess, pre_count, run_upstream("preprocess"), post_preprocess),
    Stage(7, "train", "07_train", "train", "always",
          "Train the VAE on all kept datasets.",
          build_train, pre_count, run_upstream("train"), post_train),
    Stage(8, "reconstruct", "08_reconstruct", "reconstruct", "always",
          "Reconstruct every dataset with the VAE.",
          build_reconstruct, pre_count, run_upstream("reconstruct"), post_reconstruct),
    Stage(9, "rescale", "09_rescale", "rescale", "always",
          "Undo the normalisation; difference amplitudes diff = F - reconstruction.",
          build_rescale, pre_count, run_upstream("rescale"), post_rescale),
    Stage(10, "add_phases_and_blobs", "10_phases_blobs", "add_phases_and_blobs", "always",
          "Add phases and weights, extrapolate, and find difference-density blobs.",
          build_phases, pre_phases, run_upstream("add_phases_and_blobs"), post_phases, extras_phases,
          owns=("inputs/models", "inputs/phases")),
    Stage(11, "tag_blobs", "11_tag", "tag_blobs", "always",
          "Tag blobs near the focal residue, ligands of known bound models and symmetry duplicates.",
          build_tag, pre_tag, run_upstream("tag_blobs"), post_tag, extras_tag,
          owns=("inputs/bound_models",)),
    Stage(12, "report", "report", "valdo.easy.report.build_report", "always",
          "Write report/report.html with the ranked hits, maps and viewer scripts.",
          build_report, pre_count, execute_report, post_report),
]

BY_NAME = {s.name: s for s in STAGES}
NAMES = [s.name for s in STAGES]
