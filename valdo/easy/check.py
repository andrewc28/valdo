"""`valdo check`: is everything in place to run VALDO, and is anything likely to go wrong?

`inspect_inputs(project)` reads the MTZ headers (fast, no reflection data), names the
datasets, assigns IDs and matches models, phase files and bound models to them; the result is
the base of datasets.csv. `run_checks(project, inputs)` turns that plus the environment and
the settings into findings: each says what is wrong, why it matters and how to fix it.
Errors block `valdo run`; warnings do not. Nothing here imports torch.
"""

import importlib
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

from valdo.easy import scan
from valdo.easy.ids import assign_ids, natural_key

MIN_DATASETS = 3          # fewer cannot run at all (rescale misreads N == 2)
FEW_DATASETS = 100        # the VAE learns the apo distribution from many datasets
LOW_RESOLUTION = 3.5      # A; coarser maps get little or no blurring in blob detection
MAX_INPUT_LOSS = 0.5      # if more than this fraction is unusable, something global is wrong
FALLBACK_WARN = 0.1       # more datasets than this on the apo fallback model: probably a wrong pattern


def finding(level, code, title, what, why="", fix="", items=None):
    """A check result. level: error | warning | info."""
    return {"level": level, "code": code, "title": title, "what": what, "why": why,
            "fix": fix, "items": list(items or [])}


class Inputs:
    """Everything known about the inputs before any stage runs."""

    def __init__(self):
        self.files = []            # absolute MTZ paths, sorted
        self.note = ""             # why the file list is empty, if it is
        self.headers = []          # scan.read_mtz_header() dicts, same order as files
        self.names = []            # dataset names, same order as files
        self.ids = {}              # name -> canonical ID
        self.rows = []             # base rows of datasets.csv, in ID order
        self.columns = {}          # detect_amplitude_columns() result
        self.phases = None         # detect_phase_columns() result
        self.rfree = None
        self.summary = {}          # scan.summarize_headers()
        self.rename = {}           # column renames needed in inputs/mtz/
        self.reference = None      # {"name", "id", "why"}
        self.reindex = None        # {"enabled", "ops" (gemmi triplets), "ops_hkl", "why"}
        self.models = {}           # name -> path (per-dataset matches)
        self.model_problems = []
        self.phase_files = {}      # name -> path (phases.source: existing)
        self.phase_problems = []
        self.bound = {}            # name -> path
        self.bound_problems = []
        self.findings = []         # input findings collected while inspecting

    def ok_rows(self):
        return [r for r in self.rows if r["status"] == "ok"]

    def by_id(self):
        return {r["id"]: r for r in self.rows}

    def header_by_path(self):
        return {h["path"]: h for h in self.headers}


def inspect_inputs(project, threads=16):
    """Scan the inputs named in a validated project. Never raises for bad data files."""
    inp = Inputs()
    base = project.get("_dir", os.getcwd())
    inp.files, inp.note = scan.list_data_files(project["data"], base)
    if not inp.files:
        inp.findings.append(finding(
            "error", "no_data", "No MTZ files found",
            f"'data' ({project['data']}) does not name any MTZ file: {inp.note}.",
            "VALDO needs one MTZ file per crystal.",
            "Set 'data' in valdo.yaml to the folder with your MTZ files, a quoted glob such as "
            "'/path/*/final.mtz', or a .txt file with one path per line."))
        return inp
    inp.headers = scan.scan_headers(inp.files, threads=threads)
    inp.names = scan.dataset_names(inp.files)
    inp.ids = assign_ids(inp.names)
    inp.columns = scan.detect_amplitude_columns(inp.headers)
    inp.phases = scan.detect_phase_columns(inp.headers)
    inp.rfree = scan.detect_rfree_column(inp.headers)
    inp.summary = scan.summarize_headers(inp.headers)

    amp, sig = project["columns"]["amplitude"], project["columns"]["sigma"]
    if (amp, sig) != ("F-obs", "SIGF-obs"):
        inp.rename = {amp: "F-obs", sig: "SIGF-obs"}

    names = sorted(inp.names, key=natural_key)
    models = project["models"]
    if models["per_dataset"]:
        inp.models, inp.model_problems = scan.match_files_to_names(models["per_dataset"], names, base)
    if project["phases"]["source"] == "existing" and project["phases"]["existing"]:
        inp.phase_files, inp.phase_problems = scan.match_files_to_names(
            project["phases"]["existing"], names, base, extensions=scan.MTZ_EXTENSIONS)
    if project["validation"]["bound_models"]:
        inp.bound, inp.bound_problems = scan.match_files_to_names(
            project["validation"]["bound_models"], names, base)

    _build_rows(project, inp)
    _choose_reference(project, inp)
    _decide_reindex(project, inp)
    return inp


def _build_rows(project, inp):
    amp, sig = project["columns"]["amplitude"], project["columns"]["sigma"]
    source = project["phases"]["source"]
    phase_cols = project["phases"]["columns"]
    majority_sg = inp.summary.get("space_group")
    outliers = set(inp.summary.get("cell_outliers", []))
    apo = project["models"]["apo"]
    per_dataset = bool(project["models"]["per_dataset"])
    by_name = dict(zip(inp.names, zip(inp.files, inp.headers)))
    for name in sorted(inp.names, key=natural_key):
        path, h = by_name[name]
        row = {"id": inp.ids[name], "original_name": name, "source_mtz": path, "model": "",
               "phases_source": source, "variant": "", "reindex_op": "", "status": "ok",
               "dropped_stage": "", "dropped_reason": "", "flags": "", "phase_file": "",
               "bound_model": inp.bound.get(name, ""), "space_group": h.get("sg_name") or "",
               "cell": scan.format_cell(h.get("cell")),
               "d_min": f"{h['d_min']:.2f}" if h.get("d_min") else "",
               "n_reflections": str(h["nrefl"]) if h.get("nrefl") is not None else ""}
        flags = []
        reason = None
        types = scan.column_types(h) if h["ok"] else {}
        if not h["ok"]:
            reason = f"unreadable MTZ ({h['error']})"
        elif h["nbatch"]:
            reason = "unmerged data (VALDO needs merged amplitudes)"
        elif types.get(amp) != "F" or types.get(sig) != "Q":
            missing = [c for c, t in ((amp, "F"), (sig, "Q")) if types.get(c) != t]
            reason = f"missing column(s) {', '.join(missing)}"
        elif not h["nrefl"]:
            reason = "no reflections"
        elif majority_sg and h["sg_name"] != majority_sg:
            reason = f"space group {h['sg_name']} differs from the other datasets ({majority_sg})"
        elif inp.rename and any(t in types and t not in inp.rename for t in ("F-obs", "SIGF-obs")):
            reason = "has a column named F-obs/SIGF-obs besides the selected amplitude columns"
        if reason:
            row.update(status="dropped", dropped_stage="input", dropped_reason=reason)
        if path in outliers:
            flags.append("cell_outlier")
        if h.get("d_min") and h["d_min"] > LOW_RESOLUTION:
            flags.append("low_resolution")
        if name in inp.models:
            row["model"] = inp.models[name]
        elif apo:
            row["model"] = apo
            if per_dataset:
                flags.append("fallback_model")
        if source == "input_mtz" and h["ok"] and scan.has_phase_columns(h, phase_cols):
            row["phase_file"] = path
        elif source == "existing":
            row["phase_file"] = inp.phase_files.get(name, "")
        row["flags"] = ";".join(flags)
        inp.rows.append(row)


def _choose_reference(project, inp):
    wanted = project["reference"]
    ok = [r for r in inp.rows if r["status"] == "ok"]
    if wanted and wanted != "auto":
        match = [r for r in inp.rows if wanted in (r["original_name"], r["id"])] or \
            [r for r in inp.rows if wanted in (os.path.basename(r["source_mtz"]),
                                               os.path.splitext(os.path.basename(r["source_mtz"]))[0])]
        if not match:
            inp.findings.append(finding(
                "error", "reference_unknown", "Unknown reference dataset",
                f"reference: '{wanted}' is not one of the dataset names.",
                "The reference defines the indexing and the amplitude scale for every dataset.",
                "Set 'reference' to a name from datasets.csv (column original_name), or to auto."))
            return
        row = match[0]
        if row["status"] != "ok":
            inp.findings.append(finding(
                "error", "reference_unusable", "The chosen reference dataset cannot be used",
                f"reference '{wanted}' was excluded: {row['dropped_reason']}.", "",
                "Choose another dataset for 'reference', or set it to auto."))
            return
        inp.reference = {"name": row["original_name"], "id": row["id"], "why": "set in valdo.yaml"}
    else:
        good = [r for r in ok if "cell_outlier" not in r["flags"]] or ok
        if not good:
            return
        row = good[0]
        inp.reference = {"name": row["original_name"], "id": row["id"],
                         "why": "first dataset (by ID) that passed the input checks (upstream convention)"}
    for r in inp.rows:
        if r["id"] == inp.reference["id"]:
            r["flags"] = ";".join([f for f in r["flags"].split(";") if f] + ["reference"])


def _decide_reindex(project, inp):
    setting = project["reindex"]
    if setting == "false":
        inp.reindex = {"enabled": False, "ops": ["x,y,z"], "ops_hkl": ["h,k,l"],
                       "why": "reindex: false in valdo.yaml"}
        return
    if inp.reference is None:
        inp.reindex = {"enabled": False, "ops": ["x,y,z"], "ops_hkl": ["h,k,l"], "why": "no usable reference yet"}
        return
    ref = next(h for n, h in zip(inp.names, inp.headers) if n == inp.reference["name"])
    try:
        alt_ops, alt = scan.alternative_indexing(ref["cell"], ref["sg_name"])
        triplets = [op.triplet() for op in alt_ops]
    except Exception as exc:  # gemmi missing or exotic space group
        alt, triplets = [], []
        inp.findings.append(finding("warning", "reindex_unknown", "Could not determine alternative indexing",
                                    f"gemmi could not analyse the lattice: {exc}.", "",
                                    "Set reindex: true or false in valdo.yaml explicitly."))
    ops = ["h,k,l"] + alt
    sg = ref["sg_name"]
    if setting == "auto":
        if alt:
            why = (f"{sg} with this cell allows {len(alt)} alternative indexing operator(s) ({', '.join(alt)}); "
                   "datasets may be indexed inconsistently, so they are reindexed to the reference")
        else:
            why = f"{sg} with this cell has no alternative indexing, so reindexing is not needed"
        inp.reindex = {"enabled": bool(alt), "ops": ["x,y,z"] + triplets, "ops_hkl": ops, "why": why}
    else:  # "true"
        inp.reindex = {"enabled": bool(alt), "ops": ["x,y,z"] + triplets, "ops_hkl": ops,
                       "why": "reindex: true in valdo.yaml"}
        if not alt:
            inp.findings.append(finding(
                "error", "reindex_impossible", "Reindexing is switched on but not possible",
                f"{sg} with the reference cell admits no alternative indexing, so upstream reindex would write nothing.",
                "", "Set reindex: auto (or false) in valdo.yaml."))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PACKAGES = ["numpy", "pandas", "scipy", "gemmi", "reciprocalspaceship", "torch", "scikit-learn",
            "matplotlib", "tqdm", "pyyaml"]


def package_versions():
    from importlib import metadata
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def cpu_count():
    for var in ("NSLOTS", "SLURM_CPUS_PER_TASK", "PBS_NUM_PPN"):
        if os.environ.get(var, "").isdigit():
            return int(os.environ[var]), var
    try:
        return len(os.sched_getaffinity(0)), "affinity"
    except AttributeError:
        return os.cpu_count() or 1, "os.cpu_count"


def gpu_info():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in ("", "-1"):
        return {"gpus": 0, "note": "CUDA_VISIBLE_DEVICES hides all GPUs: training runs on CPU"}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"gpus": 0, "note": "no GPU found (nvidia-smi not available): training runs on CPU"}
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10).stdout
        n = len([l for l in out.splitlines() if l.startswith("GPU")])
    except Exception:
        n = 0
    return {"gpus": n, "note": f"{n} GPU(s) visible: training uses the first" if n else "no GPU visible: training runs on CPU"}


def in_batch_job():
    return any(os.environ.get(v) for v in ("JOB_ID", "SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"))


def looks_like_login_node():
    host = socket.gethostname().split(".")[0].lower()
    return (not in_batch_job()) and bool(re.match(r"^(log(in)?\d*|.*-login\d*|dev\d+|gpudev\d*)$", host)), host


def free_space(path):
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def environment(project, need_refine=False):
    from valdo import __version__
    ncpu, source = cpu_count()
    login, host = looks_like_login_node()
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "host": host,
           "valdo": __version__, "valdo_path": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
           "packages": package_versions(), "cpus": ncpu, "cpus_source": source,
           "gpu": gpu_info(), "login_node": login, "batch_job": in_batch_job(),
           "free_disk_bytes": free_space(project["workdir"]), "programs": None}
    if need_refine:
        try:
            refine = importlib.import_module("valdo.easy.refine")
            env["programs"] = refine.detect_programs()
        except Exception as exc:
            env["programs"] = {"error": f"{type(exc).__name__}: {exc}"}
    return env


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _names(rows, limit=8):
    names = [f"{r['original_name']} ({r['id']})" for r in rows]
    return names[:limit] + ([f"... and {len(names) - limit} more"] if len(names) > limit else [])


def run_checks(project, inputs=None, deep=False, env=None):
    """Return {"findings", "errors", "warnings", "environment", "inputs"} for a validated project."""
    inp = inputs if inputs is not None else inspect_inputs(project)
    need_refine = project["phases"]["source"] == "refine"
    env = env if env is not None else environment(project, need_refine)
    out = list(inp.findings)
    out += _check_environment(project, env)
    if inp.files:
        out += _check_inputs(project, inp)
        out += _check_settings(project, inp, env)
        if deep:
            out += deep_check(project, inp)
    errors = sum(1 for f in out if f["level"] == "error")
    warnings = sum(1 for f in out if f["level"] == "warning")
    return {"findings": out, "errors": errors, "warnings": warnings, "environment": env,
            "inputs": inputs_summary(project, inp)}


def inputs_summary(project, inp):
    ok = inp.ok_rows()
    s = inp.summary or {}
    return {
        "n_files": len(inp.files), "n_usable": len(ok), "n_excluded": len(inp.rows) - len(ok),
        "space_group": s.get("space_group"), "median_cell": s.get("median_cell"),
        "d_min_range": s.get("d_min_range"), "columns": {"amplitude": project["columns"]["amplitude"],
                                                         "sigma": project["columns"]["sigma"]},
        "detected_columns": inp.columns, "detected_phases": inp.phases, "detected_rfree": inp.rfree,
        "phases_source": project["phases"]["source"],
        "n_with_phases": sum(1 for r in ok if r["phase_file"]),
        "n_with_model": sum(1 for r in ok if r["model"]),
        "n_fallback_model": sum(1 for r in ok if "fallback_model" in r["flags"]),
        "n_bound_models": sum(1 for r in ok if r["bound_model"]),
        "reference": inp.reference, "reindex": inp.reindex, "column_rename": inp.rename,
    }


def _check_environment(project, env):
    out = []
    pk = env["packages"]
    for name in ("numpy", "pandas", "gemmi", "reciprocalspaceship", "torch", "scipy", "matplotlib", "pyyaml"):
        if pk.get(name) is None:
            out.append(finding("error", "missing_package", f"Python package {name} is not installed",
                               f"'{name}' is not installed in this environment ({sys.executable}).",
                               "VALDO cannot run without it.",
                               "Activate the VALDO environment (see environment.yml) or install it."))
    gemmi_version = pk.get("gemmi")
    if gemmi_version and _version_tuple(gemmi_version) > (0, 6, 7):
        out.append(finding("error", "gemmi_too_new", "gemmi is newer than 0.6.7",
                           f"gemmi {gemmi_version} is installed; VALDO uses functions removed after 0.6.7.",
                           "Tagging and reindexing would fail part-way through.",
                           "pip install 'gemmi==0.6.7' in the VALDO environment."))
    if env["login_node"]:
        out.append(finding("warning", "login_node", "This looks like a cluster login node",
                           f"Host '{env['host']}' is not running a batch job.",
                           "A full run takes hours of CPU (or GPU) time; login nodes are shared and may kill long jobs.",
                           "Check here, then run it as a batch job: 'valdo jobscript -o run_valdo.sh', then qsub (or sbatch) run_valdo.sh."))
    ncpu = project["compute"]["ncpu"]
    if ncpu > env["cpus"]:
        out.append(finding("warning", "ncpu_high", "More workers than CPUs",
                           f"compute.ncpu is {ncpu} but only {env['cpus']} CPU(s) are available here ({env['cpus_source']}).",
                           "Oversubscribed CPUs make every stage slower.",
                           f"Set compute.ncpu to {env['cpus']} (or request more slots for the batch job)."))
    free = env["free_disk_bytes"]
    if free is not None and free < 2 * 1024 ** 3:
        out.append(finding("warning", "low_disk", "Little free disk space for the work directory",
                           f"Only {free / 1024 ** 3:.1f} GB free at {project['workdir']}.",
                           "A run writes several copies of the data (about 4x the MTZ size plus the VAE arrays).",
                           "Move 'workdir' to a bigger file system (e.g. cluster scratch)."))
    if project["phases"]["source"] == "refine":
        prog = project["phases"]["program"]
        progs = env.get("programs") or {}
        if "error" in progs:
            out.append(finding("error", "refine_unavailable", "The refinement module could not be loaded",
                               f"valdo.easy.refine: {progs['error']}.", "", "Report this as a bug."))
        elif not progs.get(prog):
            out.append(finding("error", "refine_program_missing", f"{prog} is not available",
                               f"phases.source is 'refine' with program '{prog}', but it is not on PATH.",
                               "Every dataset must be refined to get phases.",
                               "Set up PHENIX/CCP4 in this shell or job script (e.g. source .../ccp4.setup-sh) and rerun 'valdo check'."))
    return out


def _version_tuple(text):
    nums = re.findall(r"\d+", text)[:3]
    return tuple(int(n) for n in nums) + (0,) * (3 - len(nums))


def _excluded_fix(reason, amp, sig, detected=None, others_ok=True):
    """What to do about input files excluded for `reason` (the texts set in _build_rows)."""
    if reason.startswith("unreadable"):
        return ("The files are damaged or not MTZ files: copy them again from the source, or remove them "
                "from the data folder." + (" Nothing else is needed; the other datasets run without them."
                                           if others_ok else ""))
    if reason.startswith("unmerged"):
        return "Merge and scale them first (e.g. aimless), or remove them from the data folder."
    if reason.startswith("missing column"):
        det = detected or {}
        if det.get("amplitude") and (det["amplitude"], det.get("sigma")) != (amp, sig):
            return (f"The files have the amplitude/sigma columns {det['amplitude']}/{det['sigma']} "
                    f"(in {det.get('n_files', '?')} files), not {amp}/{sig}: set columns.amplitude: "
                    f"{det['amplitude']} and columns.sigma: {det['sigma']} in valdo.yaml, or rerun "
                    "'valdo init --force'.")
        return (f"Check the column names with gemmi or mtzdump. If most files use other names, set "
                f"columns.amplitude/sigma (now {amp}/{sig}) in valdo.yaml, or rerun 'valdo init --force'.")
    if reason.startswith("space group"):
        return ("Datasets in another crystal form cannot share one VAE: remove them, or run them as a "
                "separate project.")
    if reason.startswith("has a column named"):
        return ("Rename or delete that column in these files (VALDO writes F-obs/SIGF-obs itself), or "
                "choose those columns as columns.amplitude/sigma in valdo.yaml.")
    return "Check the files (see datasets.csv) or remove them from the data folder."


def _pattern_example(spec, name, base):
    """(what `spec` looks for for dataset `name`, the folder it looks in, a few files there or None)."""
    spec = str(spec)
    if "{name}" in spec:
        want = scan.expand_path(spec.replace("{name}", name), base)
        folder = os.path.dirname(scan.expand_path(spec.split("{name}")[0] + "x", base))
    else:
        path = scan.expand_path(spec, base)
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        want = f"a file whose name contains '{name}'" + (f" in {folder}" if os.path.isdir(path) else f" matching {path}")
    existing = None
    if not scan.has_glob(folder) and os.path.isdir(folder):
        existing = sorted(f for f in os.listdir(folder) if not f.startswith("."))
    return want, folder, existing


def model_pattern_finding(project, inp):
    """Error if models.per_dataset matches none of the usable datasets: a wrong pattern would otherwise
    give every dataset the apo model (or no model) and the run would go ahead."""
    spec = project["models"]["per_dataset"]
    ok = inp.ok_rows()
    if not spec or not ok or any(r["original_name"] in inp.models for r in ok):
        return None
    name = ok[0]["original_name"]
    want, folder, existing = _pattern_example(spec, name, project.get("_dir", os.getcwd()))
    items = [f"for {name} it looks for {want}"]
    if existing is None and not scan.has_glob(folder):
        items.append(f"{folder} does not exist")
    elif existing == []:
        items.append(f"{folder} is empty")
    elif existing:
        items.append(f"files in {folder}: {', '.join(existing[:4])}{', ...' if len(existing) > 4 else ''}")
    apo = project["models"]["apo"]
    return finding("error", "models_pattern_no_match", "models.per_dataset matches no dataset",
                   f"models.per_dataset ({spec}) matched none of the {len(ok)} datasets.",
                   "Without their own models, every dataset would " +
                   ("be tagged and viewed with the apo model, which is wrong wherever a dataset's phases came "
                    "from a model at another origin or indexing." if apo else
                    "get no hit list (blob detection needs a model)."),
                   "Correct models.per_dataset in valdo.yaml (a pattern with {name} = the dataset name, e.g. "
                   "'/models/{name}-final.pdb', or a folder whose file names contain the dataset names), or "
                   "remove it to use models.apo for every dataset on purpose.", items)


def _check_inputs(project, inp):
    out = []
    rows = inp.rows
    ok = inp.ok_rows()
    cols = inp.columns or {}
    amp, sig = project["columns"]["amplitude"], project["columns"]["sigma"]
    excluded = [r for r in rows if r["status"] != "ok"]
    if cols.get("amplitude") is None:
        extra = (" They contain intensities only: convert them to amplitudes first (French-Wilson, e.g. ctruncate "
                 "or phenix.french_wilson)." if cols.get("intensity_only") else "")
        out.append(finding("error", "no_amplitudes", "No amplitude columns found",
                           "No file has an amplitude column (type F) with a matching SIG column (type Q)." + extra,
                           "VALDO models structure-factor amplitudes.",
                           "Check the files with 'mtzdump' or gemmi; pass --columns F,SIGF to 'valdo init'."))
    if excluded:
        reasons = {}
        for r in excluded:
            key = re.sub(r"\(.*\)", "", r["dropped_reason"]).strip()
            reasons.setdefault(key, []).append(r)
        level = "error" if len(excluded) > MAX_INPUT_LOSS * len(rows) else "warning"
        for reason, rs_ in reasons.items():
            out.append(finding(level, "excluded_inputs", f"{len(rs_)} file(s) cannot be used: {reason}",
                               f"{len(rs_)} of {len(rows)} files will be skipped ({reason}).",
                               "They are listed in datasets.csv with dropped_stage 'input'." +
                               (" More than half of the files are affected, so a setting is probably wrong."
                                if level == "error" else ""),
                               _excluded_fix(reason, amp, sig, cols, others_ok=bool(ok)), _names(rs_)))
    if cols.get("alternatives") and cols.get("amplitude") == amp:
        out.append(finding("info", "other_amplitudes", "Other amplitude columns exist",
                           f"Using {amp}/{sig}; also found: {', '.join(cols['alternatives'][:4])}.", "",
                           "Change columns.amplitude/sigma in valdo.yaml if you want another pair."))
    if not ok:   # every file is excluded: the findings below would only repeat that
        return out
    s = inp.summary
    bad = [r for r in ok if "cell_outlier" in r["flags"]]
    if bad:
        out.append(finding("warning", "cell_outliers", f"{len(bad)} dataset(s) have unusual unit cells",
                           f"Cell edges differ by more than 3% (or angles by 3 degrees) from the median {scan.format_cell(s['median_cell'])}.",
                           "Non-isomorphous crystals scale poorly and are often removed by the min_cc filter.",
                           "Usually nothing to do; remove them from 'data' if they are from a different crystal form.",
                           _names(bad)))
    low = [r for r in ok if "low_resolution" in r["flags"]]
    if low:
        out.append(finding("warning", "low_resolution", f"{len(low)} dataset(s) are worse than {LOW_RESOLUTION} A",
                           f"Their high-resolution limit is above {LOW_RESOLUTION} A.",
                           "Blob detection blurs maps by a resolution-dependent amount; at low resolution there is "
                           "little or no blurring, and the reflection intersection used by the VAE shrinks.",
                           "Consider removing them from 'data'.", _names(low)))
    if len(ok) >= 2 and s.get("d_min_range"):
        lo, hi = s["d_min_range"]
        out.append(finding("info", "resolution", "Resolution range",
                           f"High-resolution limits range from {lo:.2f} to {hi:.2f} A (median {s['d_min_median']:.2f} A)."))
    if inp.rename:
        out.append(finding("info", "column_rename", "Columns will be renamed",
                           f"{amp}/{sig} are copied as F-obs/SIGF-obs into the work directory "
                           "(upstream reindex and filter require these names). Your files are not modified."))
    # models
    no_model = [r for r in ok if not r["model"]]
    if inp.model_problems:
        real = [p for p in inp.model_problems if not p.endswith("does not exist")]
        if real:
            out.append(finding("error", "model_ambiguous", "Some models could not be matched unambiguously",
                               "Several model files match the same dataset name.", "",
                               "Give models.per_dataset as a pattern with {name}, e.g. '/models/{name}-final.pdb'.",
                               real[:8]))
    pattern = model_pattern_finding(project, inp)
    if pattern:
        out.append(pattern)
    if no_model and not pattern:
        none = len(no_model) == len(ok)
        out.append(finding("error" if none else "warning", "missing_models",
                           "No dataset has a model" if none else f"{len(no_model)} dataset(s) have no model",
                           "No per-dataset model matched them and models.apo is not set.",
                           "Blob detection and tagging need a model: datasets without one are used to train the "
                           "VAE but get no hit list" + (", so this run would find nothing." if none else "."),
                           "Set models.apo as a fallback model (e.g. the reference dataset's own model: it must be "
                           "in the reference's indexing), or provide the missing models.", _names(no_model)))
    fallback = [r for r in ok if "fallback_model" in r["flags"]]
    if fallback and not pattern:   # (when the pattern matched nothing, that error says it all)
        many = len(fallback) > FALLBACK_WARN * len(ok)
        out.append(finding("warning" if many else "info", "fallback_models",
                           f"{len(fallback)} of {len(ok)} dataset(s) use the apo model as fallback",
                           "No per-dataset model matched them, so models.apo is used (flag fallback_model).",
                           "Tagging near the focal residue, nearest residues and the viewer use that model's "
                           "coordinates; if a dataset's phases came from a model at another origin or indexing, "
                           "they are wrong for that dataset (the run flags it model_map_mismatch)." +
                           (" This is more than 10% of the datasets: check that models.per_dataset matches your "
                            "model files." if many else ""),
                           "Give these datasets the model their phases came from (models.per_dataset)." if many else "",
                           _names(fallback)))
    # phases
    source = project["phases"]["source"]
    if source == "input_mtz":
        no_phase = [r for r in ok if not r["phase_file"]]
        c = project["phases"]["columns"]
        if len(no_phase) == len(ok) and ok:
            out.append(finding("error", "no_input_phases", "The data files contain no map coefficients",
                               f"phases.source is input_mtz, but no file has the phase columns {c['phwt']}/{c['phdelwt']}.",
                               "Difference maps need phases.",
                               "Use phases.source: refine (with phenix or dimple) or existing (your refined MTZs), "
                               "or correct phases.columns."))
        elif no_phase:
            out.append(finding("warning", "missing_input_phases", f"{len(no_phase)} dataset(s) lack map coefficients",
                               f"They have no {c['phwt']}/{c['phdelwt']} columns.",
                               "They are used to train the VAE but get no hit list.", "", _names(no_phase)))
    elif source == "existing":
        no_phase = [r for r in ok if not r["phase_file"]]
        real = [p for p in inp.phase_problems if not p.endswith("does not exist")]
        if real:
            out.append(finding("error", "phase_ambiguous", "Some refined MTZs could not be matched unambiguously",
                               "Several files match the same dataset name.", "",
                               "Give phases.existing as a pattern with {name}.", real[:8]))
        if len(no_phase) == len(ok) and ok:
            gone = [p for p in inp.phase_problems if p.endswith("does not exist")]
            out.append(finding("error", "no_existing_phases", "No refined MTZ matched any dataset",
                               f"phases.existing ({project['phases']['existing']}) matched none of the dataset names"
                               + (f" ({gone[0]})." if gone else "."),
                               "", "Use a pattern with {name}, e.g. '/refined/{name}/final.mtz'."))
        elif no_phase:
            out.append(finding("warning", "missing_existing_phases", f"{len(no_phase)} dataset(s) have no refined MTZ",
                               "They are used to train the VAE but get no hit list.", "", "", _names(no_phase)))
    # bound models
    if project["validation"]["bound_models"]:
        gone = [p for p in inp.bound_problems if p.endswith("does not exist")]
        real = [p for p in inp.bound_problems if p not in gone]
        if gone:
            out.append(finding("warning", "bound_models_missing", "The known bound models were not found",
                               f"validation.bound_models: {gone[0][:-len(' does not exist')]} does not exist.",
                               "Without them the report has no known-ligand validation; everything else runs.",
                               "Correct the path, or set validation.bound_models: null in valdo.yaml."))
        if real:
            out.append(finding("warning", "bound_ambiguous", "Some bound models could not be matched",
                               "Several files match the same dataset name; those datasets are skipped in the validation.",
                               "", "Use a pattern with {name}.", real[:8]))
        n_bound = sum(1 for r in ok if r["bound_model"])
        if not n_bound and not gone:
            out.append(finding("warning", "no_bound_models", "No bound model matched a dataset",
                               "validation.bound_models matched none of the dataset names.", "",
                               "Check the pattern; it must contain the dataset names."))
    # reference / reindex
    if inp.reference:
        out.append(finding("info", "reference", "Reference dataset",
                           f"{inp.reference['name']} ({inp.reference['id']}): {inp.reference['why']}."))
    elif ok:
        out.append(finding("error", "no_reference", "No reference dataset", "No dataset can serve as reference.",
                           "", "Set 'reference' in valdo.yaml."))
    if inp.reindex:
        out.append(finding("info", "reindex", "Reindexing " + ("on" if inp.reindex["enabled"] else "off"),
                           inp.reindex["why"] + "."))
    return out


def _check_settings(project, inp, env):
    out = []
    ok = inp.ok_rows()
    n = len(ok)
    t = project["advanced"]["train"]
    if n == 0 and inp.rows:   # every file was excluded: the excluded_inputs error says why
        return out
    if n < MIN_DATASETS:
        out.append(finding("error", "too_few_datasets", "Too few datasets",
                           f"Only {n} usable dataset(s).", "VALDO compares each dataset with the others.",
                           "Add more datasets."))
        return out
    if n < FEW_DATASETS:
        out.append(finding("warning", "few_datasets", f"Only {n} datasets",
                           f"{n} usable datasets; the validated PTP1B run used 1,620.",
                           "The VAE learns what an unbound crystal looks like from many datasets; with fewer than "
                           f"~{FEW_DATASETS} the reconstructions (and hit ranking) are less reliable.",
                           "Proceed if this is a test; otherwise add datasets (apo or soaked) from the same crystal form."))
    n_train = int(n * t["train_fraction"])
    if n_train < 1 or n - n_train < 1:
        out.append(finding("error", "bad_split", "Training split leaves an empty set",
                           f"train_fraction {t['train_fraction']} of {n} datasets gives {n_train} training and {n - n_train} validation datasets.",
                           "Upstream training crashes with an empty training or validation set.",
                           "Use train_fraction 0.8 (validated) or add datasets."))
    elif t["batch_size"] > n_train:
        out.append(finding("info", "batch_size", "Batch larger than the training set",
                           f"batch_size {t['batch_size']} > {n_train} training datasets: one batch per epoch."))
    if project["phases"]["source"] == "input_mtz" and inp.phases is None and n:
        pass  # reported by _check_inputs
    return out


def deep_check(project, inp, threads=8):
    """Read the reflection data of every usable file: NaN, non-positive F or sigma, duplicates."""
    import gemmi
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    amp, sig = project["columns"]["amplitude"], project["columns"]["sigma"]
    rows = inp.ok_rows()

    def one(row):
        try:
            mtz = gemmi.read_mtz_file(row["source_mtz"])
            f = np.array(mtz.column_with_label(amp), copy=False)
            s = np.array(mtz.column_with_label(sig), copy=False)
            hkl = mtz.make_miller_array()
            finite = np.isfinite(f) & np.isfinite(s)
            dup = len(hkl) - len(np.unique(hkl, axis=0))
            return {"nan": int((~finite).sum()), "f_nonpos": int((f[finite] <= 0).sum()),
                    "s_nonpos": int((s[finite] <= 0).sum()), "dup": int(dup), "n": int(len(f)), "error": None}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(rows)))) as pool:
        results = list(pool.map(one, rows))
    out = []
    groups = {"error": [], "nan": [], "f_nonpos": [], "s_nonpos": [], "dup": []}
    ref_bad = False
    for row, res in zip(rows, results):
        if res.get("error"):
            groups["error"].append(row)
            continue
        for key in ("nan", "f_nonpos", "s_nonpos", "dup"):
            if res[key]:
                groups[key].append(row)
        if inp.reference and row["id"] == inp.reference["id"] and res["f_nonpos"]:
            ref_bad = True
    if groups["error"]:
        out.append(finding("error", "deep_unreadable", f"{len(groups['error'])} file(s) have unreadable reflection data",
                           "The header is fine but the reflections cannot be read.", "", "Re-copy or remove them.",
                           _names(groups["error"])))
    if groups["nan"]:
        out.append(finding("info", "deep_nan", f"{len(groups['nan'])} file(s) contain missing amplitudes",
                           "Reflections with missing F or sigma are removed by the standardize stage.", "", "",
                           _names(groups["nan"])))
    if groups["f_nonpos"]:
        level = "error" if ref_bad else "warning"
        out.append(finding(level, "deep_f_nonpositive", f"{len(groups['f_nonpos'])} file(s) contain amplitudes <= 0",
                           "Scaling takes log(F); a zero or negative amplitude makes that dataset's scaled amplitudes NaN "
                           "(it is then dropped), and in the reference it spoils every dataset.",
                           "", "Use French-Wilson amplitudes (always > 0)" + (" and choose another reference" if ref_bad else "") + ".",
                           _names(groups["f_nonpos"])))
    if groups["s_nonpos"]:
        out.append(finding("warning", "deep_sigma_nonpositive", f"{len(groups['s_nonpos'])} file(s) contain sigma <= 0",
                           "Weights use 1/sigma^2 in reindexing.", "", "Check how the data were processed.",
                           _names(groups["s_nonpos"])))
    if groups["dup"]:
        out.append(finding("error", "deep_duplicates", f"{len(groups['dup'])} file(s) have repeated reflections",
                           "The same (h,k,l) appears more than once: the data are not merged.",
                           "VALDO aligns datasets reflection by reflection.", "Use merged data.",
                           _names(groups["dup"])))
    if not out:
        out.append(finding("info", "deep_ok", "Reflection data look fine",
                           f"Read {len(rows)} files: no unreadable data, no non-positive amplitudes, no duplicates."))
    return out
