"""Known failure patterns -> plain-language explanation (what went wrong, likely cause, fix).

`explain(stage, text)` scans the end of a stage log (or an exception message) for the first
known pattern that applies to that stage and returns a dict with keys id, what, cause, fix.
Unknown failures get a generic answer that points to the log.
"""

import re

# Each hint: id, regex (searched case-sensitively unless (?i) is given), stages (None = any),
# what went wrong, likely cause, how to fix.
HINTS = [
    {"id": "disk_full", "pattern": r"No space left on device|Disk quota exceeded|\[Errno 28\]|\[Errno 122\]",
     "stages": None,
     "what": "The disk (or your quota) filled up while writing results.",
     "cause": "The work directory is on a file system without enough free space for this many datasets.",
     "fix": "Free space or move 'workdir' in valdo.yaml to a larger file system (e.g. cluster scratch), then rerun 'valdo run'; finished stages are kept."},
    {"id": "memory", "pattern": r"MemoryError|std::bad_alloc|Cannot allocate memory|\[Errno 12\]|Killed|oom-kill|Out of memory",
     "stages": None,
     "what": "The process ran out of memory.",
     "cause": "Large screens need several GB per worker; parallel stages multiply that by compute.ncpu.",
     "fix": "Request more memory for the batch job (e.g. 'valdo jobscript --mem-gb 8') or lower compute.ncpu in valdo.yaml, then rerun 'valdo run'."},
    {"id": "cuda_oom", "pattern": r"CUDA out of memory|CUDA error: out of memory",
     "stages": ["train", "reconstruct", "scale"],
     "what": "The GPU ran out of memory.",
     "cause": "The VAE arrays (datasets x reflections) do not fit on this GPU.",
     "fix": "Run on CPU (set CUDA_VISIBLE_DEVICES='' before 'valdo run') or use a GPU with more memory."},
    {"id": "gpu_pickle", "pattern": r"Attempting to deserialize object on a CUDA device",
     "stages": ["reconstruct"],
     "what": "The trained model was saved on a GPU and cannot be loaded on this CPU-only machine.",
     "cause": "train ran with a GPU visible, reconstruct runs without one.",
     "fix": "Run reconstruct on a GPU node, or rerun training on CPU: CUDA_VISIBLE_DEVICES='' valdo run --from train."},
    {"id": "singular_matrix", "pattern": r"Singular matrix|LinAlgError",
     "stages": ["scale"],
     "what": "Scaling failed for a dataset: too few reflections in common with the reference.",
     "cause": "A dataset is in a different space group or setting, has very low resolution or completeness, or is not merged into the same asymmetric unit as the reference.",
     "fix": "Run 'valdo check --deep' and look at the space-group, cell and resolution warnings; remove the offending file from 'data' (or choose another 'reference') and rerun."},
    {"id": "missing_column", "pattern": r"KeyError: '(F-obs|SIGF-obs|F-obs-scaled|SIGF-obs-scaled|PH2FOFCWT|PHFOFCWT|diff|recons|WDF)'|not in index|None of \[Index",
     "stages": None,
     "what": "An expected column is missing from an MTZ file.",
     "cause": "The column names in valdo.yaml (columns.*, phases.columns.*) do not match the files, or a file was replaced after 'valdo init'.",
     "fix": "Run 'valdo check' to see which files lack which columns; correct columns/phases.columns in valdo.yaml (or 'valdo init --force' to re-detect), then rerun."},
    {"id": "file_missing", "pattern": r"FileNotFoundError|No such file or directory|\[Errno 2\]",
     "stages": None,
     "what": "A file that a stage needs does not exist.",
     "cause": "Input files moved or were deleted (cluster scratch is purged after a few weeks), or an earlier stage did not write its outputs.",
     "fix": "Run 'valdo check' (it lists missing inputs) and 'valdo status'; restore the inputs or rerun the earlier stages with 'valdo run --from <stage>'."},
    {"id": "permission", "pattern": r"PermissionError|Permission denied|\[Errno 13\]|Read-only file system",
     "stages": None,
     "what": "No permission to read an input or to write in the work directory.",
     "cause": "The files belong to someone else or the folder is read-only.",
     "fix": "Check permissions ('ls -l'); put 'workdir' in a folder you own; ask the data owner for read access."},
    {"id": "bad_mtz", "pattern": r"(?i)(not an mtz|Invalid MTZ|Failed to read|could not read mtz|unexpected end of file|truncated)",
     "stages": None,
     "what": "An MTZ file could not be read.",
     "cause": "The file is corrupt or incomplete (e.g. a copy was interrupted).",
     "fix": "Run 'valdo check' to find unreadable files; re-copy or remove them from 'data' and rerun."},
    {"id": "nothing_left", "pattern": r"No objects to concatenate|need at least one array to concatenate|matched no files|no MTZ files found",
     "stages": None,
     "what": "No datasets were left to process at this stage.",
     "cause": "Every dataset was dropped earlier, a file list was empty, or (blob stage) no dataset had a model file.",
     "fix": "Look at datasets.csv (columns status, dropped_stage, dropped_reason) and 'valdo status' to see where the datasets were lost."},
    {"id": "shape_mismatch", "pattern": r"Shape of passed values|operands could not be broadcast|size mismatch",
     "stages": ["rescale", "reconstruct", "train"],
     "what": "Arrays from different runs do not fit together.",
     "cause": "Outputs of an older run were mixed with newer ones.",
     "fix": "Rerun from preprocess: 'valdo run --from preprocess'."},
    {"id": "nan_loss", "pattern": r"(?i)loss.*\bnan\b|non-finite loss|nan values? in (the )?loss",
     "stages": ["train"],
     "what": "VAE training produced NaN losses.",
     "cause": "Usually a dataset with non-finite amplitudes, or a learning rate that is too high.",
     "fix": "Check datasets.csv for scaling_diverged flags; lower advanced.train.learning_rate (e.g. 0.0005) and run 'valdo run --from train'."},
    {"id": "no_torch", "pattern": r"No module named 'torch'",
     "stages": None,
     "what": "PyTorch is not installed in this Python environment.",
     "cause": "valdo was started from an environment without torch.",
     "fix": "Activate the VALDO environment (see environment.yml) and rerun; 'valdo check' lists the packages it found."},
    {"id": "no_package", "pattern": r"No module named '(gemmi|reciprocalspaceship|sklearn|scipy|matplotlib|tqdm|yaml)'",
     "stages": None,
     "what": "A required Python package is missing.",
     "cause": "The active environment is not the VALDO environment.",
     "fix": "Activate the VALDO environment (see environment.yml); 'valdo check' lists the packages it found."},
    {"id": "gemmi_api", "pattern": r"has no attribute '(fractionalization_matrix|orthogonalization_matrix)'|ContextualVersionConflict",
     "stages": None,
     "what": "The installed gemmi is too new for VALDO.",
     "cause": "VALDO needs gemmi 0.6.7 or older (newer versions renamed functions it uses).",
     "fix": "Install gemmi 0.6.7 in the VALDO environment: pip install 'gemmi==0.6.7'."},
    {"id": "refine_program", "pattern": r"(?i)(phenix\.refine|dimple|refmac5?)[^\n]*(not found|No such file)|command not found",
     "stages": ["refine"],
     "what": "The refinement program could not be started.",
     "cause": "phenix or CCP4 (dimple) is not set up for valdo: programs.phenix/programs.ccp4 is not set in valdo.yaml and the program is not on PATH in this shell or batch job.",
     "fix": "Set programs.phenix (or programs.ccp4) in valdo.yaml to the installation folder, check with 'valdo check', then rerun."},
    {"id": "timeout", "pattern": r"(?i)timed? ?out|TimeoutExpired",
     "stages": ["refine"],
     "what": "Refinement of some datasets took longer than the time limit.",
     "cause": "Slow refinements or an overloaded node.",
     "fix": "Increase advanced.refine.timeout_min in valdo.yaml and rerun; finished datasets are kept."},
    {"id": "pool_broken", "pattern": r"BrokenProcessPool|A process in the process pool was terminated|worker exited",
     "stages": None,
     "what": "A worker process died.",
     "cause": "Usually out of memory with many parallel workers.",
     "fix": "Lower compute.ncpu or request more memory, then rerun 'valdo run'."},
    {"id": "stale_handle", "pattern": r"Stale file handle|\[Errno 116\]",
     "stages": None,
     "what": "The network file system lost track of a file.",
     "cause": "Files were changed or deleted by another process while VALDO was running.",
     "fix": "Make sure only one VALDO run uses this work directory, then rerun 'valdo run'."},
    {"id": "too_many_files", "pattern": r"Too many open files|\[Errno 24\]",
     "stages": None,
     "what": "The process opened too many files at once.",
     "cause": "The system limit on open files is low for this many datasets and workers.",
     "fix": "Lower compute.ncpu, or raise the limit ('ulimit -n 4096') in the job script."},
]

GENERIC = {
    "id": "unknown",
    "what": "The stage stopped with an error that valdo does not recognise.",
    "cause": "See the last lines of the stage log.",
    "fix": "Read the log; fix the input or setting it points to and rerun 'valdo run' (finished stages are kept). "
           "If it looks like a bug, report it with the log file.",
}

_COMPILED = [(h, re.compile(h["pattern"])) for h in HINTS]


def explain(stage, text):
    """Best hint for `stage` given log or exception `text` (the last match in the text wins)."""
    text = text or ""
    best, best_pos = None, -1
    for hint, regex in _COMPILED:
        if hint["stages"] is not None and stage not in hint["stages"]:
            continue
        for m in regex.finditer(text):
            if m.start() > best_pos:
                best, best_pos = hint, m.start()
    chosen = best or GENERIC
    return {"id": chosen["id"], "what": chosen["what"], "cause": chosen["cause"], "fix": chosen["fix"]}


def tail(path, n_lines=40, max_bytes=200_000):
    """Last lines of a text file ('' if unreadable)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [l for l in data.replace("\r", "\n").split("\n") if l.strip()]
    return "\n".join(lines[-n_lines:])
