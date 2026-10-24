"""The project file (valdo.yaml): schema, strict loading and the commented writer.

One file replaces the ten upstream stage configs. It has three parts: what the user must
look at (data, models, workdir), what `valdo init` detected (check it), and an 'advanced'
part pre-filled with the validated PTP1B settings (upstream _TEMPLATES and
PTP1B_pipeline/configs), written out explicitly so the file is the single source of truth.

Loading is strict: unknown keys are errors with a did-you-mean suggestion, numeric text such
as '1e-3' (which YAML reads as a string) is converted with a note, '~' and $VARS are
expanded, and relative paths are resolved against the folder of valdo.yaml.
"""

import copy
import datetime
import difflib
import json
import os

from valdo.easy.scan import expand_path


class ProjectError(Exception):
    """Invalid project file; .messages holds one plain-language line per problem.

    .kind says what kind of problem it is: "missing" (no such file), "syntax" (not valid YAML;
    .details has the parser's own text) or "settings" (unknown keys, bad values)."""

    def __init__(self, messages, path=None, kind="settings", details=None):
        self.messages = list(messages)
        self.path = path
        self.kind = kind
        self.details = list(details or [])
        super().__init__("\n".join(self.messages))


def _yaml_problem(path, exc):
    """A plain-language explanation of a YAML syntax error (with the line numbers)."""
    lines = []
    for mark in (getattr(exc, "context_mark", None), getattr(exc, "problem_mark", None)):
        if mark is not None and mark.line + 1 not in lines:
            lines.append(mark.line + 1)
    where = ""
    if lines:
        where = f" near line {lines[0]}" if len(lines) == 1 else f" near lines {lines[0]}-{lines[-1]}"
    return (f"{path} is not valid YAML{where}. Check there: settings inside a section are indented by "
            "2 spaces per level, there is no space before ':' and one space after it, no tab characters, "
            "and quotes and brackets are closed.")


class Field:
    """One setting: kind, default, one-line comment, and allowed values."""

    def __init__(self, kind, default=None, doc="", choices=None, nullable=False,
                 required=False, minimum=None, maximum=None, upstream=None):
        self.kind = kind
        self.default = default
        self.doc = doc
        self.choices = choices
        self.nullable = nullable
        self.required = required
        self.minimum = minimum
        self.maximum = maximum
        self.upstream = upstream


# Validated PTP1B settings (upstream _TEMPLATES == PTP1B_pipeline/configs/config_*.yaml)
ADVANCED_DEFAULTS = {
    "reindex": {"wcorr": True, "cc_min_dif": 0.2},
    "scale": {"when_opt": "all"},
    "filter": {"max_rfree": 0.45, "min_cc": 0.55},
    "train": {"latent_dim": 7, "n_hidden_layers": [3, 6], "n_hidden_size": 100, "epochs": 500,
              "train_fraction": 0.8, "batch_size": 100, "learning_rate": 0.001, "w_kl": 1.0,
              "eps": 0.02, "stdof": 128, "include_errors": True, "random_seed": 11231,
              "activation": "relu"},
    "reconstruct": {"ml_recon": True, "repeats": 1},
    "blobs": {"sigdF_pct": 95.0, "absdF_pct": 99.99, "extrapolate_factors": [2, 4, 6, 8, 16],
              "cutoff": 3.5, "radius_in_A": 4.0},
    "refine": {"eff": None, "extra_args": [], "timeout_min": 60},
    "report": {"top_n": 20, "maps": True},
}

SCHEMA = {
    "data": Field("pattern", None, "MTZ files: a folder, a glob pattern (in quotes) or a .txt file with one path per line", required=True),
    "models": {
        "apo": Field("path", None, "one apo model (PDB) for every dataset; also the fallback for datasets without their own model", nullable=True),
        "per_dataset": Field("pattern", None, "one model per dataset: a pattern with {name}, or a folder/glob whose file names contain the dataset name", nullable=True),
    },
    "workdir": Field("path", "valdo_run", "all results go here (relative paths are relative to this file)"),
    "columns": {
        "amplitude": Field("str", "F-obs", "observed structure-factor amplitude column (same name in every file)", upstream="standardize.expcolumns"),
        "sigma": Field("str", "SIGF-obs", "its standard-deviation column", upstream="standardize.expcolumns"),
    },
    "phases": {
        "source": Field("str", "input_mtz", "where map phases come from: input_mtz (map coefficients already in the data files), refine (refine every dataset), existing (your refined MTZs)", choices=["input_mtz", "refine", "existing"]),
        "existing": Field("pattern", None, "source: existing -> your refined MTZs (pattern with {name}, folder or glob)", nullable=True),
        "program": Field("str", None, "source: refine -> refinement program: phenix or dimple", choices=["phenix", "dimple"], nullable=True),
        "columns": {
            "fwt": Field("str", "2FOFCWT", "2mFo-DFc amplitude column in the phase files (context maps)", nullable=True),
            "phwt": Field("str", "PH2FOFCWT", "2mFo-DFc phase column (phases for the VALDO difference maps)", nullable=True, upstream="add_phases_and_blobs.phase_2FOFC_col_in"),
            "delfwt": Field("str", "FOFCWT", "mFo-DFc amplitude column", nullable=True),
            "phdelwt": Field("str", "PHFOFCWT", "mFo-DFc phase column", nullable=True, upstream="add_phases_and_blobs.phase_FOFC_col_in"),
        },
        "rfree": Field("str", None, "R-free flag column in the phase files (null = none)", nullable=True, upstream="add_phases_and_blobs.rfree_label_in"),
    },
    "reindex": Field("reindex", "auto", "auto | true | false. auto = only if the lattice allows alternative indexing (as upstream reindex decides)"),
    "reference": Field("str", "auto", "reference dataset name, or auto = the first dataset (by ID) that passes the input checks"),
    "focal_residue": {
        "seqid": Field("int", None, "residue number to exclude blobs around (PTP1B: 215 for Cys215); null = none", nullable=True, upstream="tag_blobs.focal_seqid"),
        "radius": Field("float", 5.0, "exclusion radius in A", minimum=0.0, upstream="tag_blobs.focal_radius"),
        "tag": Field("str", "focal", "name of the tag column in the blob tables (e.g. cys215)", upstream="tag_blobs.focal_tag_name"),
    },
    "validation": {
        "bound_models": Field("pattern", None, "optional: known ligand-bound models, for a descriptive check in the report", nullable=True, upstream="tag_blobs.bound_models_folder"),
        "ligand_resnames": Field("list_str", ["LIG"], "ligand residue names in those models"),
    },
    "compute": {
        "ncpu": Field("int", 4, "worker processes for the parallel stages (use your batch job's slot count)", minimum=1, upstream="<stage>.ncpu"),
    },
    "advanced": {
        "reindex": {
            "wcorr": Field("bool", True, "weighted per-resolution-shell CC (upstream template value; the schema fallback would be false)", upstream="reindex.wcorr"),
            "cc_min_dif": Field("float", 0.2, "keep both indexing variants if their CCs differ by less than this", minimum=0.0, upstream="reindex.cc_min_dif"),
        },
        "scale": {
            "when_opt": Field("when_opt", "all", "\"all\", \"never\" or a number in [0, 1]: when to refine B/anisotropy numerically", upstream="scale.when_opt"),
        },
        "filter": {
            "max_rfree": Field("float", 0.45, "drop datasets whose R-free after refinement is above this (phases.source: refine only)", minimum=0.0, maximum=1.0, upstream="filter.max_rfree"),
            "min_cc": Field("float", 0.55, "drop datasets whose CC to the reference after scaling is below this (0 = off)", minimum=0.0, maximum=1.0, upstream="filter.min_cc"),
        },
        "train": {
            "latent_dim": Field("int", 7, "latent dimensions", minimum=1, upstream="train.latent_dim"),
            "n_hidden_layers": Field("list_int", [3, 6], "[encoder, decoder] hidden layers", upstream="train.n_hidden_layers"),
            "n_hidden_size": Field("int", 100, "units per hidden layer", minimum=1, upstream="train.n_hidden_size"),
            "epochs": Field("int", 500, "training epochs", minimum=1, upstream="train.epochs"),
            "train_fraction": Field("float", 0.8, "fraction of datasets used for training (the rest validates); must be < 1", minimum=0.0, maximum=1.0, upstream="train.train_fraction"),
            "batch_size": Field("int", 100, "minibatch size", minimum=1, upstream="train.batch_size"),
            "learning_rate": Field("float", 0.001, "Adam learning rate", minimum=0.0, upstream="train.learning_rate"),
            "w_kl": Field("float", 1.0, "weight of the KL term", minimum=0.0, upstream="train.w_kl"),
            "eps": Field("float", 0.02, "noise floor added to sigma(F)", minimum=0.0, upstream="train.eps"),
            "stdof": Field("int", 128, "Student-t degrees of freedom of the loss; null = Gaussian", nullable=True, minimum=1, upstream="train.stdof"),
            "include_errors": Field("bool", True, "use sigma(F) in the loss (must be true: upstream preprocess cannot run without it)", upstream="train.include_errors"),
            "random_seed": Field("int", 11231, "random seed (training split and initialisation)", upstream="train.random_seed"),
            "activation": Field("str", "relu", "relu | tanh | sigmoid", choices=["relu", "tanh", "sigmoid"], upstream="train.activation"),
        },
        "reconstruct": {
            "ml_recon": Field("bool", True, "true = deterministic (MAP) reconstruction", upstream="reconstruct.ml_recon"),
            "repeats": Field("int", 1, "> 1 samples the VAE (needs ml_recon: false) and adds SIG_recons", minimum=1, upstream="reconstruct.repeats"),
        },
        "blobs": {
            "sigdF_pct": Field("float", 95.0, "percentile of sigma(dF) where difference weights fall off", minimum=0.0, maximum=100.0, upstream="add_phases_and_blobs.sigdF_pct"),
            "absdF_pct": Field("float", 99.99, "percentile of |dF| where difference weights fall off", minimum=0.0, maximum=100.0, upstream="add_phases_and_blobs.absdF_pct"),
            "extrapolate_factors": Field("list_num", [2, 4, 6, 8, 16], "extrapolation factors written to the phased MTZs", upstream="add_phases_and_blobs.extrapolate_factors"),
            "cutoff": Field("float", 3.5, "blob threshold (sigma) on the blurred |difference map|", minimum=0.0, upstream="add_phases_and_blobs.cutoff"),
            "radius_in_A": Field("float", 4.0, "Gaussian blur radius (A)", minimum=0.0, upstream="add_phases_and_blobs.radius_in_A"),
        },
        "refine": {
            "eff": Field("path", None, "phenix parameter file (null = the packaged refine_drug.eff)", nullable=True),
            "extra_args": Field("list_str", [], "extra command-line arguments for the refinement program"),
            "timeout_min": Field("int", 60, "give up on one dataset's refinement after this many minutes", minimum=1),
        },
        "report": {
            "top_n": Field("int", 20, "number of top hits with maps and viewer entries", minimum=1),
            "maps": Field("bool", True, "write map files for the top hits"),
        },
    },
}

PATH_KINDS = ("path", "pattern")


def iter_fields(schema=SCHEMA, prefix=""):
    """Yield (dotted key, Field) for every leaf of the schema, in file order."""
    for key, spec in schema.items():
        dotted = f"{prefix}{key}"
        if isinstance(spec, dict):
            yield from iter_fields(spec, dotted + ".")
        else:
            yield dotted, spec


def default_project():
    """A nested dict with every default filled in."""
    def build(schema):
        out = {}
        for key, spec in schema.items():
            out[key] = build(spec) if isinstance(spec, dict) else copy.deepcopy(spec.default)
        return out
    return build(SCHEMA)


def get(project, dotted, default=None):
    node = project
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(project, dotted, value):
    node = project
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_project(path):
    """Load and validate valdo.yaml. Returns (project, notes); raises ProjectError.

    The returned project has every key, with path fields made absolute and the extra keys
    '_path' (absolute project file) and '_dir' (its folder).
    """
    import yaml
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise ProjectError([f"project file not found: {path}. Create one with 'valdo init DATA --apo MODEL.pdb' "
                            "(or --models PATTERN)."], path, kind="missing")
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ProjectError([_yaml_problem(path, exc)], path, kind="syntax", details=[f"YAML parser: {exc}"])
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProjectError([f"{path} must contain 'key: value' settings, not a {type(raw).__name__}"], path,
                           kind="syntax")
    project, notes = validate_project(raw, os.path.dirname(path))
    project["_path"] = path
    project["_dir"] = os.path.dirname(path)
    return project, notes


def validate_project(raw, base_dir):
    """Validate a raw (parsed YAML) dict against SCHEMA. Returns (project, notes)."""
    errors, notes = [], []
    project = default_project()
    _merge(raw, SCHEMA, project, "", errors, notes)
    if errors:
        raise ProjectError(errors)
    for dotted, spec in iter_fields():
        value = get(project, dotted)
        if spec.required and value in (None, ""):
            errors.append(f"{dotted}: required. {spec.doc}.")
        if spec.kind in PATH_KINDS and value not in (None, ""):
            set_value(project, dotted, expand_path(value, base_dir))
    errors.extend(_cross_checks(project))
    if errors:
        raise ProjectError(errors)
    return project, notes


def _all_leaf_keys():
    return {dotted.split(".")[-1]: dotted for dotted, _ in iter_fields()}


def _suggest(key, valid, prefix):
    close = difflib.get_close_matches(key, valid, n=1, cutoff=0.6)
    if close:
        return f" Did you mean '{prefix}{close[0]}'?"
    leaf = _all_leaf_keys()
    if key in leaf:
        return f" Did you mean '{leaf[key]}'?"
    close = difflib.get_close_matches(key, list(leaf), n=1, cutoff=0.75)
    if close:
        return f" Did you mean '{leaf[close[0]]}'?"
    return f" Valid keys here: {', '.join(valid)}."


def _merge(raw, schema, out, prefix, errors, notes):
    for key, value in raw.items():
        key = str(key)
        dotted = f"{prefix}{key}"
        if key not in schema:
            errors.append(f"{dotted}: unknown setting.{_suggest(key, list(schema), prefix)}")
            continue
        spec = schema[key]
        if isinstance(spec, dict):
            if value is None:
                continue
            if not isinstance(value, dict):
                errors.append(f"{dotted}: must be a section with keys {', '.join(spec)} (got {value!r})")
                continue
            _merge(value, spec, out[key], dotted + ".", errors, notes)
        else:
            ok, converted, message = coerce(value, spec)
            if not ok:
                errors.append(f"{dotted}: {message}")
            else:
                if message:
                    notes.append(f"{dotted}: {message}")
                out[key] = converted


def _number(value):
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError


def coerce(value, spec):
    """Check/convert one value. Returns (ok, value, message); message is a note if ok."""
    kind = spec.kind
    if value is None:
        if spec.nullable or spec.default is None:
            return True, None, ""
        return False, None, f"must not be empty (default: {format_scalar(spec.default)})"
    note = ""
    try:
        if kind in ("str", "path", "pattern"):
            if isinstance(value, (dict, list)):
                return False, value, f"must be text, got {type(value).__name__}"
            value = str(value)
        elif kind == "bool":
            if not isinstance(value, bool):
                if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "false", "no", "off"):
                    value = value.strip().lower() in ("true", "yes", "on")
                    note = f"read '{value}' as a true/false value"
                else:
                    return False, value, f"must be true or false, got {value!r}"
        elif kind == "int":
            number = _number(value)
            if isinstance(number, float):
                if not number.is_integer():
                    return False, value, f"must be a whole number, got {value!r}"
                note = f"read {value!r} as the whole number {int(number)}"
            value = int(number)
        elif kind == "float":
            number = _number(value)
            if isinstance(value, str):
                note = f"YAML read {value!r} as text; using the number {float(number)!r}"
            value = float(number)
        elif kind == "reindex":
            if isinstance(value, bool):
                value = "true" if value else "false"
            value = str(value).strip().lower()
            value = {"yes": "true", "no": "false", "on": "true", "off": "false"}.get(value, value)
            if value not in ("auto", "true", "false"):
                return False, value, "must be auto, true or false"
        elif kind == "when_opt":
            if isinstance(value, str) and value.strip().lower() in ("all", "never"):
                value = value.strip().lower()
            else:
                number = _number(value)
                if not isinstance(value, float):
                    note = f"read {value!r} as the number {float(number)!r} (upstream needs a float here)"
                value = float(number)
                if not 0.0 <= value <= 1.0:
                    return False, value, "must be \"all\", \"never\" or a number between 0 and 1"
        elif kind in ("list_int", "list_num", "list_str"):
            if not isinstance(value, (list, tuple)):
                return False, value, f"must be a list like [a, b], got {value!r}"
            items = []
            for item in value:
                if kind == "list_str":
                    items.append(str(item))
                else:
                    number = _number(item)
                    if kind == "list_int":
                        if float(number) != int(number):
                            return False, value, f"must contain whole numbers, got {item!r}"
                        number = int(number)
                    items.append(number)
            value = items
        else:
            raise AssertionError(kind)
    except (ValueError, TypeError):
        return False, value, f"must be a {_kind_name(kind)}, got {value!r}"
    if spec.choices is not None and value not in spec.choices:
        close = difflib.get_close_matches(str(value), spec.choices, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        return False, value, f"must be one of {', '.join(spec.choices)}, got {value!r}.{hint}"
    if kind in ("int", "float") and value is not None:
        if spec.minimum is not None and value < spec.minimum:
            return False, value, f"must be at least {spec.minimum}, got {value}"
        if spec.maximum is not None and value > spec.maximum:
            return False, value, f"must be at most {spec.maximum}, got {value}"
    return True, value, note


def _kind_name(kind):
    return {"int": "whole number", "float": "number", "list_int": "list of whole numbers",
            "list_num": "list of numbers", "when_opt": "\"all\", \"never\" or number"}.get(kind, kind)


def _cross_checks(p):
    errors = []
    models = p["models"]
    if not models["apo"] and not models["per_dataset"]:
        errors.append("models: give models.apo (one PDB for all datasets) and/or models.per_dataset "
                      "(one model per dataset); blob detection and tagging need a model for every dataset.")
    phases = p["phases"]
    if phases["source"] == "existing" and not phases["existing"]:
        errors.append("phases.existing: required when phases.source is 'existing' (your refined MTZs, "
                      "e.g. '/path/refined/{name}.mtz').")
    if phases["source"] == "refine" and not phases["program"]:
        errors.append("phases.program: required when phases.source is 'refine' (phenix or dimple).")
    if phases["source"] in ("input_mtz", "existing") and (not phases["columns"]["phwt"] or not phases["columns"]["phdelwt"]):
        errors.append("phases.columns.phwt / phdelwt: the 2mFo-DFc and mFo-DFc phase column names are needed "
                      f"for phases.source '{phases['source']}'.")
    t = p["advanced"]["train"]
    if not 0.0 < t["train_fraction"] < 1.0:
        errors.append(f"advanced.train.train_fraction: must be between 0 and 1 (exclusive), got {t['train_fraction']}; "
                      "upstream training crashes with an empty validation set. The validated value is 0.8.")
    if len(t["n_hidden_layers"]) != 2 or min(t["n_hidden_layers"]) < 1:
        errors.append("advanced.train.n_hidden_layers: must be two positive whole numbers [encoder, decoder], e.g. [3, 6].")
    if not t["include_errors"]:
        errors.append("advanced.train.include_errors: must be true; upstream preprocess crashes without sigma(F).")
    r = p["advanced"]["reconstruct"]
    if r["ml_recon"] and r["repeats"] > 1:
        errors.append("advanced.reconstruct.repeats: values above 1 need ml_recon: false "
                      "(upstream reconstruct crashes with ml_recon: true and repeats > 1).")
    if p["advanced"]["reindex"]["cc_min_dif"] <= 0:
        errors.append("advanced.reindex.cc_min_dif: must be above 0 (upstream writes no files otherwise).")
    if not p["advanced"]["blobs"]["extrapolate_factors"]:
        errors.append("advanced.blobs.extrapolate_factors: give at least one factor, e.g. [2, 4, 6, 8, 16].")
    return errors


# ---------------------------------------------------------------------------
# Writing valdo.yaml with comments
# ---------------------------------------------------------------------------

def format_scalar(value):
    """A YAML scalar that PyYAML reads back as the same value (floats always keep a '.')."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if "e" in text or "E" in text:
            mantissa, exponent = text.lower().split("e")
            if "." not in mantissa:
                mantissa += ".0"
            text = f"{mantissa}e{exponent}"
        elif "." not in text and "inf" not in text and "nan" not in text:
            text += ".0"
        return text
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    return json.dumps(str(value))


_SECTIONS = [
    ("1. YOUR PROJECT  (please check these)", ["data", "models", "workdir"]),
    ("2. DETECTED BY 'valdo init'  (change only if something is wrong)",
     ["columns", "phases", "reindex", "reference", "focal_residue", "validation", "compute"]),
    ("3. ADVANCED  (validated settings from the PTP1B screen; change only if you know why)",
     ["advanced"]),
]


def render_project(project, header_lines=()):
    """Return the text of a commented valdo.yaml for `project` (a nested dict)."""
    from valdo import __version__
    lines = [
        "# valdo.yaml: VALDO project file",
        f"# written by valdo {__version__} on {datetime.date.today().isoformat()}",
        "#",
        "# Next steps:   valdo check     (explains any problem with data, settings or computer)",
        "#               valdo run       (runs everything; resumable; results in 'workdir')",
        "# Every setting has a one-line explanation. Unknown or misspelled keys are reported as errors.",
    ]
    for line in header_lines:
        lines.append(f"# {line}" if line else "#")
    for title, keys in _SECTIONS:
        lines += ["", "# " + "=" * 78, f"# {title}", "# " + "=" * 78]
        for key in keys:
            _render(lines, key, SCHEMA[key], project.get(key), 0)
    return "\n".join(lines) + "\n"


def _render(lines, key, spec, value, indent):
    pad = "  " * indent
    if isinstance(spec, dict):
        lines.append(f"{pad}{key}:")
        value = value or {}
        for sub_key, sub_spec in spec.items():
            _render(lines, sub_key, sub_spec, value.get(sub_key), indent + 1)
        return
    text = f"{pad}{key}: {format_scalar(value)}"
    lines.append(f"{text:<44} # {spec.doc}" if spec.doc else text)


def write_project(project, path, header_lines=()):
    text = render_project(project, header_lines)
    # Round-trip check: what we write must load back to the same settings.
    import yaml
    reloaded = yaml.safe_load(text)
    for dotted, spec in iter_fields():
        if get(reloaded, dotted) != get(project, dotted):
            raise AssertionError(f"valdo.yaml round trip changed {dotted}")
    with open(path, "w") as handle:
        handle.write(text)
    return path
