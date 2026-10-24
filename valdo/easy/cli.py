"""`valdo`: one command for the whole VALDO pipeline.

    valdo init DATA --apo apo.pdb      write valdo.yaml (detects columns, phases, indexing)
    valdo check                        explain anything wrong with data, settings or computer
    valdo run                          run every stage (resumable); results in 'workdir'
    valdo status                       what has run, what is out of date, where datasets went
    valdo report                       (re)build <workdir>/report/report.html
    valdo jobscript                    batch script (SGE/SLURM) that runs 'valdo run'
    valdo stages                       the stage table

Every command takes --json: stdout becomes JSON Lines whose last line is the result
({"schema": "valdo-easy/1", "command", "ok", "next", ...}); human text goes to stderr.
Exit codes: 0 ok, 1 a problem with data/settings/a stage (the message says what, why and how
to fix), 2 an internal error in valdo-easy (please report), 130 interrupted.
"""

import argparse
import math
import os
import sys
import time
import traceback

from valdo.easy import SCHEMA_ID, UPSTREAM_BASE, UPSTREAM_VERSION, __version__, version_string

EXIT_OK, EXIT_PROBLEM, EXIT_INTERNAL, EXIT_INTERRUPTED = 0, 1, 2, 130


class CliError(Exception):
    def __init__(self, code, message, what="", cause="", fix="", exit_code=EXIT_PROBLEM, details=None, log=None):
        super().__init__(message)
        self.code, self.message, self.what, self.cause, self.fix = code, message, what or message, cause, fix
        self.exit_code, self.details, self.log = exit_code, details or [], log

    def as_dict(self):
        return {"code": self.code, "message": self.message, "what": self.what, "cause": self.cause,
                "fix": self.fix, "log": self.log, "details": self.details}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _project_args(p):
    p.add_argument("project", nargs="?", default=None, help="project file (default: ./valdo.yaml)")
    p.add_argument("-c", "--config", dest="config", default=None, help="project file (same as PROJECT)")


def _json_arg(p):
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output (JSON Lines on stdout; last line is the result)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="valdo", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="VALDO: find ligand-bound datasets in a crystallographic fragment screen.",
        epilog="Start here:\n  valdo init /path/to/mtz_folder --apo apo.pdb\n  valdo check\n  valdo run\n\n"
               "Upstream per-stage tools still work: valdo.pipeline, valdo.refine, valdo.occupancy.")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--json", action="store_true", default=False, help="machine-readable output")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("init", help="create valdo.yaml from your data (fast: reads only MTZ headers)",
                       description="Scan the MTZ files, detect columns, phases, space group and indexing, and "
                                   "write a commented valdo.yaml.")
    p.add_argument("data", nargs="?", help="folder with MTZ files, a quoted glob, or a .txt list of paths")
    p.add_argument("--apo", help="one apo model (PDB) for every dataset (also the fallback model)")
    p.add_argument("--models", help="per-dataset models: pattern with {name}, folder or glob")
    p.add_argument("-o", "--output", default="valdo.yaml", help="project file to write (default: valdo.yaml)")
    p.add_argument("--workdir", help="where results go (default: valdo_run next to the project file)")
    p.add_argument("--phases", choices=["auto", "input_mtz", "refine", "existing"], default="auto",
                   help="where phases come from (default: auto = the data files if they have map coefficients)")
    p.add_argument("--existing", help="your refined MTZs (pattern with {name}, folder or glob) for --phases existing")
    p.add_argument("--program", choices=["phenix", "dimple"], help="refinement program for --phases refine")
    p.add_argument("--columns", help="amplitude and sigma columns, e.g. F,SIGF (default: detected)")
    p.add_argument("--reindex", choices=["auto", "true", "false"], default="auto",
                   help="reindex to a common setting (default: auto = if the lattice allows alternatives)")
    p.add_argument("--reference", help="reference dataset name (default: auto = first usable dataset)")
    p.add_argument("--focal-seqid", type=int, help="residue number to exclude blobs around (e.g. 215)")
    p.add_argument("--focal-radius", type=float, help="exclusion radius in A (default 5)")
    p.add_argument("--bound-models", help="known ligand-bound models for validation (pattern, folder or glob)")
    p.add_argument("--ligand-resnames", help="ligand residue names in the bound models, comma separated (default LIG)")
    p.add_argument("--ncpu", type=int, help="worker processes (default: CPUs available here, at most 16)")
    p.add_argument("--force", action="store_true", help="overwrite an existing project file")
    _json_arg(p)

    p = sub.add_parser("check", help="check data, settings and computer; explain every problem")
    _project_args(p)
    p.add_argument("--deep", action="store_true", help="also read all reflection data (slower)")
    _json_arg(p)

    p = sub.add_parser("run", help="run all stages that are not up to date (resumable)")
    _project_args(p)
    p.add_argument("--from", dest="from_", metavar="STAGE", help="rerun from this stage on")
    p.add_argument("--until", metavar="STAGE", help="stop after this stage")
    p.add_argument("--only", metavar="STAGE", help="run only this stage (earlier stages must be done)")
    p.add_argument("--force", action="store_true", help="rerun the selected stages even if up to date")
    p.add_argument("--dry-run", action="store_true", help="show what would run; change nothing")
    p.add_argument("--verbose", action="store_true", help="also show each stage's output")
    _json_arg(p)

    p = sub.add_parser("status", help="what has run, what is out of date, where datasets were dropped")
    _project_args(p)
    _json_arg(p)

    p = sub.add_parser("report", help="(re)build <workdir>/report/")
    _project_args(p)
    p.add_argument("--top", type=int, help="number of top hits with maps (default: advanced.report.top_n)")
    p.add_argument("--no-maps", action="store_true", help="do not write map files")
    _json_arg(p)

    p = sub.add_parser("jobscript", help="write a batch script (SGE or SLURM) that runs 'valdo run'")
    _project_args(p)
    p.add_argument("--scheduler", choices=["sge", "slurm"], help="default: detected from the commands on PATH")
    p.add_argument("--slots", type=int, help="CPU cores (default: compute.ncpu)")
    p.add_argument("--hours", type=float, default=None,
                   help="wall-clock limit (default: estimated from the number of datasets, cores and "
                        "phases.source, about 3x the time measured on Wynton CPUs)")
    p.add_argument("--mem-gb", type=float, default=4.0, help="memory per core in GB (default 4)")
    p.add_argument("--gpu", action="store_true", help="request one GPU")
    p.add_argument("--env-setup", action="append", metavar="CMD",
                   help="shell line for the job's environment (repeatable). Default: activate the current one. "
                        "Lines without 'activate' (e.g. CCP4's setup script) are added after that activation; "
                        "lines with 'activate' replace it")
    p.add_argument("--log-dir", help="folder for the job log (default: <workdir>/logs)")
    p.add_argument("-o", "--output", help="write the script here instead of printing it")
    _json_arg(p)

    p = sub.add_parser("stages", help="list the pipeline stages")
    _json_arg(p)
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Project file of this invocation, when it is not ./valdo.yaml: suggested next commands
# ("next" in the result) then name it, so they work when copied as they are.
_PROJECT_FOR_NEXT = {"path": None}
_PROJECT_COMMANDS = ("check", "run", "status", "report", "jobscript")


def _remember_project(path):
    default = os.path.abspath("valdo.yaml")
    _PROJECT_FOR_NEXT["path"] = None if os.path.abspath(path) == default else os.path.abspath(path)


def _with_project(commands):
    """Append the project file to suggested 'valdo <check|run|status|report|jobscript>' commands."""
    import shlex
    path = _PROJECT_FOR_NEXT["path"]
    if not path:
        return list(commands)
    out = []
    for cmd in commands:
        parts = cmd.split()
        if len(parts) >= 2 and parts[0] == "valdo" and parts[1] in _PROJECT_COMMANDS and path not in cmd:
            cmd = f"{cmd} -c {shlex.quote(path)}"
        out.append(cmd)
    return out


def _project_path(args):
    if args.project and args.config and os.path.abspath(args.project) != os.path.abspath(args.config):
        raise CliError("bad_options", f"Two project files given ({args.project}, {args.config}).",
                       fix="Give the project file once.")
    path = os.path.abspath(os.path.expanduser(args.project or args.config or "valdo.yaml"))
    _remember_project(path)
    return path


def _load(args, console):
    from valdo.easy.project import ProjectError, load_project
    path = _project_path(args)
    try:
        project, notes = load_project(path)
    except ProjectError as exc:
        kind = getattr(exc, "kind", "settings")
        if kind == "missing":
            cause, fix = "", ("Create it with 'valdo init DATA --apo MODEL.pdb' (or --models PATTERN), or give "
                              "the right file with -c FILE.")
        elif kind == "syntax":
            cause = "A YAML syntax mistake, usually indentation, a misplaced ':' or a tab character."
            fix = "Correct the place named above; 'valdo init' writes a correctly formatted, commented example."
        else:
            unknown = any("unknown setting" in m for m in exc.messages)
            cause = ("valdo.yaml settings are checked strictly so that typos do not go unnoticed."
                     if unknown else "")
            fix = "Correct the settings listed above (docs/easy/CONFIG_REFERENCE.md lists every setting)."
        raise CliError("project_invalid", f"{path} has {len(exc.messages)} problem(s).", cause=cause, fix=fix,
                       details=exc.messages + list(getattr(exc, "details", []) or []))
    for note in notes:
        console.say(f"note: {note}")
    return project, notes


def _ncpu_override(project, console):
    value = os.environ.get("VALDO_NCPU", "").strip()
    if value:
        try:
            n = int(value)
            if n >= 1:
                if n != project["compute"]["ncpu"]:
                    console.say(f"note: VALDO_NCPU={n} overrides compute.ncpu ({project['compute']['ncpu']})")
                project["compute"]["ncpu"] = n
        except ValueError:
            console.say(f"note: ignoring VALDO_NCPU={value!r} (not a whole number)")
    return project["compute"]["ncpu"]


def _say_error(console, err):
    console.warn("")
    console.warn(f"ERROR: {err['message']}")
    for item in err.get("details") or []:
        console.warn(f"  - {item}")
    if err.get("what") and err["what"] != err["message"]:
        console.warn(f"  What went wrong: {err['what']}")
    if err.get("cause"):
        console.warn(f"  Likely cause:    {err['cause']}")
    if err.get("fix"):
        console.warn(f"  How to fix:      {err['fix']}")
    if err.get("log"):
        console.warn(f"  Log:             {err['log']}")


def _prompt(text):
    try:
        return input(f"{text}: ").strip() or None
    except EOFError:
        return None


def _say_findings(console, findings, show_info=True):
    labels = {"error": "ERROR", "warning": "WARN ", "info": "info "}
    for level in ("error", "warning", "info"):
        for f in findings:
            if f["level"] != level or (level == "info" and not show_info):
                continue
            console.say(f"{labels[level]} {f['title']}")
            if f.get("what"):
                console.say(f"      {f['what']}")
            if f.get("why"):
                console.say(f"      Why: {f['why']}")
            if f.get("fix"):
                console.say(f"      Fix: {f['fix']}")
            for item in f.get("items") or []:
                console.say(f"        - {item}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_version(args, console):
    console.say(version_string())
    return {"ok": True, "version": __version__, "upstream_version": UPSTREAM_VERSION,
            "upstream_base": UPSTREAM_BASE, "next": []}


def cmd_stages(args, console):
    from valdo.easy.stages import STAGES
    console.say(f"{'#':>2}  {'stage':<21} {'folder':<16} runs when")
    for s in STAGES:
        console.say(f"{s.number:>2}  {s.name:<21} {s.folder:<16} {s.runs_when}")
        console.say(f"    {s.description}")
    return {"ok": True, "stages": [s.describe() for s in STAGES], "next": ["valdo status"]}


def _model_resname(path, seqid):
    try:
        import gemmi
        st = gemmi.read_structure(path)
        for chain in st[0]:
            for res in chain:
                if res.seqid.num == seqid:
                    return res.name
    except Exception:
        return None
    return None


def cmd_init(args, console):
    from valdo.easy import check, scan
    from valdo.easy.project import ProjectError, default_project, load_project, write_project
    cwd = os.getcwd()
    out_path = os.path.abspath(os.path.expanduser(args.output))
    _remember_project(out_path)
    if os.path.exists(out_path) and not args.force:
        raise CliError("project_exists", f"{out_path} already exists.",
                       fix="Use --force to overwrite it, or -o to write another file.")
    tty = sys.stdin.isatty() and not args.json
    data = args.data or (_prompt("Folder (or quoted glob) with your MTZ files") if tty else None)
    if not data:
        raise CliError("missing_data", "No data given.",
                       fix="valdo init /path/to/mtz_folder --apo apo.pdb   (or --models '/models/{name}.pdb')")
    apo, models = args.apo, args.models
    if not apo and not models:
        answer = _prompt("Apo model (PDB) for all datasets, or a pattern with {name} for one model per dataset") \
            if tty else None
        if not answer:
            raise CliError("missing_models", "No model given.",
                           cause="Blob detection and tagging need a model for every dataset.",
                           fix="Add --apo /path/apo.pdb (one model for all) and/or --models '/path/{name}.pdb' "
                               "(one per dataset; a folder works too).")
        if "{name}" in answer or os.path.isdir(os.path.expanduser(answer)) or scan.has_glob(answer):
            models = answer
        else:
            apo = answer
    base = os.path.dirname(out_path)
    project = default_project()
    project["data"] = scan.expand_path(data, cwd)
    project["models"]["apo"] = scan.expand_path(apo, cwd) if apo else None
    project["models"]["per_dataset"] = scan.expand_path(models, cwd) if models else None
    project["workdir"] = scan.expand_path(args.workdir, cwd) if args.workdir else "valdo_run"
    if project["models"]["apo"] and not os.path.isfile(project["models"]["apo"]):
        raise CliError("model_missing", f"The apo model {project['models']['apo']} does not exist.",
                       fix="Check the path given with --apo.")

    t0 = time.time()
    files, note = scan.list_data_files(project["data"], cwd)
    if not files:
        raise CliError("no_data", f"No MTZ files found: {note}.",
                       fix="Give a folder with .mtz files, a quoted glob such as '/data/*/final.mtz', "
                           "or a .txt file with one path per line.")
    console.say(f"Reading {len(files)} MTZ headers ...")
    headers = scan.scan_headers(files, threads=16)
    t_scan = time.time() - t0
    readable = [h for h in headers if h["ok"]]
    if not readable:
        raise CliError("no_readable_data", f"None of the {len(files)} files is a readable MTZ file.",
                       details=[f"{h['path']}: {h['error']}" for h in headers[:5]])
    decisions = []

    # amplitude columns
    if args.columns:
        parts = [c.strip() for c in args.columns.split(",") if c.strip()]
        if len(parts) != 2:
            raise CliError("bad_columns", f"--columns {args.columns}: give two names, e.g. --columns F,SIGF.")
        amp, sig = parts
        n = sum(1 for h in readable if scan.column_types(h).get(amp) == "F" and scan.column_types(h).get(sig) == "Q")
        decisions.append({"setting": "columns", "value": f"{amp}, {sig}",
                          "why": f"given with --columns; present in {n}/{len(readable)} files"})
    else:
        det = scan.detect_amplitude_columns(headers)
        if det["amplitude"] is None:
            extra = (" They contain intensities only: convert them to amplitudes first (French-Wilson, e.g. "
                     "ctruncate or phenix.french_wilson)." if det["intensity_only"] else "")
            raise CliError("no_amplitudes", "No amplitude/sigma column pair (types F and Q) found." + extra,
                           fix="Pass the column names with --columns F,SIGF.")
        amp, sig = det["amplitude"], det["sigma"]
        alt = f"; also found {', '.join(det['alternatives'][:3])}" if det["alternatives"] else ""
        decisions.append({"setting": "columns", "value": f"{amp}, {sig}",
                          "why": f"observed amplitudes with sigmas in {det['n_files']}/{len(readable)} files{alt}"})
    project["columns"]["amplitude"], project["columns"]["sigma"] = amp, sig

    # phases
    detected = scan.detect_phase_columns(headers)
    source = args.phases
    if source == "auto":
        if detected and detected["n_files"] >= 0.5 * len(readable):
            source = "input_mtz"
        elif args.existing:
            source = "existing"
        elif args.program:
            source = "refine"
        else:
            raise CliError("no_phases", "The data files contain no map coefficients (2mFo-DFc and mFo-DFc with phases).",
                           cause="VALDO's difference maps need phases from a refined model of each dataset.",
                           fix="Run init again with --phases refine --program dimple (or phenix), or with "
                               "--phases existing --existing '/path/refined/{name}.mtz'.")
    phase_headers = headers
    if source == "input_mtz":
        if not detected:
            raise CliError("no_phases", "--phases input_mtz, but the data files contain no map coefficients.",
                           fix="Use --phases refine --program dimple|phenix or --phases existing --existing PATTERN.")
        why = (f"map coefficients {detected['phwt']}/{detected['phdelwt']} ({detected['convention']} names) "
               f"in {detected['n_files']}/{len(readable)} data files")
        cols = detected
    elif source == "existing":
        if not args.existing:
            raise CliError("missing_existing", "--phases existing needs --existing PATTERN.",
                           fix="e.g. --existing '/path/refined/{name}/final.mtz'")
        project["phases"]["existing"] = scan.expand_path(args.existing, cwd)
        names = scan.dataset_names(files)
        matched, problems = scan.match_files_to_names(project["phases"]["existing"], names, cwd,
                                                      extensions=scan.MTZ_EXTENSIONS)
        if not matched:
            raise CliError("existing_unmatched", f"--existing {args.existing} matched none of the dataset names.",
                           details=problems[:5], fix="Use a pattern with {name}, e.g. '/refined/{name}.mtz'.")
        phase_headers = scan.scan_headers(sorted(matched.values()), threads=16)
        cols = scan.detect_phase_columns(phase_headers)
        if not cols:
            raise CliError("existing_no_phases", "The refined MTZs contain no map coefficients that valdo recognises.",
                           fix="Set phases.columns in valdo.yaml by hand after init (fwt, phwt, delfwt, phdelwt).")
        why = f"your refined MTZs ({len(matched)} matched); map coefficients {cols['phwt']}/{cols['phdelwt']}"
    else:  # refine
        if not args.program:
            raise CliError("missing_program", "--phases refine needs --program phenix or dimple.")
        project["phases"]["program"] = args.program
        try:
            from valdo.easy.refine import MAP_COEFFICIENTS
            (fwt, phwt), (delfwt, phdelwt) = MAP_COEFFICIENTS[args.program]
        except Exception:
            fwt, phwt, delfwt, phdelwt = (("2FOFCWT", "PH2FOFCWT", "FOFCWT", "PHFOFCWT") if args.program == "phenix"
                                          else ("FWT", "PHWT", "DELFWT", "PHDELWT"))
        cols = {"fwt": fwt, "phwt": phwt, "delfwt": delfwt, "phdelwt": phdelwt}
        why = f"every dataset is refined with {args.program} (phases.program)"
        phase_headers = []
    project["phases"]["source"] = source
    for key in ("fwt", "phwt", "delfwt", "phdelwt"):
        project["phases"]["columns"][key] = cols.get(key)
    project["phases"]["rfree"] = scan.detect_rfree_column(phase_headers) if phase_headers else None
    decisions.append({"setting": "phases.source", "value": source, "why": why})

    # focal residue, validation, compute
    if args.focal_seqid is not None:
        project["focal_residue"]["seqid"] = args.focal_seqid
        model_for_name = project["models"]["apo"]
        if not model_for_name and project["models"]["per_dataset"]:
            found, _ = scan.match_files_to_names(project["models"]["per_dataset"], scan.dataset_names(files)[:20], cwd)
            model_for_name = next(iter(found.values()), None)
        resname = _model_resname(model_for_name, args.focal_seqid) if model_for_name else None
        project["focal_residue"]["tag"] = f"{resname.lower()}{args.focal_seqid}" if resname else f"res{args.focal_seqid}"
        decisions.append({"setting": "focal_residue", "value": f"{args.focal_seqid} ({resname or 'residue name unknown'})",
                          "why": "given with --focal-seqid; blobs within the radius are not reported as hits"})
    if args.focal_radius is not None:
        project["focal_residue"]["radius"] = float(args.focal_radius)
    if args.bound_models:
        project["validation"]["bound_models"] = scan.expand_path(args.bound_models, cwd)
    if args.ligand_resnames:
        project["validation"]["ligand_resnames"] = [x.strip() for x in args.ligand_resnames.split(",") if x.strip()]
    available, _ = check.cpu_count()
    project["compute"]["ncpu"] = args.ncpu or max(1, min(available, 16))
    project["reindex"] = args.reindex
    project["reference"] = args.reference or "auto"

    # validate, then inspect like 'valdo check' does, to report reference and reindex decisions
    from valdo.easy.project import validate_project
    try:
        validated, _ = validate_project(_plain(project), base)
    except ProjectError as exc:
        raise CliError("project_invalid", "The settings given to init are inconsistent.", details=exc.messages)
    validated["_dir"], validated["_path"] = base, out_path
    inputs = check.inspect_inputs(validated)
    ok_rows = inputs.ok_rows()
    s = inputs.summary
    if inputs.reference:
        decisions.append({"setting": "reference", "value": f"{inputs.reference['name']} ({inputs.reference['id']})",
                          "why": inputs.reference["why"]})
    if inputs.reindex:
        decisions.append({"setting": "reindex", "value": f"{args.reindex} -> {'on' if inputs.reindex['enabled'] else 'off'}",
                          "why": inputs.reindex["why"]})
    decisions.append({"setting": "compute.ncpu", "value": project["compute"]["ncpu"],
                      "why": f"{'given with --ncpu' if args.ncpu else f'{available} CPU(s) available here (max 16)'}; "
                             "use your batch job's slot count"})
    summary = {
        "n_files": len(files), "n_usable": len(ok_rows), "n_excluded": len(inputs.rows) - len(ok_rows),
        "space_group": s.get("space_group"), "median_cell": s.get("median_cell"),
        "d_min_range": s.get("d_min_range"), "scan_seconds": round(t_scan, 2),
        "n_with_model": sum(1 for r in ok_rows if r["model"]),
        "n_with_phases": sum(1 for r in ok_rows if r["phase_file"]) if source != "refine" else None,
        "n_bound_models": sum(1 for r in ok_rows if r["bound_model"]),
    }
    header = [f"Detected by 'valdo init' on {time.strftime('%Y-%m-%d %H:%M')}: {len(files)} MTZ files "
              f"({len(ok_rows)} usable), {s.get('space_group')}, cell {scan.format_cell(s.get('median_cell'))}"]
    if s.get("d_min_range"):
        header.append(f"high-resolution limits {s['d_min_range'][0]:.2f}-{s['d_min_range'][1]:.2f} A")
    for d in decisions:
        header.append(f"{d['setting']}: {d['value']}  ({d['why']})")
    write_project(_plain(project), out_path, header_lines=header)
    try:
        load_project(out_path)
    except ProjectError as exc:  # pragma: no cover - write_project checks the round trip
        raise CliError("internal", "valdo init wrote a project file it cannot read back.", details=exc.messages,
                       exit_code=EXIT_INTERNAL)

    warnings = [f["title"] + ": " + f["what"] for f in inputs.findings if f["level"] != "info"]
    pattern = check.model_pattern_finding(validated, inputs)
    if pattern:   # --models matched nothing: say so now, not only in 'valdo check'
        warnings.append(f"{pattern['title']}: {pattern['what']} ({'; '.join(pattern['items'])})")
    excluded = [r for r in inputs.rows if r["status"] != "ok"]
    if excluded:
        warnings.append(f"{len(excluded)} file(s) cannot be used ('valdo check' lists them)")
    console.say(f"Read {len(files)} MTZ headers in {t_scan:.1f} s: {len(ok_rows)} usable, {s.get('space_group')}, "
                f"cell {scan.format_cell(s.get('median_cell'))}")
    for d in decisions:
        console.say(f"  {d['setting']:<14} {d['value']}  ({d['why']})")
    for w in warnings:
        console.say(f"  warning: {w}")
    console.say(f"Wrote {out_path}")
    chk, run, job = _with_project(["valdo check", "valdo run", "valdo jobscript -o run_valdo.sh"])
    console.say(f"Next: '{chk}' (explains any problem), then '{run}' "
                f"(or '{job}' to run it as a batch job).")
    return {"ok": True, "project_file": out_path, "n_datasets": len(ok_rows), "decisions": decisions,
            "summary": summary, "warnings": warnings, "next": ["valdo check", "valdo run"]}


def _plain(project):
    return {k: v for k, v in project.items() if not k.startswith("_")}


def cmd_check(args, console):
    from valdo.easy import check
    project, notes = _load(args, console)
    console.say(f"Checking {project['_path']} ...")
    inputs = check.inspect_inputs(project)
    report = check.run_checks(project, inputs, deep=args.deep)
    _say_findings(console, report["findings"])
    errors, warnings = report["errors"], report["warnings"]
    usable = report["inputs"]["n_usable"]
    console.say("")
    console.say(f"valdo check: {errors} error(s), {warnings} warning(s); {usable} of "
                f"{report['inputs']['n_files']} datasets usable.")
    if errors:
        console.say("Fix the errors above, then run 'valdo check' again.")
        nxt = ["valdo check"]
    else:
        env = report["environment"]
        if env["login_node"]:   # never run the pipeline on a login node: write a job script and submit it
            submit = "qsub" if _detect_scheduler() == "sge" else "sbatch"
            nxt = ["valdo jobscript -o run_valdo.sh", f"{submit} run_valdo.sh"]
            console.say("Ready. This is a login node, so run it as a batch job: "
                        + ", then ".join(f"'{n}'" for n in _with_project(nxt)) + ".")
        else:
            nxt = ["valdo run"]
            console.say("Ready: " + " or ".join(f"'{n}'" for n in _with_project(nxt)) + ".")
    result = {"ok": errors == 0, "project_file": project["_path"], "errors": errors, "warnings": warnings,
              "findings": report["findings"], "environment": report["environment"], "inputs": report["inputs"],
              "notes": notes, "next": nxt}
    if errors:
        titles = [f["title"] for f in report["findings"] if f["level"] == "error"]
        result["error"] = {"code": "check_failed", "message": f"valdo check found {errors} error(s).",
                           "what": "; ".join(titles), "cause": "", "log": None, "details": titles,
                           "fix": "Each error above says how to fix it (findings[].fix in JSON); then run 'valdo check' again."}
    return result


THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _limit_threads(ncpu):
    """Numerical libraries start one thread per core of the machine unless told otherwise; keep them
    within the CPUs valdo may use (only effective before numpy/torch are imported; a value the user
    set is kept)."""
    for var in THREAD_VARIABLES:
        os.environ.setdefault(var, str(ncpu))


def cmd_run(args, console):
    from valdo.easy import check
    from valdo.easy.runner import Runner, RunError
    project, notes = _load(args, console)
    ncpu = _ncpu_override(project, console)
    _limit_threads(ncpu)
    inputs = check.inspect_inputs(project)
    report = check.run_checks(project, inputs, deep=False)
    errors = [f for f in report["findings"] if f["level"] == "error"]
    if errors:
        _say_findings(console, errors, show_info=False)
        raise CliError("check_failed", f"valdo check found {len(errors)} error(s); nothing was run.",
                       fix="Fix them (see above; 'valdo check' explains each) and run again.",
                       details=[f["title"] for f in errors])
    if report["warnings"]:
        console.say(f"note: valdo check reported {report['warnings']} warning(s) ('valdo check' explains them); continuing.")
    runner = Runner(project, inputs, console, ncpu=ncpu, verbose=args.verbose)
    base = {"project_file": project["_path"], "workdir": project["workdir"], "dry_run": bool(args.dry_run)}
    if args.dry_run:
        plan = runner.plan(args.from_, args.until, args.only, args.force)
        console.say(f"Plan for {project['workdir']} (nothing is changed):")
        for p in plan:
            console.say(f"  {p['number']:>2} {p['stage']:<21} {p['action']:<15} {p['reason']}")
        return dict(base, ok=True, plan=plan, stages=[], datasets=None, report=None,
                    next=["valdo run"])
    try:
        stages = runner.run(args.from_, args.until, args.only, args.force)
    except RunError as exc:
        raise CliError(exc.code, exc.message, what=exc.what, cause=exc.cause, fix=exc.fix, log=exc.log,
                       exit_code=exc.exit_code)
    datasets = _dataset_counts(project["workdir"])
    html = os.path.join(project["workdir"], "report", "report.html")
    st = runner.state["stages"].get("report") or {}
    report_path = html if os.path.isfile(html) and st.get("status") == "done" else None
    console.say("")
    if datasets:
        console.say(f"Datasets: {datasets['ok']} of {datasets['total']} went through; {datasets['dropped']} dropped "
                    f"(reasons in {os.path.join(project['workdir'], 'datasets.csv')}).")
    if report_path:
        console.say(f"Report: {report_path}")
        nxt = [f"open {report_path}"]
    else:
        nxt = ["valdo run", "valdo status"]
    return dict(base, ok=True, plan=[], stages=stages, datasets=datasets, report=report_path, next=nxt)


def _dataset_counts(workdir):
    from valdo.easy.ids import read_datasets_csv
    path = os.path.join(workdir, "datasets.csv")
    if not os.path.isfile(path):
        return None
    rows = read_datasets_csv(path)
    ok = sum(1 for r in rows if r["status"] == "ok")
    return {"total": len(rows), "ok": ok, "dropped": len(rows) - ok}


def cmd_status(args, console):
    from valdo.easy import check
    from valdo.easy.runner import duration, status_summary
    project, _ = _load(args, console)
    try:
        inputs = check.inspect_inputs(project)
    except Exception:
        inputs = None
    s = status_summary(project, inputs)
    console.say(f"valdo status: {project['_path']} -> {s['workdir']}")
    if not s["has_state"]:
        console.say("Nothing has run yet. Next: 'valdo check', then 'valdo run'.")
        return {"ok": True, "project_file": project["_path"], **s, "next": ["valdo check", "valdo run"]}
    console.say(f"  {'#':>2}  {'stage':<21} {'status':<12} {'time':>8}  datasets")
    for st in s["stages"]:
        if not st["applicable"]:
            status = "n/a"
        elif st["stale"]:
            status = "out of date"
        else:
            status = st["status"]
        c = st.get("counts") or {}
        counts = f"{c.get('in')} -> {c.get('out')}" if c.get("in") is not None else ""
        if c.get("dropped"):
            counts += f" ({c['dropped']} dropped)"
        secs = duration(st["seconds"]) if st.get("seconds") is not None else ""
        console.say(f"  {st['number']:>2}  {st['stage']:<21} {status:<12} {secs:>8}  {counts}")
        if st["status"] == "failed" and st.get("error"):
            console.say(f"      {st['error'].get('what')}  (log: {st.get('log')})")
    d = s["datasets"]
    by_stage = ", ".join(f"{k} {v}" for k, v in d["dropped_by_stage"].items())
    console.say(f"Datasets: {d['total']} total, {d['ok']} ok, {d['dropped']} dropped" + (f" ({by_stage})" if by_stage else ""))
    if s["lock"]:
        lk = s["lock"]
        console.say(f"Lock: a run is {'probably dead' if lk.get('stale') else 'active'} "
                    f"(pid {lk.get('pid')} on {lk.get('host')}, started {lk.get('started')})")
    lr = s.get("last_run") or {}
    if lr:
        console.say(f"Last run: started {lr.get('started')}, finished {lr.get('finished') or '-'}, "
                    f"{'ok' if lr.get('ok') else ('failed' if lr.get('ok') is False else 'not finished')}")
    todo = [st for st in s["stages"] if st["applicable"] and (st["stale"] or st["status"] != "done")]
    html = os.path.join(s["workdir"], "report", "report.html")
    result = {"ok": True, "project_file": project["_path"], **s}
    if s["lock"] and not s["lock"].get("stale"):
        # a run is active: starting another one would only stop with 'locked'
        nxt = ["valdo status"]
        result["wait"] = {"reason": "a valdo run is active in this work directory", "poll": "valdo status",
                          "seconds": 60}
        console.say("Next: wait for the active run to finish; check again with "
                    + ", ".join(f"'{n}'" for n in _with_project(nxt)) + " (every minute or so).")
    else:
        nxt = ["valdo run"] if todo else [f"open {html}"]
        console.say("Next: " + ", ".join(_with_project(nxt)))
    return dict(result, next=nxt)


def cmd_report(args, console):
    from valdo.easy import check
    from valdo.easy.runner import Runner, RunError
    project, _ = _load(args, console)
    if args.top is not None:
        project["advanced"]["report"]["top_n"] = max(1, args.top)
    if args.no_maps:
        project["advanced"]["report"]["maps"] = False
    ncpu = _ncpu_override(project, console)
    _limit_threads(ncpu)
    inputs = check.inspect_inputs(project)
    runner = Runner(project, inputs, console, ncpu=ncpu)
    names = [s.name for s, ok, _ in runner.applicable() if ok]
    missing = [n for n in names[:-1] if (runner.state["stages"].get(n) or {}).get("status") != "done"]
    if missing:
        from valdo.easy import report
        console.say(f"note: the pipeline has not finished ({', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''} "
                    "not done); the report shows what exists so far.")
        os.makedirs(project["workdir"], exist_ok=True)
        # bound models already brought into the reference setting by tag_blobs, if it ran before
        bound = os.path.join(project["workdir"], "inputs", "bound_models")
        bound = bound if os.path.isdir(bound) and any(f.endswith(".pdb") for f in os.listdir(bound)) else None
        result = report.build_report(project["workdir"], top_n=project["advanced"]["report"]["top_n"],
                                     known_bound_dir=bound,
                                     ligand_resnames=project["validation"]["ligand_resnames"] or None,
                                     write_maps=project["advanced"]["report"]["maps"])
    else:
        try:
            runner.run(only="report")
        except RunError as exc:
            raise CliError(exc.code, exc.message, what=exc.what, cause=exc.cause, fix=exc.fix, log=exc.log,
                           exit_code=exc.exit_code)
        result = (runner.ctx.scratch.get("report") or {})
    report_dir = os.path.join(project["workdir"], "report")
    html = os.path.join(report_dir, "report.html")
    console.say(f"Report: {html}")
    return {"ok": True, "workdir": project["workdir"], "report_dir": report_dir, "result": result,
            "next": [f"open {html}"]}


def _detect_scheduler():
    import shutil
    if shutil.which("qsub") and (os.environ.get("SGE_ROOT") or not shutil.which("sbatch")):
        return "sge"
    if shutil.which("sbatch"):
        return "slurm"
    return "sge"


def _current_env_setup():
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix) and os.path.isfile(os.path.join(sys.prefix, "bin", "activate")):
        return [f"source {sys.prefix}/bin/activate   # the environment 'valdo jobscript' ran in"]
    conda_prefix, conda_exe = os.environ.get("CONDA_PREFIX"), os.environ.get("CONDA_EXE")
    if conda_prefix and conda_exe:
        root = os.path.dirname(os.path.dirname(conda_exe))
        return [f"source {root}/etc/profile.d/conda.sh", f"conda activate {conda_prefix}"]
    return None


def _job_env_setup(extra):
    """Environment lines for the job script.

    Without --env-setup: activate the environment `valdo jobscript` runs in. With --env-setup lines
    that activate an environment themselves (contain 'activate'): exactly those lines. Otherwise
    (e.g. only CCP4's setup script) the current environment is activated first and the lines follow,
    so that `valdo` stays on PATH in the job.
    """
    current = _current_env_setup()
    if not extra:
        return current
    if current and not any("activate" in line for line in extra):
        return list(current) + list(extra)
    return list(extra)


# Measured on Wynton CPU nodes (16 cores, phases from the data files): 300 PTP1B datasets in 5 min,
# 1,680 in 51 min, i.e. about 30 core-seconds per dataset. dimple took about 1 core-minute per
# dataset; phenix.refine takes several.
JOB_CORE_SECONDS_PER_DATASET = 30
JOB_REFINE_CORE_SECONDS = {"dimple": 60, "phenix": 300}


def suggest_hours(n_datasets, slots, phases_source=None, program=None):
    """Wall-clock limit for a job: 3x the measured time plus 15 minutes, in half hours (at least 0.5)."""
    per = JOB_CORE_SECONDS_PER_DATASET
    if phases_source == "refine":
        per += JOB_REFINE_CORE_SECONDS.get(program or "", 300)
    seconds = 3 * n_datasets * per / max(1, int(slots)) + 900
    return max(0.5, math.ceil(seconds / 1800.0) / 2.0)


def cmd_jobscript(args, console):
    from valdo.easy import jobscript, scan
    path = _project_path(args)
    slots, workdir, project = args.slots, None, None
    if os.path.isfile(path):
        try:
            project, _ = _load(args, console)
            slots = slots or project["compute"]["ncpu"]
            workdir = project["workdir"]
        except CliError:
            console.say(f"note: {path} has problems ('valdo check' explains them); writing the script anyway")
    else:
        console.say(f"note: {path} does not exist yet; create it with 'valdo init' before submitting")
    slots = slots or 16
    hours, hours_why = args.hours, "given with --hours"
    if hours is None:
        n = len(scan.list_data_files(project["data"], project["_dir"])[0]) if project else 0
        if n:
            source = project["phases"]["source"]
            program = project["phases"]["program"] if source == "refine" else None
            hours = suggest_hours(n, slots, source, program)
            hours_why = (f"estimated for {n} datasets on {slots} cores" + (f" with {program} refinement" if program else "")
                         + ": about 3x the time measured on Wynton CPUs; change it with --hours")
        else:
            hours, hours_why = 24.0, "default (no data to estimate from); change it with --hours"
    scheduler = args.scheduler or _detect_scheduler()
    env_setup = _job_env_setup(args.env_setup)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else (os.path.join(workdir, "logs") if workdir else None)
    if args.output and log_dir:
        os.makedirs(log_dir, exist_ok=True)    # the scheduler writes the job log there
    text = jobscript.render(scheduler, path, slots=slots, hours=hours, mem_gb=args.mem_gb, gpu=args.gpu,
                            env_setup=env_setup, log_dir=log_dir)
    out = None
    if args.output:
        out = os.path.abspath(args.output)
        with open(out, "w") as handle:
            handle.write(text)
        os.chmod(out, 0o755)
        submit = "qsub" if scheduler == "sge" else "sbatch"
        console.say(f"Wrote {out} ({scheduler.upper()}, {slots} cores, {hours:g} h: {hours_why}). "
                    f"Submit with: {submit} {out}")
        nxt = [f"{submit} {out}", "valdo status"]
    else:
        if console.json:
            console.say(text)
        else:
            console.out.write(text)
            console.out.flush()
        nxt = ["valdo jobscript -o run_valdo.sh"]
    return {"ok": True, "scheduler": scheduler, "path": out, "script": text, "hours": hours, "hours_why": hours_why,
            "slots": slots, "next": nxt}


COMMANDS = {"init": cmd_init, "check": cmd_check, "run": cmd_run, "status": cmd_status, "report": cmd_report,
            "jobscript": cmd_jobscript, "stages": cmd_stages}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    from valdo.easy.runner import Console, Interrupted, RunError
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) or a usage error (2 -> 1)
        return 0 if exc.code in (0, None) else EXIT_PROBLEM
    json_mode = bool(getattr(args, "json", False))
    console = Console(json_mode=json_mode)
    command = "version" if args.version else (args.command or "help")
    result, exit_code = None, EXIT_OK
    try:
        if args.version:
            result = cmd_version(args, console)
        elif not args.command:
            console.say(parser.format_help())
            result = {"ok": True, "next": ["valdo init DATA --apo apo.pdb"]}
        else:
            result = COMMANDS[args.command](args, console)
        exit_code = EXIT_OK if result.get("ok") else EXIT_PROBLEM
    except CliError as exc:
        err = exc.as_dict()
        _say_error(console, err)
        result = {"ok": False, "error": err, "next": _next_after_error(command, exc.code)}
        exit_code = exc.exit_code
    except RunError as exc:
        err = exc.as_dict()
        _say_error(console, err)
        result = {"ok": False, "error": err, "next": _next_after_error(command, exc.code)}
        exit_code = exc.exit_code
    except (KeyboardInterrupt, Interrupted):
        console.warn("\nInterrupted. Finished stages are kept; 'valdo run' continues from the interrupted stage.")
        result = {"ok": False, "error": {"code": "interrupted", "message": "Interrupted.",
                                         "what": "The command was interrupted (Ctrl-C or a signal from the batch system).",
                                         "cause": "", "fix": "Run 'valdo run' again; finished stages are kept.",
                                         "log": None},
                  "next": ["valdo run"]}
        exit_code = EXIT_INTERRUPTED
    except Exception as exc:  # a bug in valdo-easy
        tb = traceback.format_exc()
        console.warn(tb)
        err = {"code": "internal_error", "message": f"Internal error in valdo-easy: {type(exc).__name__}: {exc}",
               "what": "valdo-easy hit an unexpected error (a bug, not a problem with your data).",
               "cause": "", "fix": "Please report it with the traceback above (and the stage log, if any).",
               "log": None, "traceback": tb}
        _say_error(console, err)
        result = {"ok": False, "error": err, "next": []}
        exit_code = EXIT_INTERNAL
    finally:
        if result is not None:
            final = {"schema": SCHEMA_ID, "command": command}
            final.update(result)
            if isinstance(final.get("error"), dict):   # same keys for every error, whatever raised it
                for key in ("code", "message", "what", "cause", "fix", "log"):
                    final["error"].setdefault(key, None)
            final["next"] = _with_project(final.get("next") or [])
            console.event(**final)
        console.close()
    return exit_code


def _next_after_error(command, code):
    if code in ("project_invalid", "check_failed"):
        return ["valdo check"]
    if code == "locked":   # another run is active: watch it, do not start one
        return ["valdo status"]
    if command == "run":
        return ["valdo status", "valdo run"]
    return []


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
