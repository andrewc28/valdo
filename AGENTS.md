# AGENTS.md: driving VALDO from an LLM agent

For AI coding agents (Claude Code, Codex, ...) asked to run VALDO on a crystallographic
fragment screen, or to work on this repository. Human documentation starts at
[docs/easy/QUICKSTART.md](docs/easy/QUICKSTART.md).

This repository is **valdo-easy** (`0.0.6+easy.1`): upstream VALDO `1ec32c6` plus an
accessibility layer in `valdo/easy/`. Use the `valdo` command; do not call upstream stage
functions or edit generated configs yourself.

## Workflow

```bash
valdo --version --json                                    # is valdo installed? (exit 0)
valdo init DATA --apo APO.pdb [--models PATTERN] [--focal-seqid N] [--bound-models DIR] --json
valdo check --json                                        # ALWAYS before run
valdo run --dry-run --json                                # what would run, and why
valdo run --json                                          # or: valdo jobscript -o run_valdo.sh --json; qsub run_valdo.sh
valdo status --json                                       # any time, also during a run
valdo report --json                                       # rebuild <workdir>/report/
valdo stages --json                                       # the stage table
```

All commands take the project file as `PROJECT` or `-c FILE` (default `./valdo.yaml`).
`--json` goes before or after the subcommand.

## JSON contract (`schema: "valdo-easy/1"`)

- stdout is JSON Lines; human text goes to stderr. Parse only stdout, and only the **last
  line** as the result. Earlier lines (from `valdo run`) are progress events with an `event`
  key: `stage_start`, `stage_done` (`counts {in, out, dropped}`, `seconds`), `stage_skipped`,
  `stage_failed`, `message`.
- Every result has `schema`, `command`, `ok` (bool) and `"next"` (list of suggested next
  shell commands; they name the project file when it is not `./valdo.yaml`). If `ok` is
  false there is `"error": {code, message, what, cause, fix, log}`: show `what`, `cause` and
  `fix` to the human and read `log` (a file path) before retrying.
- Exit codes: 0 ok; 1 a problem with data/settings/a stage (fixable); 2 internal valdo-easy
  bug (report it, do not work around it); 130 interrupted.

Command-specific keys:

| command | keys |
|---|---|
| `version` | `version`, `upstream_version`, `upstream_base` |
| `init` | `project_file`, `n_datasets`, `decisions[{setting, value, why}]`, `summary`, `warnings` |
| `check` | `project_file`, `errors`, `warnings` (counts), `findings[{level, code, title, what, why, fix, items}]`, `environment` (with `refine_program`: the refinement program, version and folder, or its `error`), `inputs` (with `twin`: `possible` twin laws of the reference lattice, `configured`, `ok`), `notes` |
| `run` | `project_file`, `workdir`, `dry_run`, `plan[{stage, number, action, reason}]`, `stages[{stage, status, seconds, counts}]`, `datasets{total, ok, dropped}`, `report` (path of report.html or null) |
| `status` | `project_file`, `workdir`, `has_state`, `stages[{stage, number, folder, status, applicable, stale, next_action, reason, seconds, counts, log, error}]`, `datasets{total, ok, dropped, dropped_by_stage}`, `lock`, `last_run`, `refine_program` (program, version and folder the refine stage used, or null), `warnings`; while a run is active also `wait{reason, poll, seconds}` (poll `valdo status` about every `seconds`; `next` is then `["valdo status"]`) |
| `report` | `workdir`, `report_dir`, `result{counts, funnel, top_hits[], validation, files, warnings, ...}` |
| `jobscript` | `scheduler`, `path`, `script`, `slots`, `hours` (wall-clock limit written into the script), `hours_why` (estimated from the number of datasets unless `--hours` was given), `notes` (comments written into the script, e.g. where the refinement program comes from), `warnings` |
| `stages` | `stages[{name, number, folder, upstream, runs_when, description}]` |

Plan actions: `run`, `skip` (up to date), `keep`, `missing`, `later`, `not_applicable`.

## Ask the human before

- **Starting compute.** `valdo run` takes about an hour of 16 CPU cores per ~1,700 datasets
  (longer with `phases.source: refine`). On a cluster login node (`check` finding
  `login_node`), never run it there: write a job script with `valdo jobscript` and ask before
  submitting it.
- **Judgement calls reported by `valdo check`**: excluded files (`excluded_inputs`), cell
  outliers, low-resolution datasets, datasets without models or phases, fewer than ~100
  datasets (`few_datasets`), the choice of reference, reindexing, `phases.source`, and the
  focal residue. Report them; do not silently change settings to make warnings disappear.
- **Changing anything under `advanced:`** in `valdo.yaml` (validated PTP1B settings), or
  using PHENIX (licence) instead of dimple.
- **Where PHENIX or CCP4 is installed.** With `phases.source: refine`, valdo needs
  `programs.phenix` or `programs.ccp4` in `valdo.yaml` (the installation folder; one line covers
  every command and batch job) unless the program is on PATH. If `valdo check` reports
  `refine_program_missing`, `program_not_found` or `program_setup_failed`, ask the human for the
  installation folder (or their lab's) and set it. Do not search the disk for installations
  (no `find /`); `valdo init` already looks on PATH, and a folder that holds several
  installations is listed in the error. Never install PHENIX or CCP4 yourself, and never accept
  a licence for the human; valdo sources the installation's setup script, so only use
  installations the human names.
- **Twin refinement** (`phases.twin_law`): whether the crystals are twinned is the human's
  call. Offer the laws `valdo check --json` lists in `inputs.twin.possible` (the `twin_law`
  finding), and set the law the human chooses (`--twin-law -h,-k,l` or `--twin-law=-h,-k,l`);
  `auto` only when the human agrees to twin refinement and the lattice has exactly one law
  (`twin_law_ambiguous` otherwise). Do not invent one, and never put a twin law into
  `advanced.refine.extra_args`.
- **Interpreting hits.** A blob means "something changed here", not "a ligand". Present the
  top hits with their flags; the human judges the maps.

## Safety rules

- Never delete or modify input data (MTZ, PDB). valdo only reads them.
- Never delete anything outside the work folder. valdo itself only removes files inside
  `<workdir>` in folders carrying its `.valdo-managed` marker. If it reports
  `foreign_folder` or `unsafe_path`, stop and ask; do not remove the folder yourself.
- Never delete `<workdir>/.lock`. While `valdo status --json` shows `lock` as active (not
  `stale`) a run is going: poll `valdo status` (the result's `wait`), do not start another run.
  A lock with no sign of life for 5 minutes is replaced by the next `valdo run` automatically.
  After `locked`, `next` is `["valdo status"]`.
- Do not edit `<workdir>/valdo_state.json`, `datasets.csv` or `configs/*.yaml` by hand; change
  `valdo.yaml` and rerun. `valdo run --from STAGE` reruns from a stage.
- Validate edits to `valdo.yaml` with `valdo check --json` (unknown keys are errors with a
  did-you-mean suggestion).
- Do not `pip install` into an environment other people use.

## Reading results

- `valdo status --json`: a stage with `stale: true` would rerun; `next_action` says what the
  next `valdo run` does; `datasets.dropped_by_stage` says where datasets were lost.
- `<workdir>/datasets.csv`: one row per input file (`id`, `original_name`, `status`,
  `dropped_stage`, `dropped_reason`, `flags`, ...). Always report dataset names as
  `original_name (id)`.
- `<workdir>/report/report.json`: `counts`, `funnel`, `top_hits` (ranked by `score`; each has
  `rank`, `id`, `original_name`, `sample`, `blob` (`<sample> #<rank in its dataset>`), `score`,
  `peakz`, `nearest_residue`, `diff_max_sigma` and `diff_min_sigma` (the most positive and most
  negative difference density within the blob, sigma), `diff_sign` (the dominant side at
  3 sigma: `positive`, `negative`, `mixed` or `weak`; a ligand is usually positive),
  `diff_peak_sigma` (the larger in magnitude of the two; do not read its sign alone),
  `on_known_ligand` (true/false with a known bound model for that dataset, else null),
  `flags`), `validation` (only with
  known bound models; descriptive), `warnings`. Top hits leave out symmetry copies of a blob
  already listed (`top_hits_symmetry_copies_left_out`), so ranks can skip numbers.
  `report/blobs.csv` has every kept blob (copies marked in `symmetry_copy_of`).

## Troubleshooting

Every error `code` and check finding `code` is explained in
[docs/easy/TROUBLESHOOTING.md](docs/easy/TROUBLESHOOTING.md). Common ones: `check_failed`
(fix the listed findings), `project_invalid` (typo in valdo.yaml), `locked` (another run;
wait), `stage_failed` (read `error.log`; `error.fix` says what to change; then `valdo run`
again; finished stages are kept).

## Working on this repository

- Tests: `python -m pytest tests/easy` (needs torch, gemmi 0.6.7, reciprocalspaceship; about
  5-10 minutes on 4 CPU cores, depending on the node). The fast subset without torch:
  `python -m pytest tests/easy/test_core_project.py tests/easy/test_core_scan.py tests/easy/test_docs.py`.
- Upstream scientific code may change only to fix crashes or silent data loss; record every
  change in `docs/easy/CHANGES_FROM_UPSTREAM.md` with a test. Design: `docs/easy/DESIGN.md`.
- After changing the `valdo.yaml` schema or `valdo/easy/hints.py`, run
  `python -m valdo.easy.docgen` (tests fail when the generated docs are stale).
