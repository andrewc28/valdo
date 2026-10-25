---
name: valdo
description: Run VALDO (VAE-assisted ligand discovery, valdo-easy) on a crystallographic fragment screen of MTZ files to find candidate ligand-bound datasets, and read its results. Use when the user wants to set up, check, run (locally or as an SGE/SLURM job), monitor, resume or interpret a VALDO project (valdo.yaml, a valdo_run work folder, report.html, blobs), or asks why a VALDO stage failed.
---

# VALDO (valdo-easy)

The `valdo` command runs the whole VALDO pipeline from one project file (`valdo.yaml`).
Install: copy this folder to `~/.claude/skills/valdo/`. Full agent notes: `AGENTS.md` at the
root of the valdo repository; human docs: `docs/easy/`.

## Workflow

1. `valdo --version --json` (must succeed; otherwise ask the human which environment to
   activate).
2. `valdo init DATA --apo APO.pdb [--models PATTERN] [--focal-seqid N] [--bound-models DIR] --json`
   DATA = folder, quoted glob or .txt list of MTZ files. `--models '/path/{name}.pdb'` or a
   folder for per-dataset models. Read `decisions[]` (columns, `phases.source`, reference,
   reindex) and tell the human what was decided and why.
3. `valdo check --json`. If `ok` is false: report each error finding (`title`, `what`, `fix`)
   and stop. Report warnings and ask the human about judgement calls (below).
4. `valdo run --dry-run --json`, show the plan, then **ask before compute**:
   - workstation: `valdo run --json`;
   - cluster (finding `login_node`): `valdo jobscript -o run_valdo.sh --json`, show the script,
     ask, then `qsub run_valdo.sh` (SGE) or `sbatch` (SLURM).
5. `valdo status --json` to monitor (also safe during a run; while a run is active its result
   has `wait` and `next: ["valdo status"]`: poll, do not start another run). Resume after any
   interruption with the same `valdo run`: finished stages are kept.
6. When `report` (run result) or `report/report.html` exists: read
   `<workdir>/report/report.json` and summarise `counts`, the `funnel`, the top 10
   `top_hits` (`original_name (id)`, `blob`, `score`, `peakz`, `nearest_residue`,
   `diff_max_sigma` / `diff_min_sigma` with `diff_sign` (the dominant side: positive, negative,
   mixed, weak; do not judge the sign from `diff_peak_sigma` alone), `on_known_ligand`, `flags`)
   and `validation` if present (descriptive only). Ranks can skip numbers: symmetry copies of a
   listed blob are left out. `valdo report --top 50 --json` rebuilds the report with maps for
   more hits.

## JSON rules

- Parse stdout only; the **last line** is the result: `schema` (`valdo-easy/1`), `command`,
  `ok`, `"next"` (suggested shell commands), and on failure
  `error {code, message, what, cause, fix, log}`.
- Earlier lines of `valdo run --json` are events: `stage_start`, `stage_done`
  (`counts {in, out, dropped}`), `stage_skipped`, `stage_failed`, `message`.
- Exit 0 ok, 1 fixable problem, 2 valdo-easy bug (report it with `error.log`), 130 interrupted.
- `valdo status --json`: `stages[].status` / `stale` / `next_action`, `datasets.dropped_by_stage`,
  `lock`.

## Ask the human

- before `valdo run` or submitting a job (about 1 h of 16 cores per ~1,700 datasets; longer
  with refinement);
- about `valdo check` warnings: `excluded_inputs`, `cell_outliers`, `low_resolution`,
  `missing_models`, `fallback_models`, `missing_input_phases`, `few_datasets`, and about the reference,
  reindexing, `phases.source` and the focal residue;
- before changing anything under `advanced:` in `valdo.yaml`, or using PHENIX (licence);
- for the PHENIX/CCP4 installation folder when `valdo check` reports `refine_program_missing`,
  `program_not_found` or `program_setup_failed`: set `programs.phenix` / `programs.ccp4` in
  `valdo.yaml` to the folder they name (one line covers every command and batch job). Do not
  search the disk for installations. Never install PHENIX or CCP4 or accept a licence for them;
- before twin refinement (`phases.twin_law`): offer the laws in `valdo check --json`
  `inputs.twin.possible`; the human decides whether the crystals are twinned. Set the law they
  choose (`--twin-law -h,-k,l`; `auto` = the lattice's only law), never in
  `advanced.refine.extra_args`;
- for the scientific verdict: a blob means "something changed here", not "a ligand"; the
  human judges the maps (`report/view_top_hits.py` for Coot, `.pml` for PyMOL).

## Safety

- Never modify or delete input MTZ/PDB files, or anything outside the work folder.
- On `foreign_folder`, `unsafe_path`, or an active `lock`: stop and ask; do not delete.
- Do not hand-edit `valdo_state.json`, `datasets.csv` or `configs/`; edit `valdo.yaml`, run
  `valdo check --json`, then `valdo run` (or `valdo run --from STAGE`).

## Troubleshooting

| code | do |
|---|---|
| `check_failed` | fix the error findings it lists (each has `fix`), rerun `valdo check` |
| `project_invalid` | `error.details` lists bad keys with did-you-mean suggestions; fix `valdo.yaml` |
| `locked` | another run uses the work folder; poll `valdo status` (about every minute) until `lock` is gone or `stale`, then `valdo run`; never delete `.lock` |
| `stage_failed` | read `error.log` (tail), relay `what`/`cause`/`fix`; after the fix, `valdo run` again |
| `prerequisite_missing` | run plain `valdo run` first |
| `internal_error` | stop; give the human the traceback and log path to report |

All codes: `docs/easy/TROUBLESHOOTING.md` in the valdo repository.
