# Changelog

All notable changes to valdo-easy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are
`<upstream version>+easy.<n>` (PEP 440 local versions; see
[docs/easy/CHANGES_FROM_UPSTREAM.md](docs/easy/CHANGES_FROM_UPSTREAM.md)).

## [0.0.6+easy.1] - 2026-10-24

First release of valdo-easy, based on upstream VALDO 0.0.6, commit `1ec32c6`
(merge of PR #45 `pipeline-cli`).

### Added
- `valdo` command with subcommands `init`, `check`, `run`, `status`, `report`, `jobscript`,
  `stages` and `--version`; every subcommand takes `--json` (JSON Lines, schema `valdo-easy/1`).
- `valdo.yaml`: one commented project file with strict validation (unknown keys are errors with
  a did-you-mean suggestion) and the validated PTP1B settings written out explicitly.
- `valdo init`: reads MTZ headers (1,680 PTP1B files in a few seconds) and detects amplitude
  columns, map coefficients, space group, cell, resolution, reindexing and the reference dataset.
- `valdo check`: environment, input and settings checks in plain language (`--deep` reads the
  reflection data).
- `valdo run`: all stages in order, resumable, with per-stage logs, fingerprints, a lock file,
  post-conditions and explanations of known errors; canonical dataset IDs and
  `<workdir>/datasets.csv` recording what happened to every dataset.
- Refinement stage (`phases.source: refine`) with PHENIX or CCP4 dimple, in parallel and resumable.
- Model-map check: with phases from the data files or the user's refined files, datasets whose
  model does not sit in the 2mFo-DFc density of their phases are flagged `model_map_mismatch`
  (focal tagging and viewer models would be unreliable for them).
- Report: `report.html` (self-contained), `report.md`, `report.json`, ranked hit tables, maps of
  the top hits and Coot/PyMOL viewer scripts; optional descriptive validation against known
  bound models.
- `valdo jobscript`: SGE or SLURM batch scripts.
- Documentation in `docs/easy/` (QUICKSTART, USER_GUIDE, CONFIG_REFERENCE, TROUBLESHOOTING,
  WYNTON, DESIGN, CHANGES_FROM_UPSTREAM), `AGENTS.md` and a Claude Code skill.
- Test suite `tests/easy/` including a synthetic end-to-end screen and an upstream-equivalence test.
- `environment.yml` with the versions validated on UCSF Wynton.

### Fixed (upstream code; details and tests in CHANGES_FROM_UPSTREAM.md)
- Paths built by string concatenation required a trailing `/` (helper.py, scaling.py, tag.py).
- `run_filter`: IDs wider than 4 digits; R-free cut did nothing when reindex was skipped.
- `python -m valdo.commandline.pipeline` did nothing.
- `valdo.refine`: stray `)` in the PHENIX output prefix.
- `import valdo` needed `pkg_resources` and an installed package, and imported torch eagerly.
- Packaging: `refine_drug.eff` and `VERSION` were not installed by `pip install .`.

### Fixed (valdo-easy, found in review before the release)
- A dataset whose refinement failed was still used to train the VAE (upstream filter keeps scaled
  files without an R-free row); such files now get a row that drops them, and filter/preprocess
  stop if anything else gets through.
- Re-indexed datasets: phases were moved with the raw twin law but their model with the
  space-group-safe variant (different for e.g. I 41); both now use the safe operator, and
  datasets that would need an origin shift are dropped with a reason. Existing work folders
  rerun `add_phases_and_blobs`, `tag_blobs` and `report` once (also when the `fwt`/`delfwt`
  columns change).
- Report: the top-hit table showed only the largest-magnitude difference value, so ligands
  next to negative density looked negative; it now shows the most positive and most negative
  value, which sign dominates and (with known bound models) whether the blob is on the ligand. Lower
  symmetry copies of a listed blob are left out of the top hits (marked in `blobs.csv`). The
  PyMOL and Coot scripts also work when started from the work folder.
- `valdo check`: a `models.per_dataset` pattern that matches nothing is an error
  (`models_pattern_no_match`; `valdo init` warns); a missing `validation.bound_models` path has
  its own warning (`bound_models_missing`); wrong amplitude columns name the columns the files
  have; with no usable file, only the reasons are shown.
- Lock: a lock file being written by another run was taken for stale, and removing a stale lock
  could remove a fresh one; the `locked` message now gives the expiry time and `next` is
  `valdo status`. `valdo status` says to wait while a run is active; on a login node
  `valdo check` suggests a batch job.
- A stage folder that is a symbolic link is never emptied (`unsafe_path`).
- `valdo.yaml` syntax errors are explained with the line number; `project_invalid` gives a
  cause that fits the problem.
- `valdo jobscript`: the time limit is estimated from the number of datasets (was 24 h), the
  script sets `PYTHONNOUSERSITE=1`, and its notes match the situation.

### Changed
- `valdo/VERSION` is `0.0.6+easy.1`.
