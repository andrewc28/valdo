# valdo-easy: design

Audience: reviewers and maintainers. User documentation lives elsewhere (QUICKSTART, README).
This file describes what the accessibility layer (`valdo/easy/`) does, the contracts between its
modules, and every place where it touches upstream VALDO.

valdo-easy is an **accessibility layer, not new science**. It is based on upstream VALDO
`1ec32c6` (version 0.0.6, merge of PR #45 `pipeline-cli`). Its version is `0.0.6+easy.1`
(`valdo/VERSION`; PEP 440 local version = upstream 0.0.6 + easy release 1).

- One command, `valdo`, with subcommands `init`, `check`, `run`, `status`, `report`,
  `jobscript`, `stages` and the flag `--version`.
- One project file, `valdo.yaml`, instead of ten stage configs.
- One work directory per project with a fixed layout, a dataset table (`datasets.csv`),
  a resumable state file (`valdo_state.json`) and one log per stage.
- The scientific work is done by the **unchanged upstream stage functions**
  (`valdo.commandline.pipeline.STAGE_REGISTRY`), called in-process with generated config
  dicts. Every generated config is written to `<workdir>/configs/<stage>.yaml`, so any stage
  can be rerun with plain `valdo.pipeline <stage> <workdir>/configs/<stage>.yaml`.
- With default settings the numbers are the upstream numbers. Upstream code is changed only
  for the fixes listed in section 10 (each has a test in `tests/easy/test_core_upstream.py`).
  `tests/easy/test_e2e.py` reruns every stage of a finished toy run with plain
  `valdo.pipeline STAGE configs/STAGE.yaml --force` and checks that the outputs are the same.

`valdo.pipeline`, `valdo.refine` and `valdo.occupancy` keep working unchanged.

## 1. Modules

| module | owner | purpose |
|---|---|---|
| `valdo/easy/__init__.py` | A | `SCHEMA_ID = "valdo-easy/1"`, `UPSTREAM_BASE = "1ec32c6"`, `__version__` |
| `cli.py` | A | argparse front end; human vs `--json` output; exit codes |
| `project.py` | A | `valdo.yaml` schema, loading, strict validation, commented YAML writer |
| `scan.py` | A | fast MTZ header reader, file listing, dataset naming, column/phase detection, name matching |
| `ids.py` | A | canonical IDs, `datasets.csv` read/write, stem parsing |
| `check.py` | A | environment / input / settings checks (`valdo check`) |
| `hints.py` | A | table of known error patterns -> what / cause / fix |
| `stages.py` | A | the stage table: config builders, pre-hooks, post-conditions |
| `runner.py` | A | state, fingerprints, locking, log capture, cleaning, progress |
| `refine.py` | B | refinement with phenix / dimple (`phases.source: refine`) |
| `programs.py` | B | `programs.phenix` / `programs.ccp4`: finds the installation, sources its setup script in a clean `bash`, returns the program and its environment (cached per process) |
| `twin.py` | B | twin laws (`phases.twin_law`): hkl parsing, possible laws of a lattice, checks, REFMAC/PHENIX twin output parsers |
| `report.py`, `maps.py` | B | report, per-hit maps, viewer scripts |
| `jobscript.py` | B | SGE / SLURM job scripts |
| `docgen.py` | C | generates `docs/easy/CONFIG_REFERENCE.md` (from the schema) and the hint table of `TROUBLESHOOTING.md` (from `hints.py`); `tests/easy/test_docs.py` fails when they are stale |

`init`, `check`, `status`, `stages`, `jobscript` and `--help` never import torch (nor
`valdo.helper`, `valdo.scaling`, `valdo.vae_networks`). `valdo/__init__.py` loads its
torch-dependent members lazily (section 10). B's `detect_programs()` must not import torch either.
Only `valdo run` imports the upstream stage modules, inside the stage that needs them.

## 2. Command line

```
valdo --version [--json]
valdo init [DATA] (--apo PDB | --models PATTERN) [--apo PDB] [-o valdo.yaml] [--workdir DIR]
           [--phases auto|input_mtz|refine|existing] [--existing PATTERN] [--program phenix|dimple]
           [--phenix PATH] [--ccp4 PATH] [--twin-law LAW|auto]
           [--columns F,SIGF] [--reindex auto|true|false] [--reference NAME]
           [--focal-seqid N] [--focal-radius A] [--bound-models PATTERN] [--ligand-resnames LIG[,X]]
           [--ncpu N] [--force] [--json]
valdo check [PROJECT | -c FILE] [--deep] [--json]
valdo run [PROJECT | -c FILE] [--from STAGE] [--until STAGE] [--only STAGE] [--force] [--dry-run] [--verbose] [--json]
valdo status [PROJECT | -c FILE] [--json]
valdo report [PROJECT | -c FILE] [--top N] [--no-maps] [--json]
valdo jobscript [PROJECT | -c FILE] [--scheduler sge|slurm] [--slots N] [--hours H] [--mem-gb G] [--gpu]
                [--env-setup CMD] [--log-dir DIR] [-o FILE] [--json]
valdo stages [--json]
```

- `PROJECT` defaults to `./valdo.yaml`; `-c/--config FILE` is the same as the positional argument
  (job scripts use `valdo run -c "$CONFIG"`). Giving both is an error (`bad_options`).
- `VALDO_NCPU=N` in the environment overrides `compute.ncpu` for `run` and `report` (job scripts
  set it to the slots they were given); a note is printed when it differs.
- `run` and `report` limit numerical libraries to `ncpu` threads: `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to `ncpu` unless already set, and
  `torch.set_num_threads(ncpu)` is called before train and reconstruct (torch otherwise starts one
  thread per core of the node, far more than a batch job's slots).
- `init`: `DATA` is a folder of MTZ files, a quoted glob, or a `.txt` list. At least one of
  `--apo`/`--models` is required. If something required is missing and stdin is a TTY, `init`
  asks for it; otherwise it exits 1 naming the exact flag to pass. Refuses to overwrite an
  existing project file without `--force`.
- `check`: header-only by default; `--deep` reads reflection data (NaN, F <= 0, duplicates).
- `run`: runs `check` first (errors stop the run), then the stages in order. `--from S`
  forces S and everything after it; `--until S` stops after S; `--only S` runs S alone, forced
  (its prerequisites must be done); `--force` reruns the selected stages even if up to date;
  `--dry-run` prints the plan (section 7) and writes nothing; `--verbose` streams stage output
  to the console.
- `report`: if every stage before the report is done, reruns the `report` stage (with `--top` /
  `--no-maps` applied to this build only; the next `valdo run` sees the changed settings and
  rebuilds the report from `valdo.yaml`); otherwise builds a partial report from what exists,
  without touching the state.
- `status`: the recorded state plus the dry-run plan; `stale` marks a done stage that a run
  would redo (settings, inputs or earlier results changed).
- `jobscript`: prints (or writes with `-o`) a batch script that runs `valdo run PROJECT`
  (rendered by `valdo.easy.jobscript.render`).
- `stages`: the stage table below.

Exit codes: 0 success; 1 a user, data or stage problem (message says what/why/fix);
2 unexpected internal error in valdo-easy ("please report", with the log path); 130 interrupted.

## 3. Project file `valdo.yaml`

Written by `valdo init` with a comment on every field. Three parts: **1. your project**
(data, models, workdir), **2. detected by init** (check, change only if wrong),
**3. advanced** (validated PTP1B settings = upstream `_TEMPLATES` / `PTP1B_pipeline/configs`,
all written out explicitly so the file is the single source of truth).

Loading rules (`project.load_project`): unknown keys are errors with a did-you-mean
suggestion; numeric strings such as `'1e-3'` are converted for numeric fields (with a note);
`~` and `$VARS` are expanded in path fields; relative paths are resolved against the directory
of `valdo.yaml`; enums and ranges are checked; missing optional keys take the defaults below.

| key | default | meaning | upstream field |
|---|---|---|---|
| `data` | required | MTZ files: folder, glob or `.txt` list | standardize `source_path` (via `inputs/mtz/`) |
| `models.apo` | null | one apo PDB for every dataset; fallback for datasets without their own model; refinement start model | blobs/tag `model_folder` (via `inputs/models/`) |
| `models.per_dataset` | null | per-dataset models: a `{name}` pattern, folder or glob (matched by dataset name) | same |
| `workdir` | `valdo_run` | all outputs | every output path |
| `programs.phenix` / `programs.ccp4` | null | installation folder (or setup script, bin folder, program) of PHENIX / CCP4; null = from PATH | not in any config (see section 4, refine) |
| `columns.amplitude` / `.sigma` | detected | observed amplitude and sigma columns | standardize `expcolumns`; renamed to `F-obs`/`SIGF-obs` in `inputs/mtz/` if different |
| `phases.source` | detected | `input_mtz` (map coefficients in the data files), `refine` (refine every dataset), `existing` (user's refined MTZs) | add_phases_and_blobs `phasing_path` |
| `phases.existing` | null | refined MTZs for `existing` (`{name}` pattern, folder or glob) | |
| `phases.program` | null | `phenix` or `dimple` for `refine` | |
| `phases.twin_law` | null | twin law (hkl notation, reference indexing) for twin refinement, or `auto` (the only law of the reference lattice); `refine` only | `configs/refine.yaml` `twin_law` (only when set; `auto` resolved to the law, `check.resolve_twin_law`) |
| `phases.columns.fwt/phwt/delfwt/phdelwt` | detected | 2mFo-DFc amplitude/phase, mFo-DFc amplitude/phase in the phase files | `phase_2FOFC_col_in` = phwt, `phase_FOFC_col_in` = phdelwt |
| `phases.rfree` | detected | R-free flag column in the phase files | `rfree_label_in` |
| `reindex` | `auto` | `auto` (only if the reference lattice admits alternative indexing, `gemmi.find_twin_laws(cell, sg, 3, False)` as upstream), `true`, `false` | reindex stage on/off |
| `reference` | `auto` | dataset name, or first dataset in ID order that passes the input checks (upstream convention: the first dataset) | reindex `reference_file`, scale `reference_mtz` |
| `focal_residue.seqid` | null | residue number whose surroundings are excluded from the hit list (PTP1B: 215); null = none (`-9999`) | tag_blobs `focal_seqid` |
| `focal_residue.radius` | 5.0 | radius (A) | `focal_radius` |
| `focal_residue.tag` | `focal` | column name for the tag (init: e.g. `cys215`) | `focal_tag_name` |
| `validation.bound_models` | null | known ligand-bound models (descriptive validation only) | tag_blobs `bound_models_folder` (via `inputs/bound_models/`) |
| `validation.ligand_resnames` | `[LIG]` | ligand residue names in the bound models | report |
| `compute.ncpu` | init: CPUs available, max 16 | worker processes | every `ncpu` |
| `advanced.reindex.wcorr` | true | weighted per-shell CC | `wcorr` (schema fallback would be false) |
| `advanced.reindex.cc_min_dif` | 0.2 | keep both variants if CCs differ by less | `cc_min_dif` |
| `advanced.scale.when_opt` | `all` | `all`, `never` or a number in [0, 1] | `when_opt` (schema fallback 0.2) |
| `advanced.filter.max_rfree` | 0.45 | R-free cut (refine only) | `max_rfree` |
| `advanced.filter.min_cc` | 0.55 | post-scaling CC cut | `min_cc` |
| `advanced.train.*` | latent_dim 7, n_hidden_layers [3, 6], n_hidden_size 100, epochs 500, train_fraction 0.8, batch_size 100, learning_rate 0.001, w_kl 1.0, eps 0.02, stdof 128, include_errors true, random_seed 11231, activation relu | VAE (all 13 written explicitly; schema fallbacks differ for stdof, random_seed, activation) | train keys |
| `advanced.reconstruct.ml_recon` / `.repeats` | true / 1 | | reconstruct keys |
| `advanced.blobs.sigdF_pct`, `absdF_pct`, `extrapolate_factors`, `cutoff`, `radius_in_A` | 95.0, 99.99, [2,4,6,8,16], 3.5, 4.0 | | add_phases_and_blobs keys |
| `advanced.refine.eff` / `.extra_args` / `.timeout_min` | null / [] / 60 | refinement (B) | |
| `advanced.report.top_n` / `.maps` | 20 / true | report (B) | |

Validation beyond types (each error says what to change): `train_fraction` in (0, 1) and at least
one validation sample; `include_errors` must be true (upstream preprocess crashes otherwise);
`repeats > 1` requires `ml_recon: false` (upstream crashes otherwise); `activation` in
relu/tanh/sigmoid (upstream silently falls back to relu); `stdof` integer or null;
`n_hidden_layers` two integers; `when_opt` `all`, `never` or a float in [0, 1];
`cc_min_dif > 0`; `max_rfree` in (0, 1]; `min_cc` in [0, 1); `phases.existing` required for
`existing`, `phases.program` for `refine`.

Fixed plumbing (not user settings): column names after standardize `F-obs`/`SIGF-obs`,
`F-obs-scaled`/`SIGF-obs-scaled`, `recons`, `diff`, `WDF`, phase columns out
`PH2FOFCWT`/`PHFOFCWT`, blob phase column `PH2FOFCWT`, prefixes `""`.

## 4. Dataset identity: IDs and `datasets.csv`

- Dataset **name** = MTZ file stem, without an ending that every stem shares when it starts at a
  separator (`_`, `-`, `.`) and the names stay unique (PTP1B: `PTP1B-y0001_mrflagsref_idxs.mtz` ->
  `PTP1B-y0001`, the name its models carry; `x10`, `x20` stay as they are); if stems are not
  unique, the shortest unique path suffix (`parent/stem`), and if every stem is the same (e.g.
  `*/dimple.mtz`), the parent folder name. `reference:` accepts the name, the ID or the file stem.
- Datasets are sorted by name with a **natural sort** (digits compared as numbers, so `x2`
  before `x10`; ties broken by the plain string) and get canonical IDs `0001, 0002, ...`
  (width 4, or wider if there are more than 9999). Upstream stages only ever see IDs.
- Adding or removing data files renumbers the IDs and reruns everything.
- **Stems** downstream: `{ID}_{symop}` when reindex runs (upstream writes every dataset as
  `{ID}_{i}.mtz`), `{ID}` when it does not.
- Models, phase files and bound models with arbitrary names are matched to datasets by name:
  a pattern containing `{name}` is formatted per dataset; a folder or glob is matched by finding
  the dataset name in the file name (basename first, then the parent folder name) with a
  boundary rule (the characters next to the match must not be digits or letters continuing the
  name), longest name wins; two files for one dataset is an error.

`<workdir>/datasets.csv` (one row per input file, all values strings, IDs zero-padded):

| column | meaning |
|---|---|
| `id` | canonical ID |
| `original_name` | dataset name (see above) |
| `source_mtz` | absolute path of the input MTZ |
| `model` | model used for this dataset (absolute path; empty if none) |
| `phases_source` | `input_mtz`, `refine` or `existing` |
| `variant` | stem kept after filter (`0042_1`), empty before filter or if dropped earlier |
| `reindex_op` | operator of the kept variant in hkl notation (`h,k,l`, `-h,-k,l`); empty if reindex did not run |
| `status` | `ok` (still in the pipeline / finished) or `dropped` |
| `dropped_stage` | stage that removed it (`input` = excluded before standardize) |
| `dropped_reason` | plain-language reason |
| `flags` | `;`-separated: `reference`, `reindexed`, `ambiguous_indexing`, `model_frame_mismatch`, `model_map_mismatch`, `fallback_model`, `cell_outlier`, `low_resolution`, `low_phase_coverage`, `scaling_diverged` |
| `phase_file` | phase MTZ used by add_phases_and_blobs |
| `bound_model` | known bound model (validation), if any |
| `space_group`, `cell`, `d_min`, `n_reflections` | from the MTZ header |
| `twin_law`, `twin_fraction`, `twin_note` | twin refinement only (`phases.twin_law`): the law refined for the kept variant, its refined twin fraction (empty if not found) and a note (another law kept by REFMAC, no twin domain, fraction not found); empty otherwise |

`datasets.csv` is a view rebuilt after every stage from the base table (input scan) plus the
recorded contributions (drops, flags, values) of each finished stage in `valdo_state.json`, so a
rerun of a stage replaces exactly its own contributions.

## 5. Work directory

```
<workdir>/valdo_state.json  datasets.csv  .lock
<workdir>/configs/<stage>.yaml            generated configs, block-style YAML (+ refine.yaml, report.yaml)
<workdir>/configs/<stage>_files.txt       explicit input lists; configs/filter_input.csv, configs/refine_jobs.json
<workdir>/logs/<stage>.log  logs/<stage>.prev.log  logs/run.log
<workdir>/inputs/mtz/{ID}.mtz             symlink to the input MTZ (renamed copy if columns differ)
<workdir>/inputs/models/{stem}.pdb        model for blob detection: symlink, or a transformed copy (below)
<workdir>/inputs/phases/{stem}.mtz        phase files in the frame/ASU of each variant (input_mtz, existing)
<workdir>/inputs/refine_models/{stem}.pdb refine only: per-dataset models transformed into a variant's frame
<workdir>/inputs/bound_models/{ID}.pdb    known bound models: symlink, or a transformed copy (below)
<workdir>/01_standardize/ 02_reindex/ 03_scale/ 04_refine/{runs,phases,models,refine_summary.csv}
<workdir>/05_filter/ 06_preprocess/ 07_train/ 08_reconstruct/ 09_rescale/
<workdir>/10_phases_blobs/{mtz,blobs}/ 11_tag/ report/
```

Stage folders contain only what the stage writes (plus a hidden `.valdo-managed` marker), so
running the upstream function on `configs/<stage>.yaml` reproduces the folder exactly
(`test_every_stage_matches_plain_upstream`). The `inputs/` subfolders belong to the stage that
fills them (standardize: `mtz`; refine: `refine_models`; add_phases_and_blobs: `models`, `phases`;
tag_blobs: `bound_models`) and are rebuilt when that stage reruns.

**Models of reindexed datasets.** When a dataset's kept variant was reindexed with operator R
(`h' = hR`), a model refined against the original data is in the other setting. Per-dataset models
and bound models are therefore transformed with `valdo.easy.refine.reindex_model` (fractional
`x' = R^-1 x`, the real-space counterpart of the reciprocal-space operation; the toy screen checks
that a transformed model superposes on the apo model). The shared apo model (`models.apo`, also
the fallback for datasets without their own model) is assumed to be in the reference setting and
is never transformed. If a transformation fails, the original model is used and the dataset is
flagged `model_frame_mismatch` (focal tagging and viewer coordinates are then unreliable for it).
With phases from the data files or the user's refined files, add_phases_and_blobs also measures
how well each model sits in the 2mFo-DFc map of its phases (`stages.model_map_fit`: mean density
in sigma at the protein atoms; outputs `model_fit` in the state) and flags `model_map_mismatch`
below 1 sigma. On 293 PTP1B datasets the 286 own models gave 1.6-3.8 sigma (median 3.3); 6 of the
7 datasets that used the apo fallback gave about 0 (their phases came from models at origin (0,0,1/2) and/or in
the other indexing), as did every reindexed dataset evaluated with its untransformed model.

## 6. Stages

`valdo stages` prints this table. Numbering is fixed; the console counts only applicable stages.

| # | stage | folder | upstream | runs when |
|---|---|---|---|---|
| 1 | standardize | 01_standardize | standardize | always |
| 2 | reindex | 02_reindex | reindex | reindex resolved true |
| 3 | scale | 03_scale | scale | always |
| 4 | refine | 04_refine | (B: `valdo.easy.refine.refine_datasets`) | `phases.source: refine` |
| 5 | filter | 05_filter | filter | always |
| 6 | preprocess | 06_preprocess | preprocess | always |
| 7 | train | 07_train | train | always |
| 8 | reconstruct | 08_reconstruct | reconstruct | always |
| 9 | rescale | 09_rescale | rescale | always |
| 10 | add_phases_and_blobs | 10_phases_blobs | add_phases_and_blobs | always |
| 11 | tag_blobs | 11_tag | tag_blobs | always |
| 12 | report | report | (B: `valdo.easy.report.build_report`) | always |

Per stage: generated config (all folder paths absolute with a trailing `/`, so they also work
with pristine upstream code), pre-hook, post-conditions, and what goes into `datasets.csv`.

1. **standardize**: `source_path inputs/mtz/`, `destination_path 01_standardize/`,
   `mtz_file_pattern '.*/(\d+)\.mtz$'`, `expcolumns [F-obs, SIGF-obs]`. Pre: build `inputs/mtz/`
   (symlinks; gemmi-renamed copies when the amplitude/sigma columns are not `F-obs`/`SIGF-obs`,
   because upstream reindex and filter hard-code those names). Post: `{ID}.mtz` exists and has
   reflections for every input; otherwise dropped ("could not be read").
2. **reindex**: `input_files configs/reindex_files.txt`, `reference_file 01_standardize/{ref}.mtz`,
   `columns [F-obs, SIGF-obs]`, `wcorr`, `cc_min_dif`. Post: every input has a record row and its
   variant files; flags `ambiguous_indexing` (num_duplicates > 1), `reindexed` (best_symop != 0).
   The operator list `[x,y,z] + find_twin_laws(ref cell, ref sg, 3, False)` is stored in the state
   (`outputs.ops_hkl`).
3. **scale**: `file_list configs/scale_files.txt` (all reindex variants, or all standardized files),
   `reference_mtz` = the reference's `{ref}_{best_symop}.mtz` (or `{ref}.mtz`), `prefix ""`,
   `when_opt`. Post: one scaled file + metrics row per input; flag `scaling_diverged`
   (all-NaN `F-obs-scaled`; filter drops it).
4. **refine** (B): jobs = every scaled variant `{"stem", "id", "symop", "mtz": 03_scale/{stem}.mtz,
   "pdb": per-dataset model (transformed into the variant's frame, `inputs/refine_models/`) or
   apo}` (also written to `configs/refine_jobs.json`); `refine_datasets(jobs, program, 04_refine/,
   ncpu, eff, extra_args, timeout_min, log)`. The folder is **not** emptied when the stage reruns:
   `refine_datasets` resumes per variant (`04_refine/runs/<stem>/result.json` with input
   fingerprints), so a job killed after hours keeps its finished refinements. Post: datasets whose
   variants all failed are dropped ("refinement failed"). The refined MTZ is related back to the
   input indexing by amplitude CC under the identity and each twin law (`find_frame_op`): the best
   operator must reach 0.9 and beat every other one by 0.1. (dimple/refmac write F after
   anisotropic scaling: CC 0.977-0.9998 in the right frame on 39 PTP1B datasets, 0.47 in the
   wrong one; an earlier fixed 0.98 cut-off rejected a valid refinement.) phenix.refine never
   re-indexes and writes the input's own F, so when the margin is not met its identity is kept
   if it is the best operator and reaches 0.9 (`refine._phenix_identity`; a real PTP1B dataset
   gave x,y,z 1.000, -x,-y,z 0.921 with PHENIX 1.21.2 and 2.2.1 and was dropped before), with the
   twin hint in the result's `reason`.
   Program location: `programs.for_project` resolves `programs.<family>` (or PATH) once per
   process; the stage passes the executable and the environment from the installation's setup
   script (`program_info`) to `refine_datasets` and records `programs.public(info)` as the stage
   output `program` (shown by `valdo status`, the report, and compared by `valdo check`:
   `program_changed`). The location is deliberately not part of the stage config or the
   per-dataset fingerprint, so moving or updating an installation does not refine everything
   again; `valdo check` compares only the version (`programs.version_differs`) with the versions
   counted in `refine_summary.csv`. phenix.refine command (PHENIX 1.21.2 and 2.2.1 checked):
   `<exe> <pdb> <mtz> output.prefix=refine_<stem> <eff> miller_array.labels.name=<F>,<SIGF>
   [xray_data.r_free_flags.generate=True | miller_array.labels.name=<free> when several flag
   columns] xray_data.r_free_flags.ignore_pdb_hexdigest=True [xray_data.twin_law=<law>]
   <extra_args> --overwrite` (each of valdo's settings left out when `extra_args` sets it;
   `PHENIX_COMMAND_VERSION` in the per-dataset fingerprint). `prepare_eff` writes the packaged
   `refine_drug.eff` to `04_refine/phenix_refine.eff` with three repairs (the missing R-free
   `file_name` line and `output.quiet` dropped, the unclosed `refine {` scope closed before
   `main {`); a user `.eff` is used as it is unless its braces do not balance (error).
   Twin refinement (`phases.twin_law`, added to the stage config only when set, so other
   projects keep their fingerprint; `auto` is resolved from the reference lattice by
   `check.resolve_twin_law` in `build_refine`, and `execute_refine` refuses an unresolved one):
   phenix `xray_data.twin_law=<law>`, dimple `--refmac-key twin` (jelly-body and final
   restrained refmacat runs only). The law is given in the reference indexing; `pre_refine`
   writes a job's own `twin_law` for a variant whose re-indexing operator does not commute with
   it (`twin.law_for_variant`: M^-1 L M with M = B^-1 V, B the dataset's best operator, V the
   variant's; only lattices with several non-commuting laws, e.g. pseudo-cubic ones). In twin mode the frame operator comes
   from the program instead of `find_frame_op` (REFMAC's output F are detwinned: CC with the
   input 0.89-0.999 under the right operator, and near-equal under both at a twin fraction
   of 0.5): phenix identity; dimple: identity unless the `[pointless]` section of `dimple.log`
   has a non-empty `alt_reindex` (dimple 2.7 then refines pointless's `reindexed.mtz`), in which
   case the candidate under which `reindexed.mtz` equals the input amplitudes exactly, else the
   inverse of the first `alt_reindex` operator; then a loose CC check (>= 0.5). Per variant: `twin_law`, `twin_fraction`, `twin_note`
   (REFMAC domains from final.mmcif `_pdbx_reflns_twin`, else REMARK 3 "TWIN DETAILS"; laws
   conjugated back into the input indexing and compared modulo the space group's rotations and
   -1; PHENIX: REMARK 3 "TWINNING INFORMATION" FRACTION/OPERATOR, else the mmCIF
   `_pdbx_reflns_twin`, else the last `r_work=... twin_fraction=... twin_law=...` line of its log,
   formats checked on 1.21.2 and 2.2.1 output); stage outputs `twin_law` and `twin` {stem: [law,
   fraction, note]}. Without twin refinement, a failed frame check where two operators reach
   0.9 (or a phenix.refine result kept in the input indexing in that situation) gets the hint
   "both indexings fit, as for twinned data" (`refine.TWIN_HINT`), and `post_refine` counts
   those datasets in a stage message.
5. **filter**: upstream `run_filter` on `configs/filter_input.csv` (columns `file_idx, symop,
   Rf_final, Rf_refined, selected, note`). With refinement: `Rf_final` = refined R-free for the
   lowest-R-free variant of each dataset, `99` for the other variants and failed refinements
   (identical to upstream for two variants; well defined for three or more). Without refinement
   (E8): `Rf_final` = 0 for `best_symop`, 99 for other variants, so exactly the best-CC variant
   survives; the all-NaN and `min_cc` checks still run. `metrics` is passed explicitly.
   With twin refinement (`phases.twin_law`) R-free cannot tell indexing variants apart (the twin
   fraction becomes 1 - f), so among several refined variants the one with the lowest refined
   twin fraction is kept when it is lower by >= 0.1 (`TWIN_CHOICE_MARGIN`), else `best_symop`,
   else the R-free rule (extras `twin_rule`, only in twin mode).
   Upstream's filter keeps every scaled file without a row in this table, so every other file in
   `03_scale/` (variants of datasets dropped after scale, e.g. by failed refinement) gets a row
   with `Rf_final` 99 and the note "dataset no longer in the pipeline (<stage>: <reason>)"; the
   list of those stems is part of the stage's extras (only when non-empty, so existing work folders
   keep their fingerprint).
   Post: `variant`, `reindex_op` filled; every dataset without a kept variant is dropped with
   the reason (R-free value, CC value, scaling diverged, not the selected variant); the stage
   fails if `kept_files.txt` lists anything else than the chosen variant of a dataset still in
   the pipeline (preprocess re-checks the VAE input the same way).
6. **preprocess**: `file_list 05_filter/kept_files.txt`, pickles and arrays in `06_preprocess/`.
   Post: shapes agree (`vae_input` rows = `intersection.columns` = `vae_output` rows; union columns
   = `F_` + intersection columns: this catches the silent row misalignment); no NaN in
   `vae_input` (would give NaN reconstructions); datasets missing from the intersection are dropped.
7. **train**: all 13 hyperparameters and all 4 paths explicit. Post: model + loss plot exist,
   final losses finite.
8. **reconstruct**: Post: `recons.npy` rows = number of preprocess rows.
9. **rescale**: `input_files 05_filter/kept_files.txt`, `info_folder 06_preprocess/`.
   Post: one file per kept stem; datasets with non-finite `diff`/`SIGF-obs-scaled` are dropped
   (upstream would empty the file and abort the next stage).
10. **add_phases_and_blobs**: pre: models (`inputs/models/{stem}.pdb`, or `04_refine/models/` for
    refine), phase files (`inputs/phases/{stem}.mtz`: phase/map columns + R-free from the
    standardized input or the user's refined MTZ, transformed by
    `rs.DataSet.apply_symop(op).hkl_to_asu()`, which shifts phases correctly; or
    `04_refine/phases/` for refine). `op` is `stages.phase_operator(variant op, space group)`: the
    variant's operator when it maps the space group onto itself, else the member of its
    point-group coset that does (`refine.normalising_variant`; same amplitudes, e.g. y,x,-z for
    gemmi's I 41 law -x,y,-z). The dataset's own model is transformed with the same operator
    (`phase_op` in the job result); the apo fallback model is not transformed. When no member
    works (an origin shift would be needed) the dataset is dropped with that reason. Datasets with
    no model, no phase columns or zero phase coverage are dropped before the stage (upstream would
    skip them silently or abort the whole stage); `<90 %` coverage is flagged. The stage's extras
    include the copied phase columns (and `fwt`/`delfwt`, used for the model-map check and the
    report) and `PHASE_PREP_VERSION`, so a change in how inputs are prepared reruns this stage
    and the ones after it.
    Config: phase columns from `phases.columns` (or `refine.phase_columns(program)`),
    `rfree_label_in` from `phases.rfree`. Post: one phased MTZ per input with reflections;
    `blob_stats.pkl` exists.
11. **tag_blobs**: pre: `inputs/bound_models/{ID}.pdb` (transformed like the models).
    Config: `bound_models_folder` (absolute, only if bound models exist; omitted otherwise),
    `focal_seqid` (or -9999), `focal_tag_name`, `mtz_folder 10_phases_blobs/mtz/`.
    Post: both pickles exist.
12. **report** (B): `build_report(workdir, top_n, rank_by=None, known_bound_dir=inputs/bound_models
    or None, ligand_resnames, write_maps)`. Ranks are upstream's kept list ranked by `score`;
    `symmetry_copy_of` marks a blob whose centroid is a symmetry/lattice image of a stronger kept
    blob of the same sample (closer than max(1.5 A, half the smaller radius)), and the top hits
    are the first `top_n` non-copies. Per hit: the extremes of the unblurred difference map within
    the blob radius and `diff_sign` (a side counts when it reaches 3 sigma; when both do, the one
    at least 1.5 times stronger dominates, else `mixed`), and `on_known_ligand` from the
    validation's overlap rule (computed before the hits).

## 7. State, fingerprints, resume, cleaning, locking

`valdo_state.json`:

```json
{"schema": "valdo-easy/1", "valdo_version": "0.0.6+easy.1", "upstream_base": "1ec32c6",
 "project_file": "/abs/valdo.yaml", "workdir": "/abs/run", "created": "...", "updated": "...",
 "inputs": {"n_files": 300, "reference": {"name": "...", "id": "0001"}, "reindex": true,
            "ops_hkl": ["h,k,l", "-h,-k,l"], "column_rename": {}, "fingerprint": "..."},
 "stages": {"scale": {"status": "done", "fingerprint": "...", "token": "...",
            "started": "...", "finished": "...", "seconds": 123.4,
            "counts": {"in": 300, "out": 298, "dropped": 2},
            "outputs": {...}, "log": "logs/scale.log", "error": null,
            "contrib": {"dropped": {"0007": "reason"}, "flags": {"0012": ["reindexed"]},
                        "values": {"0012": {"variant": "0012_1"}}}}},
 "last_run": {"started": "...", "finished": "...", "ok": true, "argv": [...]}}
```

- Stage `status`: `pending`, `running`, `done`, `failed`, `interrupted`, `skipped`
  (not applicable). A `running` stage found by a new run (lock free) counts as interrupted.
- **Fingerprint** of a stage = SHA-256 of its generated config (minus `ncpu`, which does not
  change results), stage-specific extras (input file list with size and mtime, model / phase /
  bound-model files, easy options) and the **token** of the previous applicable stage. A token
  is a fresh random ID written each time a stage completes, so any rerun upstream invalidates
  everything downstream. Configs and extras are built from the datasets as they enter the stage
  (`Context.before`: drops recorded by later stages are ignored), so a dataset dropped by filter
  does not make standardize look changed.
- **Plan**: for each selected stage, in order: forced (`--force`, `--from`, `--only`) -> run
  "forced"; not `done` -> run ("not run yet", "failed last time", "was interrupted"); an earlier
  stage of this run reruns -> run "an earlier stage reruns"; fingerprint changed -> run "settings,
  inputs or earlier results changed"; otherwise skip "up to date". Stages outside the selection
  are `keep` (done, before `--from`), `missing` (not done, before `--from`: the run refuses to
  start), `later` (after `--until`) or `not_applicable`. When a stage runs, every later stage is
  reset to `pending`. Upstream sentinels are irrelevant: before a stage runs its folder is
  emptied (this fixes upstream's stale-file traps; refine excepted, see section 6) and `_force`
  is passed.
- **Cleaning** only touches folders inside the workdir that carry the `.valdo-managed` marker
  written when the runner created them; entries are removed without following symlinks
  (a symlink is unlinked, never its target); a stage folder that is itself a symlink, or lies
  below one inside the workdir, is refused (`unsafe_path`; the workdir itself may be a link);
  the workdir itself and anything outside it are never removed.
- **Lock**: `<workdir>/.lock` created with `O_EXCL` (pid, host, time, argv); a heartbeat thread
  touches it every 30 s. A lock is stale if its process no longer runs (same host) or it has not
  been touched for 5 minutes (any host, e.g. a batch job killed at its time limit). A lock file
  that cannot be parsed counts as stale only when it is older than 10 s (another run may be
  between creating and writing it). A stale lock is removed safely against concurrent runs: it is
  renamed to a unique `.lock.stale-<host>-<pid>-<random>` name (atomic), and if what was moved is
  not the lock judged stale (another run replaced it in between) it is put back with `link()`,
  which never overwrites; the remover then retries (3 times at most). Otherwise the run stops
  with code `locked`, the time the lock will count as abandoned, and `next: ["valdo status"]`.
  `status` and `check` do not take the lock; `status` shows it, and while it is active its result
  carries `wait` and `next: ["valdo status"]`. SIGTERM and Ctrl-C mark the running stage
  `interrupted` (exit 130); after SIGKILL the stage stays `running` and the next run redoes it.
- **Logs**: each stage's stdout/stderr (including multiprocessing workers and C-level output) goes
  to `logs/<stage>.log` (previous one kept as `<stage>.prev.log`); the console shows one line per
  stage, e.g. `[3/11] scale ... done in 12 min: 298/300 datasets; 2 dropped, see datasets.csv`.
  `--verbose` also streams the log. `logs/run.log` collects the console lines of every run.
- **Failure**: the log tail is matched against `hints.py`; the console prints what went wrong,
  likely cause, how to fix, and the log path; exit 1. A bug in valdo-easy itself gives exit 2.

## 8. JSON output (`--json`)

stdout becomes JSON Lines; human text goes to stderr. Progress events (run only) have an `event`
key: `stage_start {stage, index, total, reason, log}`, `stage_done {stage, index, total, seconds,
counts{in, out, dropped}}`, `stage_skipped {stage, index, total, reason}`, `stage_failed {stage,
error}`, `message {level, stage, text}`.
The **last line** is always the result object:

| key | all commands |
|---|---|
| `schema` | `"valdo-easy/1"` |
| `command` | `init`, `check`, `run`, `status`, `report`, `jobscript`, `stages`, `version` |
| `ok` | bool |
| `next` | list of suggested next shell commands; `valdo check/run/status/report/jobscript` suggestions carry `-c <project>` when the project file is not `./valdo.yaml`, so they work when copied as they are |
| `error` | only if `ok` is false: always the keys `{code, message, what, cause, fix, log}` (null when unknown), plus `details`/`stage`/`traceback` where they apply; `check` with errors gives code `check_failed` |

Command-specific keys: `version {version, upstream_base}`; `init {project_file, n_datasets,
decisions, summary, warnings}`; `check {project_file, errors, warnings, findings[{level, code,
title, what, why, fix, items}], environment, inputs}`; `run {project_file, workdir, dry_run,
plan[{stage, number, action, reason}], stages[{stage, status, seconds, counts}], datasets{total,
ok, dropped}, report}` (`report` = path of report.html or null); `status {project_file, workdir,
has_state, stages[{stage, number, folder, status, applicable, stale, next_action, reason, started,
finished, seconds, counts, log, error, messages}], datasets{total, ok, dropped, dropped_by_stage},
lock, last_run, inputs}`; `report {workdir, report_dir, result}` (`result` = the dict returned by
`build_report`); `jobscript {scheduler, path, script}`; `stages {stages[{name, number, folder,
upstream, runs_when, description}]}`.

## 9. Error hints (`hints.py`)

A list of `{id, pattern (regex), stages, what, cause, fix}`; `explain(stage, text)` returns the
first match for the stage (or a generic answer). Covered: out of disk, memory (MemoryError,
`Killed`, `bad_alloc`), CUDA out of memory, GPU pickle on a CPU node, `Singular matrix` in scale,
missing MTZ columns (`KeyError: 'F-obs'`), missing files/dangling symlinks (scratch purge),
permissions, unreadable MTZ, empty concatenation (no datasets left), rescale shape mismatch,
NaN loss, missing torch/gemmi/pkg_resources, gemmi API mismatch, refinement program missing,
timeouts, broken process pool, stale NFS handles, too many open files.

## 10. Upstream changes

Each has a test in `tests/easy/test_core_upstream.py`. None changes results for configs that
already worked.

| file | change | why |
|---|---|---|
| `valdo/__init__.py` | version read from `valdo/VERSION` (no `pkg_resources`); `Scaler`, `Scaler_pool`, `VAE` and the submodules load lazily (PEP 562 `__getattr__`) | `import valdo` failed without `pkg_resources`/an install, reported another copy's version, and imported torch for every CLI call |
| `valdo/VERSION` | `0.0.6+easy.1` | E13 |
| `helper.py` `standardize_input_mtzs` | `glob(os.path.join(source_path, "*.mtz"))`; a globbed path is no longer joined to `source_path` again | without a trailing `/` nothing matched; a relative `source_path` was doubled and every file skipped |
| `helper.py` `add_phases`, `add_phases_from_pool_map` | `os.path.join` for the output path | without a trailing `/` the phased files landed in the parent folder and no blobs were found |
| `scaling.py` `Scaler.batch_scaling`, `Scaler_pool` | `os.path.join` for scaled MTZs and the metrics pickle | same trap (FileNotFoundError after scaling everything) |
| `tag.py` `apply_determine_locations` | `os.path.join(folder, sample + ".mtz")` | same trap |
| `commandline/pipeline.py` `run_filter` | `file_idx`/`symop` read as text; the R-free cut also matches `{file_idx}.mtz` | IDs wider than 4 digits never matched; with reindex skipped the R-free cut silently did nothing |
| `commandline/pipeline.py` | `if __name__ == "__main__": main()` at the end | `python -m valdo.commandline.pipeline ...` exited 0 without doing anything |
| `commandline/refine.py` | stray `)` in `output.prefix` (multi-PDB branch) | every output prefix ended in `)` |
| `setup.py`, `MANIFEST.in` | `valdo` console script, package data (`VERSION`, `refine_drug.eff`), `tests` excluded, `dev` extra | `pip install .` lost the data files; `valdo` needs an entry point |
