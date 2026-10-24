# Troubleshooting

**For:** anyone whose `valdo` command printed an error or a warning, or whose results look
wrong.
**You need first:** the message (or `valdo check` / `valdo status` output) and, for a failed
stage, its log: `<workdir>/logs/<stage>.log`.

Every message from valdo has the same parts: **what** went wrong, **why** it matters (or the
likely cause), **how to fix** it, and the **log** to read. With `--json`, the same text is in
the result's `error` object (`code`, `what`, `cause`, `fix`, `log`) or in `findings[]`
(`code`, `what`, `why`, `fix`). The codes below are the `code` values.

General rules:

- Nothing is lost by rerunning. `valdo run` keeps finished stages and continues from the first
  unfinished one.
- `valdo status` shows which stage failed and where datasets were dropped.
- `<workdir>/datasets.csv` (columns `status`, `dropped_stage`, `dropped_reason`, `flags`) says
  what happened to each dataset.
- Exit codes: 0 success; 1 a problem with data, settings or a stage (fixable by you);
  2 an internal error in valdo-easy (please report it with the log); 130 interrupted.

## Messages from `valdo init`

| code | what it means | what to do |
|---|---|---|
| `project_exists` | `valdo.yaml` (or the `-o` file) already exists. | Add `--force` to overwrite it, or write another file with `-o other.yaml`. |
| `missing_data` | No data folder given (and no terminal to ask). | `valdo init /path/to/mtz_folder --apo apo.pdb`. |
| `missing_models` | Neither `--apo` nor `--models` given. | Add `--apo apo.pdb` (one model for all) and/or `--models '/path/{name}.pdb'`. |
| `model_missing` | The `--apo` file does not exist. | Check the path. |
| `no_data` | No MTZ files found at the given place. | Give a folder with `.mtz` files, a quoted glob (`'/data/*/final.mtz'`), or a `.txt` file listing paths. |
| `no_readable_data` | None of the files could be read as MTZ. | Check that these are merged MTZ files (not images or unmerged data). |
| `no_amplitudes` | No amplitude + sigma column pair (MTZ types F and Q). If only intensities exist, convert them first (French-Wilson: `ctruncate`, `phenix.french_wilson`). | Or name the columns: `--columns F,SIGF`. |
| `bad_columns` | `--columns` needs two names. | `--columns F,SIGF`. |
| `no_phases` | The data files have no map coefficients. | `--phases refine --program dimple` (or `phenix`), or `--phases existing --existing '/path/refined/{name}.mtz'`. |
| `missing_program` | `--phases refine` without `--program`. | Add `--program dimple` or `--program phenix`. |
| `missing_existing` | `--phases existing` without `--existing`. | Add `--existing PATTERN`. |
| `existing_unmatched` | No refined MTZ matched a dataset name. | Use a pattern with `{name}`, e.g. `'/refined/{name}/final.mtz'`. |
| `existing_no_phases` | The refined MTZs have no recognised map coefficients. | Set `phases.columns` (fwt, phwt, delfwt, phdelwt) in `valdo.yaml` by hand. |
| `project_invalid` | The settings are inconsistent (details listed). | Correct them as listed. |

`valdo init` also warns (it still writes the project) when `--models` matches none of the
datasets; `valdo check` then reports it as the error `models_pattern_no_match` (below).

## Messages from `valdo check` (and the start of `valdo run`)

Errors (`ERROR`) stop `valdo run`. Warnings (`WARN`) and information (`info`) do not. If
`valdo check` reports errors, its JSON result has `ok: false` and `error.code` `check_failed`.

**Environment**

| code | level | what to do |
|---|---|---|
| `missing_package` | error | A Python package is missing: activate the VALDO environment (see [QUICKSTART.md](QUICKSTART.md)) or install it. |
| `gemmi_too_new` | error | VALDO needs gemmi 0.6.7 or older: `pip install 'gemmi==0.6.7'`. |
| `login_node` | warning | You are on a shared login node: check here, then run as a batch job (`valdo jobscript -o run_valdo.sh`). |
| `ncpu_high` | warning | `compute.ncpu` is larger than the CPUs available: lower it, or request more slots. |
| `low_disk` | warning | Little free space for the work folder: a run needs roughly 12 MB per dataset (measured on PTP1B: 3.7 GB for 300 datasets, 20 GB for 1,680). Move `workdir` to scratch. |
| `refine_program_missing` | error | dimple or phenix.refine is not on PATH: load CCP4 (`source .../ccp4.setup-sh`) or PHENIX in this shell or job script. |
| `refine_unavailable` | error | The refinement module failed to load: a bug, please report. |

**Inputs**

| code | level | what to do |
|---|---|---|
| `no_data` | error | `data` names no MTZ files: fix `data` in `valdo.yaml`. |
| `excluded_inputs` | warning (error when more than half of the files are affected) | Some files cannot be used (unreadable, unmerged, missing columns, other space group, or a clashing `F-obs` column). Each group says how to fix it; the rest of the screen runs without them (`dropped_stage` `input`). When the configured amplitude/sigma columns are missing, the fix names the columns the files do have (e.g. "set columns.amplitude: F-obs and columns.sigma: SIGF-obs"). When no file is usable, only these findings are shown. |
| `no_amplitudes` | error | The configured amplitude/sigma columns are missing everywhere: correct `columns.amplitude`/`sigma` or rerun `valdo init --force`. |
| `other_amplitudes` | info | Other amplitude columns exist; change `columns.*` only if you want them. |
| `column_rename` | info | Your columns are copied under the names upstream expects (`F-obs`, `SIGF-obs`); nothing to do. |
| `cell_outliers` | warning | Unusual unit cells (more than 3% or 3 degrees from the median). Usually nothing to do; remove them if they are another crystal form. |
| `low_resolution` | warning | Datasets worse than 3.5 A: consider removing them (they shrink the set of reflections the VAE uses). |
| `resolution` | info | Resolution range of the screen. |
| `models_pattern_no_match` | error | `models.per_dataset` matched none of the datasets (a wrong folder or pattern). The finding shows what it looked for for the first dataset and which files are in that folder. Correct the pattern (`{name}` = dataset name), or remove it to use `models.apo` for every dataset on purpose. |
| `missing_models` | warning (error when no dataset has a model) | Datasets without a model are dropped at the blob stage: give `models.apo` as a fallback or fix the pattern. |
| `fallback_models` | info (warning above 10% of the datasets) | These datasets use the apo model (flag `fallback_model`). Many of them usually means `models.per_dataset` misses most model files: check it. |
| `model_ambiguous` | error | Two model files match one dataset: make the pattern specific (use `{name}`). |
| `no_input_phases` | error | `phases.source: input_mtz` but the data have no map coefficients: use `refine` or `existing`. |
| `missing_input_phases` | warning | Some data files lack map coefficients; those datasets are dropped at the blob stage. |
| `no_existing_phases` | error | `phases.existing` matched nothing: fix the pattern. |
| `missing_existing_phases` | warning | Some datasets have no refined MTZ; they are dropped at the blob stage. |
| `phase_ambiguous` | error | Two refined MTZs match one dataset: make the pattern specific. |
| `bound_models_missing` | warning | The `validation.bound_models` folder or file does not exist: correct the path or set it to `null`. Only the report's known-ligand validation is affected. |
| `no_bound_models` | warning | `validation.bound_models` exists but matched no dataset name (validation is skipped). |
| `bound_ambiguous` | warning | Some bound models matched more than one dataset; they are ignored. |
| `reference` | info | Which dataset is the reference and why. |
| `no_reference` | error | No usable dataset to be the reference. |
| `reference_unknown` | error | `reference` is not a dataset name: use a name from `datasets.csv` (`original_name`) or `auto`. |
| `reference_unusable` | error | The chosen reference was excluded: choose another or use `auto`. |
| `reindex` | info | Whether reindexing runs and why. |
| `reindex_impossible` | error | `reindex: true` but the lattice allows no alternative indexing: set `reindex: auto`. |
| `reindex_unknown` | warning | Alternative indexing could not be determined (unusual cell or space group); set `reindex` explicitly. |

**Settings**

| code | level | what to do |
|---|---|---|
| `too_few_datasets` | error | Fewer than 3 usable datasets: VALDO compares datasets with each other. |
| `few_datasets` | warning | Fewer than about 100 datasets: the VAE learns less well what an unbound crystal looks like, so rankings are less reliable. Fine for a test. |
| `bad_split` | error | `advanced.train.train_fraction` leaves no training or no validation dataset: use 0.8. |
| `batch_size` | info | The batch is larger than the training set; one batch per epoch (harmless). |

**`valdo check --deep`** (reads all reflection data)

| code | level | what to do |
|---|---|---|
| `deep_unreadable` | error | Reflection data could not be read: re-copy or remove the files. |
| `deep_nan` | info | Missing amplitudes; standardize removes those reflections (nothing to do). |
| `deep_f_nonpositive` | warning or error | Amplitudes <= 0 give NaN after scaling (error if the reference has them): use French-Wilson amplitudes; choose another reference. |
| `deep_sigma_nonpositive` | warning | Sigma <= 0 in some files; check how the data were processed. |
| `deep_duplicates` | error | Repeated reflections (unmerged or badly merged data): merge them properly. |
| `deep_ok` | info | Nothing found. |

## Messages from `valdo run`

| code | what it means | what to do |
|---|---|---|
| `check_failed` | `valdo check` found errors; nothing was run. | Fix them (run `valdo check` for details). |
| `project_invalid` | `valdo.yaml` cannot be used: a misspelled key (`advanced.train.epoch: unknown setting. Did you mean 'advanced.train.epochs'?`), a bad value, a missing file, or a YAML syntax mistake ("is not valid YAML near line 7": usually indentation, a misplaced `:` or a tab character; the parser's own text is in `details`). The same code comes from every command that reads the project. | Correct the listed settings or the named line. |
| `locked` | Another `valdo run` uses this work folder (its lock file, `<workdir>/.lock`, had a sign of life less than 5 minutes ago). | Wait, watching `valdo status` (its JSON result then has `wait` and `next: ["valdo status"]`). If that run was killed, its lock counts as abandoned 5 minutes after its last heartbeat (the message gives the time) and the next `valdo run` replaces it; do not delete `.lock` by hand while a run may be active. A job resubmitted within 5 minutes of a killed one can stop with `locked`: submit it again a few minutes later. |
| `bad_stage` | `--from/--until/--only` names an unknown stage. | `valdo stages` lists the names. |
| `stage_not_applicable` | That stage does not run in this project (e.g. `refine` with phases from the data). | `valdo status` shows the stages that apply. |
| `bad_options` | Options that cannot be combined (`--only` with `--from`, `--until` before `--from`, two project files). | Use one form. |
| `prerequisite_missing` | `--from`/`--only` asks for a stage whose earlier stages have not finished. | Run `valdo run` (or `--until` the missing stage) first. |
| `foreign_folder` | A stage folder exists, is not empty and was not made by valdo. valdo never deletes it. | Move it away or choose another `workdir`. |
| `not_a_folder` | A file sits where a stage folder should be. | Move it away. |
| `unsafe_path` | valdo refused to empty a stage folder because it is a symbolic link (exit code 1), or because it lies outside the work folder (a safety stop; exit code 2). valdo only empties real folders inside the work folder, so it can never delete files elsewhere. | For a link: replace it by a real folder (move the contents back), or put the whole work folder on the other disk with `workdir` in `valdo.yaml` (the work folder itself may be a link). Otherwise please report it. |
| `stage_failed` | A stage stopped. The message gives what/cause/fix from the table below and the log path. | Follow the fix, then `valdo run` again. |
| `internal_error` | A bug in valdo-easy (exit code 2). | Please report it with the traceback and the stage log. |
| `interrupted` | Ctrl-C or the batch system stopped the run (exit code 130). | `valdo run` again; it continues from the interrupted stage. |

### When a stage fails

valdo matches the end of the stage log against known error patterns and prints the matching
explanation. This table is generated from `valdo/easy/hints.py`:

<!-- BEGIN GENERATED: hints (python -m valdo.easy.docgen) -->

| id | stages | what went wrong | likely cause | how to fix |
|---|---|---|---|---|
| `disk_full` | any | The disk (or your quota) filled up while writing results. | The work directory is on a file system without enough free space for this many datasets. | Free space or move 'workdir' in valdo.yaml to a larger file system (e.g. cluster scratch), then rerun 'valdo run'; finished stages are kept. |
| `memory` | any | The process ran out of memory. | Large screens need several GB per worker; parallel stages multiply that by compute.ncpu. | Request more memory for the batch job (e.g. 'valdo jobscript --mem-gb 8') or lower compute.ncpu in valdo.yaml, then rerun 'valdo run'. |
| `cuda_oom` | train, reconstruct, scale | The GPU ran out of memory. | The VAE arrays (datasets x reflections) do not fit on this GPU. | Run on CPU (set CUDA_VISIBLE_DEVICES='' before 'valdo run') or use a GPU with more memory. |
| `gpu_pickle` | reconstruct | The trained model was saved on a GPU and cannot be loaded on this CPU-only machine. | train ran with a GPU visible, reconstruct runs without one. | Run reconstruct on a GPU node, or rerun training on CPU: CUDA_VISIBLE_DEVICES='' valdo run --from train. |
| `singular_matrix` | scale | Scaling failed for a dataset: too few reflections in common with the reference. | A dataset is in a different space group or setting, has very low resolution or completeness, or is not merged into the same asymmetric unit as the reference. | Run 'valdo check --deep' and look at the space-group, cell and resolution warnings; remove the offending file from 'data' (or choose another 'reference') and rerun. |
| `missing_column` | any | An expected column is missing from an MTZ file. | The column names in valdo.yaml (columns.*, phases.columns.*) do not match the files, or a file was replaced after 'valdo init'. | Run 'valdo check' to see which files lack which columns; correct columns/phases.columns in valdo.yaml (or 'valdo init --force' to re-detect), then rerun. |
| `file_missing` | any | A file that a stage needs does not exist. | Input files moved or were deleted (cluster scratch is purged after a few weeks), or an earlier stage did not write its outputs. | Run 'valdo check' (it lists missing inputs) and 'valdo status'; restore the inputs or rerun the earlier stages with 'valdo run --from <stage>'. |
| `permission` | any | No permission to read an input or to write in the work directory. | The files belong to someone else or the folder is read-only. | Check permissions ('ls -l'); put 'workdir' in a folder you own; ask the data owner for read access. |
| `bad_mtz` | any | An MTZ file could not be read. | The file is corrupt or incomplete (e.g. a copy was interrupted). | Run 'valdo check' to find unreadable files; re-copy or remove them from 'data' and rerun. |
| `nothing_left` | any | No datasets were left to process at this stage. | Every dataset was dropped earlier, a file list was empty, or (blob stage) no dataset had a model file. | Look at datasets.csv (columns status, dropped_stage, dropped_reason) and 'valdo status' to see where the datasets were lost. |
| `shape_mismatch` | rescale, reconstruct, train | Arrays from different runs do not fit together. | Outputs of an older run were mixed with newer ones. | Rerun from preprocess: 'valdo run --from preprocess'. |
| `nan_loss` | train | VAE training produced NaN losses. | Usually a dataset with non-finite amplitudes, or a learning rate that is too high. | Check datasets.csv for scaling_diverged flags; lower advanced.train.learning_rate (e.g. 0.0005) and run 'valdo run --from train'. |
| `no_torch` | any | PyTorch is not installed in this Python environment. | valdo was started from an environment without torch. | Activate the VALDO environment (see environment.yml) and rerun; 'valdo check' lists the packages it found. |
| `no_package` | any | A required Python package is missing. | The active environment is not the VALDO environment. | Activate the VALDO environment (see environment.yml); 'valdo check' lists the packages it found. |
| `gemmi_api` | any | The installed gemmi is too new for VALDO. | VALDO needs gemmi 0.6.7 or older (newer versions renamed functions it uses). | Install gemmi 0.6.7 in the VALDO environment: pip install 'gemmi==0.6.7'. |
| `refine_program` | refine | The refinement program could not be started. | phenix or CCP4 (dimple) is not set up in this shell or batch job. | Load PHENIX/CCP4 before running VALDO (e.g. source .../ccp4.setup-sh), check with 'valdo check', then rerun. |
| `timeout` | refine | Refinement of some datasets took longer than the time limit. | Slow refinements or an overloaded node. | Increase advanced.refine.timeout_min in valdo.yaml and rerun; finished datasets are kept. |
| `pool_broken` | any | A worker process died. | Usually out of memory with many parallel workers. | Lower compute.ncpu or request more memory, then rerun 'valdo run'. |
| `stale_handle` | any | The network file system lost track of a file. | Files were changed or deleted by another process while VALDO was running. | Make sure only one VALDO run uses this work directory, then rerun 'valdo run'. |
| `too_many_files` | any | The process opened too many files at once. | The system limit on open files is low for this many datasets and workers. | Lower compute.ncpu, or raise the limit ('ulimit -n 4096') in the job script. |

<!-- END GENERATED: hints -->

## Results that look wrong

| symptom | likely cause | what to do |
|---|---|---|
| Many datasets dropped at `filter` | Low correlation with the reference after scaling (`min_cc` 0.55): another crystal form, poor data, or a poor reference. | Look at `dropped_reason` in `datasets.csv` and the scaling plot; consider another `reference`. |
| Many datasets dropped at `add_phases_and_blobs` | No model or no phases for them. | Check the `models.*` / `phases.*` patterns; `valdo check` lists missing ones before the run. |
| The top hits are all at one residue | A change common to many datasets (e.g. a reactive residue). | Set `focal_residue.seqid` to that residue to exclude it from the hit list. |
| A known ligand is not found | Weak or partial occupancy; the ligand is present in many datasets (the VAE absorbs it); its dataset was dropped; low resolution. | Check the dataset in `datasets.csv`; open its maps; see "Known limitations" in [USER_GUIDE.md](USER_GUIDE.md). |
| `model_frame_mismatch` flags | A per-dataset model could not be converted into the reindexed setting. | Tags and viewer positions for these datasets may be off; refine them in the setting VALDO used, or check them by hand. |
| `model_map_mismatch` flags (and a note after `add_phases_and_blobs`) | The model does not sit in the 2mFo-DFc density of the dataset's phases: it is not the model those phases came from (typically the apo fallback, placed at another origin or in another indexing; seen for 6 PTP1B datasets). | Blobs are still found, but focal tagging, nearest residues and viewer models are unreliable for these datasets. Give them the model their phases came from (`models.per_dataset`), or use `phases.source: refine`. The value per dataset is `model_fit` in `valdo_state.json` (about 1.5-4 = fits, about 0 = elsewhere). |
| `valdo status` shows stages `out of date` after a complete run | A setting, an input file (or its time stamp) changed after the run, or a newer valdo-easy prepares that stage's inputs differently (e.g. 0.0.6+easy.1 reruns `add_phases_and_blobs`, `tag_blobs` and `report` once). | `valdo run --dry-run` says what changed and what would rerun. |
| A top hit with `Sign` `negative` or `mixed` | The blob search is sign-blind (it ranks \|difference\|): a blob can be density that left (negative), or new and lost density of similar strength (mixed, e.g. a moved side chain). | `Sign` says which side dominates within the blob (at 3 sigma, by a factor 1.5); a bound ligand is usually positive, often with weaker negative density of displaced water. Open the map: green = appeared, red = left. |
| Ranks in the top-hit table skip numbers | Lower-ranked blobs that are symmetry copies of a blob already listed are left out (upstream's duplicate tag only catches copies next in peak order and within 1 A). | Nothing to do; `blobs.csv` keeps every blob and marks copies in `symmetry_copy_of` (the rank of the blob they copy). |
| Datasets dropped at `add_phases_and_blobs` with "needs an origin shift" | They were re-indexed, and in this space group their own phases (and model) cannot be moved into the reference's setting without an origin shift (e.g. P 21 21 21 with a = b). | Use `phases.source: refine`: refinement then happens in the re-indexed setting. |

## Pitfalls of plain upstream VALDO that valdo-easy avoids

If you also use upstream's `valdo.pipeline` directly, keep these in mind; `valdo run` handles
them for you:

- Folder paths needed a trailing `/` (now fixed in the code; see
  [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)).
- Unknown config keys are silently ignored, so a typo runs the default; some schema defaults
  differ from the validated PTP1B values (`wcorr`, `when_opt`, `stdof`, `random_seed`,
  `activation`). valdo-easy rejects unknown keys and writes every value explicitly.
- YAML reads `1e-3` as text; valdo-easy converts it.
- File names are parsed for numbers (`PTP1B-y0049` becomes ID 1 in tag_blobs; phase files are
  found by the first 4 characters as a last resort). valdo-easy renames every dataset to a
  canonical ID first.
- Omitting `vae_sigF_path` with `include_errors: true` silently uses zero errors; with
  `train_fraction: 1.0` training crashes. valdo-easy always sets the paths and validates the
  fraction.
- Without reindexing, the upstream R-free filter matched nothing (fixed). Without a
  refinement summary there was no variant choice; valdo-easy keeps the best-CC variant.
- Upstream's filter keeps every scaled file that has no row in its R-free table, so a dataset
  whose refinement failed was still used to train the VAE. valdo-easy gives each such file a
  row that the filter drops, and stops if `kept_files.txt` or the VAE input contains anything
  else than the chosen variant of a dataset still in the pipeline.
- Datasets without a phase file were silently skipped, and one NaN difference could abort the
  blob stage without a `blob_stats.pkl`. valdo-easy drops such datasets with a recorded reason.
- `bound_models_folder` must be absolute; valdo-easy always writes absolute paths.
- Old output files from a previous run could be picked up again. valdo-easy empties a stage's
  folder before rerunning it.
