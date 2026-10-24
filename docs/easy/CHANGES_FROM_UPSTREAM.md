# Changes from upstream VALDO

**For:** lab members and maintainers who want to know exactly how valdo-easy differs from
upstream VALDO, for example before comparing results or merging upstream updates.
**You need first:** nothing. Some familiarity with upstream's `valdo.pipeline` helps
(see [../pipeline_cli.md](../pipeline_cli.md)).

valdo-easy `0.0.6+easy.1` is upstream VALDO commit `1ec32c6` (version 0.0.6, the merge of
pull request #45 `pipeline-cli`, github.com/Hekstra-Lab/valdo) plus:

1. a new package `valdo/easy/` (the `valdo` command; nothing in it changes the science),
2. a small set of fixes to upstream files (section A), each with a test,
3. behaviour of the `valdo run` workflow that differs from running the upstream stages by
   hand (section B),
4. new files: documentation, tests, packaging and the agent harness (section C).

**Guarantee.** With default settings, valdo-easy gives the same numbers as upstream on the
same inputs and settings. `tests/easy/test_e2e.py::test_every_stage_matches_plain_upstream`
reruns every stage of a finished `valdo run` with plain
`valdo.pipeline STAGE <workdir>/configs/STAGE.yaml --force` and compares every output file
(MTZ columns, NumPy arrays, pickles including the trained VAE weights). The largest relative
difference is 0.0 for all 10 upstream stages.

The same holds on real data. A random 300-file subset of the PTP1B screen
(`numpy.random.default_rng(20261024)`; 298 usable, 293 analysed) was run with `valdo run`
(16 cores, CPU), and then every stage was rerun on the same node with **pristine upstream
code** (`git archive 1ec32c6`, not this branch) from the configs in `<workdir>/configs/`, each
upstream stage reading the previous upstream stage's outputs (only the easy layer's own files
were reused: the canonical-ID input links, the variant-selection list `filter_input.csv`, and
the phase files and models prepared for blob finding). All 1,505 compared files were identical
(standardized, reindexed and scaled MTZs, `reindex_record.pkl`, `scaling_metrics.pkl`, the
preprocess pickles and arrays, the VAE weights and loss history, `recons.npy`, rescaled and
phased MTZs, `blob_stats.pkl` and both tagged tables; largest difference 0). The only
byte-level difference is inside the pickled VAE file, where torch records a process-specific
storage key.

## A. Changes to upstream files

None of these changes the result of a configuration that already worked upstream. The tests
are in `tests/easy/test_core_upstream.py` unless noted.

| file | change | why (what went wrong before) | test |
|---|---|---|---|
| `valdo/__init__.py` | Version read from `valdo/VERSION` instead of `pkg_resources.require('rs-valdo')`. `VAE`, `Scaler`, `Scaler_pool` and the submodules are loaded lazily on first use (PEP 562 `__getattr__`); the public names are unchanged. | `import valdo` failed when the package was not pip-installed or `pkg_resources` was missing, reported the version of another installed copy, and imported torch on every command (slow start of every CLI call). | `test_version_comes_from_version_file`, `test_import_valdo_is_lazy_and_needs_no_pkg_resources`, `test_lazy_public_api_still_works` |
| `valdo/VERSION` | `0.0.6` -> `0.0.6+easy.1` | Version scheme, see below. | `test_version_changelog_and_environment_agree` (test_docs.py) |
| `valdo/helper.py` (`standardize_single_mtzs`, `standardize_input_mtzs`) | `glob(os.path.join(source_path, "*.mtz"))`; a path that `glob` already returned is not joined to `source_path` a second time. | Without a trailing `/` on `source_path` nothing matched; with a relative `source_path` every path was doubled and every file was skipped. | `test_standardize_without_trailing_slash_and_relative_paths` |
| `valdo/helper.py` (`add_phases`, `add_phases_from_pool_map`) | `os.path.join` for the output file. | Without a trailing `/` the phased MTZs were written into the parent folder and no blobs were found. | `test_add_phases_output_folder_without_trailing_slash` |
| `valdo/scaling.py` (`Scaler.batch_scaling`, `Scaler_pool`) | `import os`; `os.path.join` for the scaled MTZs and `scaling_metrics.pkl`. | Same trap: files landed in the parent folder and the next step failed with FileNotFoundError after all the scaling work. | `test_scaler_output_folder_without_trailing_slash` |
| `valdo/tag.py` (`apply_determine_locations`) | `os.path.join(folder, sample + ".mtz")` | Same trap. | `test_determine_locations_folder_without_trailing_slash` |
| `valdo/commandline/pipeline.py` (`run_filter`) | `refine_summary` is read with `file_idx` and `symop` as text; the R-free cut also removes `{file_idx}.mtz`. | IDs wider than 4 digits lost their zero padding and never matched; when reindex was skipped (files are `{file_idx}.mtz`), the R-free cut silently removed nothing. | `test_filter_wide_ids_and_no_reindex_rfree_cut` |
| `valdo/commandline/pipeline.py` | `if __name__ == "__main__": main()` at the end. | `python -m valdo.commandline.pipeline ...` exited 0 without doing anything. | `test_pipeline_module_runs_main` |
| `valdo/commandline/refine.py` | Removed a stray `)` in `output.prefix` (multi-PDB branch). | Every PHENIX output prefix ended in `)`. | `test_refine_output_prefix_has_no_stray_parenthesis` |
| `setup.py` | Console script `valdo = valdo.easy.cli:main`; `package_data` for `VERSION` and `refine_drug.eff`; `tests` excluded from the packages; `dev` extra (pytest). Upstream console scripts `valdo.pipeline`, `valdo.refine`, `valdo.occupancy` unchanged. | `pip install .` (non-editable) lost the data files; the new command needs an entry point. | `test_setup_declares_console_script_and_package_data` |
| `MANIFEST.in` | Also includes `valdo/commandline/refine_drug.eff`. | Same as above (source distributions). | same |
| `README.md` | A short "New here?" section at the very top pointing to the valdo-easy documentation. Upstream text unchanged below it. | Discoverability. | `test_readme_points_new_users_to_the_quickstart` (test_docs.py) |

Not changed although known to be broken: `valdo/temp.py` has a syntax error upstream. Nothing
imports it, so it is left as it is.

### Version scheme

`0.0.6+easy.1` is a PEP 440 *local version*: upstream base 0.0.6 plus valdo-easy release 1.
PyPI refuses local versions, so the upstream PyPI publishing workflow cannot accidentally
release this branch. `valdo --version` prints the version and the upstream base commit
(`valdo 0.0.6+easy.1 (upstream VALDO 0.0.6, commit 1ec32c6)`). The next valdo-easy release on
the same upstream base is `0.0.6+easy.2`; after merging a new upstream release X.Y.Z it is
`X.Y.Z+easy.1`.

## B. How `valdo run` differs from running the upstream stages by hand

The upstream stage functions do the scientific work unchanged. The differences below are in
what valdo-easy feeds them and what it does between stages. All of them are recorded in
`<workdir>/datasets.csv` or the stage logs, and every generated upstream config is kept in
`<workdir>/configs/<stage>.yaml`.

| area | upstream (by hand) | valdo-easy | why |
|---|---|---|---|
| Dataset names | File names are parsed per stage (first 4 characters, digits in the name). `PTP1B-y0049` parses as ID 1. | Every input file gets a canonical zero-padded ID (`0001`, `0002`, ...) in natural sort order of the dataset names; upstream stages only see IDs. `datasets.csv` keeps the mapping. | Removes the naming traps; reports show both names. |
| Column names | Hard-coded `F-obs`/`SIGF-obs` in reindex and filter. | Other amplitude/sigma names are renamed to `F-obs`/`SIGF-obs` in renamed copies under `inputs/mtz/`. | Data from other pipelines work without editing files. |
| Other crystal forms | Processed (and usually fail to scale). | Files whose space group differs from the majority are excluded at input with a warning (PTP1B: 3 files in P 32 2 1). | One VAE cannot model two crystal forms. |
| Unreadable files | A stage crashes or silently skips them. | Excluded at input with the reason (`dropped_stage = input`). | Clear accounting. |
| Variant choice without refinement | `filter` needs a refinement summary; without one, both indexing variants of an ambiguous dataset go on. | One variant per dataset: the lower refined R-free when refinement ran (upstream rule), otherwise the best reindexing CC (`best_symop`). The all-NaN and `min_cc` checks always run. | Never two copies of one dataset in the VAE; the filter never silently does nothing. |
| Datasets lost after scaling | `run_filter` keeps every scaled file that has no row in the refinement summary, so a dataset whose refinement failed still reaches `kept_files.txt` and trains the VAE. | Every scaled file of a dataset no longer in the pipeline gets a summary row with `Rf_final` 99 (upstream's own R-free rule drops it; the note says why); filter and preprocess stop if anything else than the chosen variants of the remaining datasets gets through. Upstream code unchanged. | Only datasets that passed every stage train the VAE (`test_failed_refinement_keeps_the_dataset_out_of_the_vae`). |
| Models and phases of reindexed datasets | Not handled (upstream's PTP1B run refined every variant). | Per-dataset models, phase columns and known bound models are transformed into the setting of the kept variant. Phases and the dataset's own model are moved by the same operator: the reindexing operator when it maps the space group onto itself (PTP1B's P 31 2 1: always), otherwise the member of its point-group coset that does (same amplitudes; e.g. y,x,-z for gemmi's I 41 law -x,y,-z); a dataset for which every choice needs an origin shift is dropped with that reason. A model that cannot be transformed is used as it is and the dataset is flagged `model_frame_mismatch`. The shared apo model is assumed to be in the reference setting. | Phases and tags must match the reindexed data, and the model must sit in the map of the phases it is used with (`test_core_frames.py`). |
| Missing model or phases | `find_phase_file` silently skips the dataset. | Dropped before the stage with the reason. Under 90% phase coverage is flagged `low_phase_coverage`. | No silent losses. |
| Non-finite differences | One NaN in `diff`/`SIGF` empties a file in add_phases_and_blobs and can abort the stage. | Such datasets are dropped after rescale with the reason. | The run continues; the loss is recorded. |
| Stale outputs | Upstream skips a stage whose output exists (unless `--force`) and never removes old files. | Before a stage runs, its own output folder is emptied (only folders valdo created, never through symlinks, never outside the work directory); `_force` is passed. | Old files cannot mix with new ones. |
| Threads | Every library uses all cores of the machine. | `OMP/MKL/OPENBLAS_NUM_THREADS` = `compute.ncpu` (if unset); `torch.set_num_threads(ncpu)` before train and reconstruct. | Fair use of shared nodes; makes training reproducible for a given ncpu. |
| Refinement | `valdo.refine`: PHENIX, one dataset after another; R factors collected by a separate script. | `phases.source: refine`: PHENIX (same `refine_drug.eff`) or CCP4 dimple, `compute.ncpu` datasets at once, per-dataset logs, timeout and resume; `refine_summary.csv` feeds the filter directly. dimple is given a cleaned copy with only F/SIGF/FreeR_flag columns. | Usability; dimple for groups without a PHENIX licence. |
| Report | Notebooks / `PTP1B_pipeline/compute_valdo_metrics.py`. | `report/` with ranked hits, maps per hit, Coot/PyMOL scripts, QC plots and the dataset funnel. The top-hit table shows the most positive and most negative difference density of each blob with a sign summary, and leaves out symmetry copies of a blob already listed (upstream's `duplicate` tag misses some); `blobs.csv` keeps upstream's full ranked list, copies marked. | New; read-only, does not change any upstream output or ranking. |
| Settings | Unknown config keys are silently ignored; schema fallbacks differ from the validated PTP1B values. | Unknown keys are errors with a suggestion; every value is written explicitly from the validated PTP1B templates. | A typo cannot silently run a default. |

## C. New files (not in upstream)

| path | what |
|---|---|
| `valdo/easy/` | The accessibility layer: `cli.py`, `project.py`, `scan.py`, `ids.py`, `check.py`, `hints.py`, `stages.py`, `runner.py`, `refine.py`, `report.py`, `maps.py`, `jobscript.py`, `docgen.py`. |
| `tests/easy/` | Test suite (unit tests, a synthetic toy screen end to end, the upstream-equivalence test). |
| `docs/easy/` | QUICKSTART, USER_GUIDE, CONFIG_REFERENCE (generated), TROUBLESHOOTING, WYNTON, DESIGN, this file. |
| `AGENTS.md`, `integrations/claude-code/skills/valdo/SKILL.md` | Instructions for LLM agents. |
| `CHANGELOG.md`, `environment.yml` | Release notes; the environment validated on UCSF Wynton. |

## Merging a new upstream release

1. Merge upstream into the `valdo-easy` branch. Conflicts can only occur in the files of
   section A.
2. For each row of section A, check whether upstream fixed the problem itself; if so, take
   upstream's version and keep the test.
3. Update `UPSTREAM_BASE`/`UPSTREAM_VERSION` in `valdo/easy/__init__.py`, `valdo/VERSION`,
   `CHANGELOG.md` and this file.
4. Run `python -m pytest tests/easy` on a CPU node. `test_every_stage_matches_plain_upstream`
   shows whether the easy layer still reproduces upstream exactly.
