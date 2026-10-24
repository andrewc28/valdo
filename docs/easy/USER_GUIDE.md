# VALDO user guide

**For:** crystallographers using VALDO on their own screens who want to understand what it
does, which choices matter and how to judge the hits.
**You need first:** a working installation and a first project
([QUICKSTART.md](QUICKSTART.md)); the terms MTZ, map coefficients, reindexing, VAE, blob and
z-score are defined there.

## What VALDO does

A fragment screen gives hundreds or thousands of datasets of the same crystal form. Most
contain no bound fragment; a few do. VALDO (VAE-assisted ligand discovery, Hekstra lab) uses
the whole screen to learn what an "ordinary" dataset looks like and then points to the
places where an individual dataset deviates from it.

1. All datasets are put on a common footing: same indexing, same scale.
2. A VAE (a small neural network) is trained on the amplitudes of all datasets. Because most
   datasets are unbound, it learns to reproduce the unbound crystal.
3. For each dataset, the VAE's reconstruction is subtracted from the measured amplitudes.
   With phases from a refined model, this difference becomes a difference map.
4. Strong connected regions of that map (blobs) are found, tagged (near the focal residue,
   on a known ligand, symmetry duplicates) and ranked.

```
 MTZ files ─► standardize ─► reindex ─► scale ─► [refine] ─► filter ─► preprocess ─► train
                                                                                      │
 report ◄── tag_blobs ◄── add_phases_and_blobs ◄── rescale ◄── reconstruct ◄──────────┘
```

`valdo stages` prints the same table with folder names. `reindex` runs only when the crystal
form allows alternative indexing; `refine` only when `phases.source` is `refine`.

## The project file

`valdo.yaml` holds every setting. `valdo init` writes it with a comment on each line, in
three parts: your project (data, models, work folder), what init detected (check it), and
advanced settings (the validated PTP1B values). Typos are errors, not silent defaults.
Every setting is listed in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

The work folder holds everything valdo writes. Your data files are never changed: valdo
reads them through links in `<workdir>/inputs/`.

## Dataset names and IDs

Every input file gets a number (`0001`, `0002`, ...) in the sorted order of the dataset
names; the stages only see those numbers. The dataset name is the file name without the
ending that all files share (`PTP1B-y0001_mrflagsref_idxs.mtz` becomes `PTP1B-y0001`).
`<workdir>/datasets.csv` lists, for every dataset: ID, original name, input file, model,
whether it is still in the pipeline (`status` ok or dropped), where and why it was dropped,
and flags:

| flag | meaning |
|---|---|
| `reference` | the dataset everything is reindexed and scaled to |
| `reindexed` | the kept indexing variant is not the original one |
| `ambiguous_indexing` | both indexings correlated about equally well with the reference |
| `model_frame_mismatch` | the model could not be brought into the reindexed setting; treat this dataset's tags and viewer positions with care |
| `fallback_model` | no own model; the apo model was used |
| `model_map_mismatch` | the model does not sit in the 2mFo-DFc density of the dataset's phases (mean below 1 sigma at its atoms; a model that fits gives about 1.5-4). Blobs are found as usual, but focal tagging, nearest residues and viewer models are unreliable: give this dataset the model its phases came from. Seen on PTP1B for datasets without their own model, whose phases came from models at another origin than the apo model |
| `cell_outlier` | unit cell more than 3% (or 3 degrees) from the median |
| `low_resolution` | worse than 3.5 A |
| `low_phase_coverage` | phases for less than 90% of the reflections |
| `scaling_diverged` | scaling gave no usable values (the dataset is dropped by filter) |

Adding or removing data files renumbers the datasets, and everything is recomputed.

## The stages in plain words

For each stage: why it exists, what can go wrong, and what to look at. Each stage writes its
full output to `<workdir>/logs/<stage>.log`.

**1. standardize** (`01_standardize/`). Copies each dataset's amplitudes and sigmas into a
uniform file named by its ID and removes reflections without a value. *Can go wrong:* a file
is damaged or lacks the columns (it is excluded with the reason, before anything runs).
*Look at:* `valdo check` before the run.

**2. reindex** (`02_reindex/`, only when needed). In space groups such as P 31 2 1 with this
kind of cell, a crystal can be indexed as `h,k,l` or `-h,-k,l`; both are equally valid, but
datasets must agree before they can be compared. Every dataset is compared with the
reference under each possible indexing, and the one with the highest correlation (CC) wins.
When two indexings are about equally good (difference below `advanced.reindex.cc_min_dif`),
both are kept until the filter stage chooses one. *Can go wrong:* datasets with very weak or
incomplete data correlate poorly under every indexing. *Look at:* the CC histogram in the
report; the `reindexed` and `ambiguous_indexing` flags. Twinned crystals (where both
indexings are present in one crystal) are not handled specially: their correlations are
lower and they may be dropped by filter.

**3. scale** (`03_scale/`). Puts every dataset on the reference's scale (overall scale and
anisotropic B factor). *Can go wrong:* a dataset with too few reflections in common with the
reference does not converge (`scaling_diverged`). *Look at:* the scaling QC plot.

**4. refine** (`04_refine/`, only with `phases.source: refine`). Refines the model against
each dataset (PHENIX or CCP4 dimple), several datasets at once, to get phases and R-free.
Each dataset has its own folder with a log in `04_refine/runs/`. A dataset whose refinement
fails is dropped with the reason; the others continue. A stopped run resumes where it left
off, and only datasets whose inputs changed are refined again. To retry failed datasets (for
example after fixing their cause), run `valdo run --from refine` (in a batch job: add
`--from refine` to the `valdo run` line of the job script): datasets that refined
successfully are reused and only the failed ones run again. *Look at:*
`04_refine/refine_summary.csv` (R-work and R-free per dataset).

**5. filter** (`05_filter/`). Keeps one indexing variant per dataset (the lower R-free after
refinement, otherwise the higher reindexing CC) and drops datasets that (a) have an R-free
above `advanced.filter.max_rfree` (refinement only), (b) did not scale, or (c) correlate with
the reference below `advanced.filter.min_cc` (0.55). Datasets that left the pipeline before
(e.g. their refinement failed) are kept out too: the table the filter reads
(`configs/filter_input.csv`) gives each of their scaled files a row it drops. *Look at:*
`datasets.csv`, rows with `dropped_stage` filter.

**6. preprocess** (`06_preprocess/`). Builds the VAE's input table: the reflections measured
in every kept dataset. A single very incomplete dataset shrinks this common set for everyone,
so very low-resolution or incomplete datasets matter here. *Look at:* `logs/preprocess.log`
(number of common reflections).

**7. train** (`07_train/`). Trains the VAE on all kept datasets (80% for training, 20% to
check it). The settings are the validated PTP1B values. *Can go wrong:* NaN losses (usually a
bad dataset or a too-high learning rate). *Look at:* the loss plot in the report: both curves
should fall and level off.

**8. reconstruct** (`08_reconstruct/`). The trained VAE predicts every dataset's
amplitudes.

**9. rescale** (`09_rescale/`). Converts the predictions back to the data scale and computes
the difference amplitudes (measured minus predicted). Datasets with non-finite differences
are dropped here (they would otherwise break the next stage).

**10. add_phases_and_blobs** (`10_phases_blobs/`). Combines the differences with the phases,
weights them (large and uncertain differences are down-weighted), computes the difference
map, blurs it slightly and finds blobs above `advanced.blobs.cutoff` (3.5) sigma. Datasets
without a model or without phases are dropped here with the reason. *Look at:* the number of
blobs per dataset in the report.

**11. tag_blobs** (`11_tag/`). Marks blobs within `focal_residue.radius` of the focal residue
(these are not reported as hits), blobs that are symmetry copies of another blob, and (if you
gave known bound models) blobs on a known ligand. Upstream's duplicate tag only compares a
blob with the next one in peak order (within 1 A), so some symmetry copies stay in the list;
the report marks them (`symmetry_copy_of` in `blobs.csv`) and leaves them out of the top hits.

**12. report** (`report/`). See "Reading the report" below.

## Choosing where the phases come from

VALDO's difference map needs phases for every dataset, from a model refined against that
dataset (or a very similar one).

- **`input_mtz`** (default when available): the data files already contain map coefficients,
  as XChem/PanDDA input files do (`2FOFCWT`/`PHI2FOFCWT`, `FOFCWT`/`PHIFOFCWT`). Fastest.
  Give the models that produced those phases (`--models`), so that tagging and viewing use
  the same coordinates.
- **`refine`**: valdo refines every dataset. `dimple` (CCP4, free for academic use) runs
  rigid-body and restrained refinement; `phenix` uses `phenix.refine` with the packaged
  `refine_drug.eff`, as upstream VALDO did. PHENIX must come from your own licence and
  installation. Refinement takes about a minute per dataset per core.
- **`existing`**: you refined every dataset already; point `--existing` at the refined MTZs
  (`'/path/{name}/final.mtz'`). The column names are detected.

When a dataset was reindexed, its own model and phases are converted into the new indexing
automatically, both with the same operator: one that maps the space group onto itself (for
most space groups, including PTP1B's P 31 2 1, the reindexing operator itself; for
I 41, for example, gemmi's -x,y,-z becomes y,x,-z). Where every choice would need
an origin shift (e.g. P 21 21 21 with a = b), the dataset is dropped with that reason;
use `phases.source: refine` for such a screen. The single apo model (`models.apo`) is assumed
to be in the reference's setting and is never converted.

## Reading the report

`<workdir>/report/report.html` lists the kept blobs ranked by `score` (the blob's integrated
difference density; upstream's ranking). For each top hit it gives the dataset (ID and
original name), the blob (`0042_0 #2` = the second blob of dataset 0042 in variant 0), peak
height (`peakz`, in sigma), volume, the nearest residue, and the most positive and most
negative value of the unblurred difference map within the blob ("Diff. max / min", in sigma;
positive = new density, negative = something left). The blob search ranks |difference|, so a
blob can hold both: `Sign` says which side dominates (positive or negative when only that side
reaches 3 sigma or it is at least 1.5 times stronger than the other; mixed when both reach
3 sigma and neither dominates; weak when neither reaches it). A bound ligand is usually
positive, often with weaker negative density of the waters it displaced. With known bound
models, "Known ligand" says whether the blob overlaps the ligand of that dataset's bound model.
Symmetry copies of a blob already listed are left out (so ranks can skip numbers); `blobs.csv`
has every blob.

**A blob means "something changed here", not "a ligand is here".** Things that also give
blobs: moved side chains, alternative conformations, changed waters or ions, radiation
damage, a different cryoprotectant, crystal contacts, and errors in the data. To judge a hit:

1. Open it in Coot or PyMOL (`report/view_top_hits.py` or `.pml`). The difference map is
   contoured at +3 sigma (green) and -3 sigma (red).
2. Look at the shape. A ligand gives a compact, positive blob of a plausible size, next to
   the protein surface, not overlapping existing atoms. Paired positive and negative density
   next to a side chain usually means it moved.
3. Check the flags (`low_resolution`, `cell_outlier`, `model_frame_mismatch`,
   `model_map_mismatch`, `fallback_model`, `ambiguous_indexing`). Hits in flagged datasets
   deserve extra caution.
4. Compare with the 2mFo-DFc map from the dataset's own refinement and, ideally, refine a
   ligand into the density.

Known limitations:

- **The VAE can absorb a frequent change.** It learns what is typical for the screen. A
  ligand (or any change) present in many datasets, such as a common buffer molecule or a
  fragment that binds in a large fraction of the screen, becomes "typical" and its blobs get
  weaker. VALDO is best at finding changes that are rare across the screen.
- **Ranking by `score` favours large blobs.** `score` adds up the blurred |difference| over the
  whole blob, so a large blob outranks a smaller one of the same height, and a small ligand can
  rank lower than its peak suggests. `blobs.csv` also has `peakz` to sort by. On the full PTP1B
  screen the two rankings did about equally well (AUC 0.968 by `score`, 0.976 by `peakz`, over
  13,974 blobs, 129 of them on known ligands). `score` put more known ligands in the global
  top 100 (65 against 50 of 110 datasets); `peakz` made the ligand the dataset's top blob
  slightly more often (105 against 103).
- **Few datasets.** The VAE needs many datasets to learn the unbound crystal; below about 100
  usable datasets `valdo check` warns that rankings are less reliable.
- **Twinning.** Twinned datasets correlate worse with the reference, may be reindexed
  inconsistently and are often dropped by filter. VALDO does no twin refinement; check
  twinned datasets separately.
- **One crystal form per project.** Datasets in another space group or with a very different
  cell are excluded or dropped; run them as a separate project.
- **Focal residue.** Blobs near `focal_residue.seqid` are removed from the hit list on
  purpose (PTP1B's reactive Cys215 changes in many datasets). Set it to `null` to see them.
- **Known-ligand validation is descriptive.** The section on known bound models reports how
  VALDO ranked those ligands; it is not a calibrated error rate.

## Resuming and rerunning

- `valdo run` skips finished stages. A stage reruns only if it failed, was interrupted, or
  something it depends on changed (its settings, its inputs, or an earlier stage's results).
  Before a stage reruns, its output folder is emptied, so old and new files never mix.
- `valdo run --dry-run` shows the plan and why each stage would (or would not) run.
- `valdo run --from STAGE` reruns from that stage on; `--until STAGE` stops after it;
  `--only STAGE` reruns one stage; `--force` reruns even up-to-date stages.
- Only one `valdo run` can use a work folder at a time (a lock file `<workdir>/.lock`, touched
  every 30 seconds while the run lives). A lock left behind by a killed job counts as abandoned
  5 minutes after its last sign of life, and the next `valdo run` replaces it (at once if the
  run was on the same computer and its process is gone). Until then a new run stops with
  `locked`; `valdo status` shows the lock. Do not delete `.lock` by hand.
- valdo never empties a stage folder that is a symbolic link (error `unsafe_path`); to keep
  the outputs on another disk, set `workdir` there instead.
- Every stage's exact upstream settings are in `<workdir>/configs/<stage>.yaml`. Upstream's
  own tool can rerun any stage from them:
  `valdo.pipeline scale <workdir>/configs/scale.yaml --force`. This gives the same result as
  `valdo run` (tested).
- valdo only ever deletes inside the work folder, and only folders it created itself.

## GPUs

valdo runs entirely on CPUs. Training is the only stage a GPU speeds up; on 16 CPU cores it
took 22 minutes for the full PTP1B screen (1,626 datasets, 500 epochs; upstream reported 2
minutes on an RTX A6000 GPU), so a GPU is optional.

- With a GPU visible, training uses it automatically. `valdo jobscript --gpu` requests one
  (Wynton: see [WYNTON.md](WYNTON.md)).
- Results on GPU and CPU differ slightly (floating-point order); for exactly reproducible
  results use the CPU: `CUDA_VISIBLE_DEVICES='' valdo run` (CPU job scripts do this).
- A model trained on a GPU cannot be loaded on a CPU-only node; if reconstruct fails with
  "Attempting to deserialize object on a CUDA device", run `valdo run --from train` on the
  CPU.

## Using valdo from scripts or AI agents

Every command takes `--json`: the output becomes one JSON object per line, the last line
being the result (with `ok`, an `error` explanation if something went wrong, and `next`,
suggested next commands). [../../AGENTS.md](../../AGENTS.md) describes this for AI coding
agents.
