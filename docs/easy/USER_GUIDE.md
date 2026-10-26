# VALDO user guide

**For:** crystallographers using VALDO on their own screens who want to understand what it
does, which choices matter and how to judge the hits.
**You need first:** a working installation and a first project
([QUICKSTART.md](QUICKSTART.md)). The terms MTZ, map coefficients, reindexing, VAE, blob and
z-score are explained in [Words you will meet](#words-you-will-meet).

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

## Words you will meet

- **MTZ file**: the standard file of merged reflection data (one per dataset), with columns
  such as `F` (structure-factor amplitude) and `SIGF` (its uncertainty).
- **Map coefficients**: amplitude and phase columns from refinement (for example `2FOFCWT` /
  `PH2FOFCWT` and `FOFCWT` / `PHFOFCWT`, or `FWT` / `PHWT` and `DELFWT` / `PHDELWT`) from which
  an electron-density map is computed. VALDO needs phases for every dataset.
- **Reindexing**: in some crystal forms the same crystal can be indexed in two (or more)
  equivalent ways. Before comparing datasets, VALDO puts them all in the same indexing as a
  reference dataset.
- **VAE (variational autoencoder)**: a small neural network. VALDO trains it on all datasets
  of the screen so that it learns what a typical (mostly unbound) dataset looks like. For each
  dataset it then predicts the "expected" amplitudes; the difference between measured and
  expected amplitudes highlights what is unusual about that dataset.
- **Blob**: a connected region of strong difference density found in that difference map. A
  blob means "something changed here". It is not proof of a ligand.
- **z-score / sigma**: how many standard deviations above the map's average a value is.
  `peakz` is the blob's highest value in those units.

## Setting up a project

`valdo init DATA` reads only the file headers (1,680 PTP1B files in 8 seconds), writes
`valdo.yaml` and tells you what it decided and why. `DATA` is a folder with MTZ files, a quoted
glob (`'/path/*/final.mtz'`) or a `.txt` file listing the paths. The options:

- `--apo FILE`: one model of the unbound protein, used for every dataset. It must be in the
  same indexing as the reference dataset (by default your first dataset), so the safest choice
  is the reference dataset's own model.
- `--models PATTERN`: one model per dataset, for example `--models '/path/models/{name}.pdb'`
  (`{name}` is replaced by each dataset's name), or simply the folder that holds them. Give
  `--apo` as well to have a fallback for datasets without their own model (`valdo check` lists
  datasets that have no model). When the phases come from your data files, each dataset's model
  should be the one its phases came from: `valdo run` flags datasets whose model does not sit in
  their map (`model_map_mismatch`; on PTP1B, 6 of the 7 datasets without their own model).
- `--focal-seqid N`: a residue whose surroundings should not be reported (for PTP1B, the
  reactive Cys215 always changes, so the PTP1B run used `--focal-seqid 215`).
- `--bound-models FOLDER`: optional, models of datasets known to have a ligand. The report
  then says how well VALDO found them (for information only).
- `--phases`, `--program`, `--ccp4`, `--phenix`, `--existing` and `--twin-law`: see
  [Choosing where the phases come from](#choosing-where-the-phases-come-from).
  `valdo init --help` lists the others (`--workdir`, `--reference`, `--ncpu`, ...).

`valdo check` then looks at your computer (packages, CPUs, disk space), your files (readable,
columns, space group, cell, resolution, a model and phases for every dataset) and your
settings. Every message says what is wrong, why it matters and how to fix it. **Errors** stop
`valdo run`; **warnings** do not. `valdo check --deep` also reads all reflection data (slower;
finds missing values and duplicated reflections). A misspelled setting is an error with a
suggestion: `advanced.train.epoch: unknown setting. Did you mean 'advanced.train.epochs'?`

## The project file

`valdo.yaml` holds every setting. `valdo init` writes it with a comment on each line, in
three parts: your project (data, models, work folder), what init detected (check it), and
advanced settings (the validated PTP1B values; leave them alone unless you have a reason).
Typos are errors, not silent defaults. Every setting is listed in
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

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
indexings are present in one crystal) correlate less well under either indexing and may be
dropped by filter; with refinement they can be refined as twinned (see
[Twinned crystals](#twinned-crystals)).

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
refinement, otherwise the higher reindexing CC; with twin refinement the lower twin fraction,
see [Twinned crystals](#twinned-crystals)) and drops datasets that (a) have an R-free
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
- **`refine`** (`--phases refine --program dimple --ccp4 /path/to/ccp4-9`, or
  `--program phenix --phenix /path/to/phenix-2.2-6143`): valdo refines every dataset. `dimple`
  (CCP4, free for academic use) runs rigid-body and restrained refinement; `phenix` uses
  `phenix.refine` with the packaged `refine_drug.eff`, as upstream VALDO did (repaired: as
  shipped, PHENIX ignored the whole file; see below). PHENIX must come from your own licence
  and installation. Refinement takes about a minute per dataset per core with dimple and about
  6-11 with phenix.refine (batch runs on Wynton: median 6.4 minutes with PHENIX 2.2.1 and 11.3
  with 1.21.2). Tell valdo where the program is installed with one line in `valdo.yaml` (see
  [Refinement programs](#refinement-programs)); twinned crystals can be refined as twinned
  (see [Twinned crystals](#twinned-crystals)).
- **`existing`**: you refined every dataset already:
  `--phases existing --existing '/path/{name}/final.mtz'` (a pattern with `{name}`, a folder or
  a glob). The column names are detected.

When a dataset was reindexed, its own model and phases are converted into the new indexing
automatically, both with the same operator: one that maps the space group onto itself (for
most space groups, including PTP1B's P 31 2 1, the reindexing operator itself; for
I 41, for example, gemmi's -x,y,-z becomes y,x,-z). Where every choice would need
an origin shift (e.g. P 21 21 21 with a = b), the dataset is dropped with that reason;
use `phases.source: refine` for such a screen. The single apo model (`models.apo`) is assumed
to be in the reference's setting and is never converted.

## Refinement programs

With `phases.source: refine`, valdo needs PHENIX (`phenix.refine`) or CCP4 (`dimple`). Tell
valdo once where the program is installed, in part 1 of `valdo.yaml`:

```yaml
programs:
  phenix: /path/to/phenix-2.2-6143    # used with phases.program: phenix
  ccp4: /path/to/ccp4-9               # used with phases.program: dimple
```

That one line is enough for the whole pipeline: `valdo check` tests that installation, the
refine stage of `valdo run` uses it (on your computer or in a batch job), and the script from
`valdo jobscript` needs no setup line for it. `valdo init` takes the same with `--phenix PATH`
or `--ccp4 PATH`; without them, it looks for the program on PATH and, if that installation has
its setup script, writes its folder for you (with the comment "found on PATH by valdo init").
`valdo init --force` keeps the `programs` settings of the `valdo.yaml` it replaces unless they are
given again (it does not keep `phases.twin_law`, and says so). The separate upstream commands
`valdo.refine` and `valdo.occupancy` (which runs `phenix.maps`) do not read `valdo.yaml`: before
using them, set up PHENIX in your shell (`source /path/to/phenix-2.2-6143/phenix_env.sh`).

- **What to give:** the installation folder (PHENIX: the folder with `phenix_env.sh`, e.g.
  `phenix-2.2-6143`; CCP4: the folder with `bin/ccp4.setup-sh`, e.g. `ccp4-9`), or the setup
  script itself, its `bin` folder or the program. A pasted setup line also works
  (`phenix: source /opt/phenix-2.2-6143/phenix_env.sh` is read as the script's path). Relative
  paths are relative to `valdo.yaml`; `~` works. csh scripts (`ccp4.setup-csh`,
  `phenix_env.csh`) cannot be used. On a cluster, use a folder the compute nodes can see (your
  home or a lab folder), not one on your laptop.
- **How it works:** for the refine stage, valdo sources the installation's own setup script
  (`phenix_env.sh`, `bin/ccp4.setup-sh`) in a separate `bash` (without your profile or rc
  files), reads back the environment that script makes, and runs the program with it. Your
  shell, the batch job and valdo's own Python are not changed, and valdo's Python settings
  (`PYTHONPATH`, `PYTHONHOME`) are not passed to PHENIX or CCP4. Sourcing a script runs it:
  point these settings only at installations you trust (your own or your lab's).
- **Without the setting** (`null`, the default) the program must be on PATH, as before: set up
  PHENIX or CCP4 in your shell and in the job script (`valdo jobscript --env-setup '...'`).
- **Which version ran:** `valdo check` names the program, version and folder
  (`refine_program`); every refinement records it (`program_version` in
  `04_refine/refine_summary.csv`, and the first lines of each `04_refine/runs/<stem>/refine.log`),
  and so does the report (with the number of refined variants per version when there are
  several). If the version set up now differs from the one that made the existing refinements
  (moving the same version to another folder is no change), `valdo check` and `valdo status`
  say so (`program_changed`): datasets that were already refined are not refined again just
  because of that. To refine every dataset with the new version, delete
  `<workdir>/04_refine/runs` and run `valdo run --from refine`.
- **What valdo gives phenix.refine** (tested with PHENIX 1.21.2-5419 and 2.2.1-6174): the model,
  the MTZ, `output.prefix`, the `.eff`, the data columns as `miller_array.labels.name=F-obs,SIGF-obs`
  (PHENIX 1.20 and later ignore `xray_data.labels`; with an MTZ that also holds intensities,
  PHENIX would otherwise refine against those), `xray_data.r_free_flags.generate=True` when the
  MTZ has no R-free flags (or `miller_array.labels.name=<flags>` when it has several flag columns),
  `xray_data.r_free_flags.ignore_pdb_hexdigest=True`, the twin law, your `extra_args`, and
  `--overwrite` last (PHENIX 1.x rejects parameters after it). `ignore_pdb_hexdigest`: a start
  model that PHENIX refined before against another R-free set (every PanDDA input model, for
  example) carries a hexdigest of that set, and PHENIX stops at the mismatch unless told to
  ignore it, which is what dimple does without asking. The R-free set is then this dataset's
  own (or a new one), not the one the start model was refined against. A setting you give in
  `extra_args` (labels, `ignore_pdb_hexdigest`) replaces valdo's.
- **The packaged `.eff`:** upstream's `refine_drug.eff` names an R-free file on the authors'
  cluster, sets `output.quiet` (not a phenix.refine parameter) and never closes its `refine {`
  scope, so PHENIX did not read it at all. valdo writes a repaired copy to
  `04_refine/phenix_refine.eff` (those three lines fixed; the log of the refine stage says so)
  and leaves the upstream file unchanged. With `advanced.refine.eff` set to your own file,
  valdo uses that as it is (it stops only if its braces do not balance).
- **Installing:** valdo installs neither program. PHENIX needs a licence (free for academic
  users, phenix-online.org); CCP4 is free for academic use (ccp4.ac.uk). Install them yourself
  or ask your lab or cluster administrators, then give valdo the folder.

## Twinned crystals

In some lattices (for example P 31 2 1, P 3, P 4, or P 21 21 21 with a close to b) a crystal can
be twinned: two orientations of the lattice overlap, related by a *twin law* written in hkl
notation (e.g. `-h,-k,l`). The twin fraction is the share of the second orientation (0 =
untwinned, 0.5 = perfect twin). With `phases.source: refine`, valdo can refine every dataset as
twinned:

```yaml
phases:
  source: refine
  program: dimple        # or phenix
  twin_law: -h,-k,l
```

(`valdo init DATA --apo apo.pdb --phases refine --program dimple --twin-law -h,-k,l`;
`--twin-law=-h,-k,l` works too.) `twin_law: auto` stands for the only twin law the reference
lattice allows, when it allows exactly one (e.g. P 31 2 1, P 3 2 1, P 31 1 2, P 41, P 61):
`valdo check` and `valdo init` say which law that is. They refuse `auto` when the lattice allows
none (e.g. P 41 21 2, P 61 2 2, P 21 21 21 with three different cell edges) or several (P 3, P 31,
P 32, pseudo-cubic cells); then give the law itself.

- **Which law:** `valdo check` lists the twin laws the reference dataset's space group and cell
  allow (finding `twin_law`; in `valdo check --json`: `inputs.twin.possible`), and `valdo init`
  writes them in the comment of `phases.twin_law`. Whether your crystals are twinned (and by
  how much) is best judged with the usual tools (phenix.xtriage, ctruncate). A law that is a
  symmetry operation of the crystal, or that the lattice does not allow, is an error. Laws that
  differ only by a symmetry operation are the same twin law: in P 31 2 1, REFMAC writes
  `-h,-k,l` as `-K, -H, -L`. The law is written in the reference dataset's indexing. Every
  indexing variant of a dataset is refined; in a variant re-indexed by an operator that does
  not commute with the law (only lattices with several twin laws, e.g. pseudo-cubic cells), the
  same twinning has another law there, and valdo refines that one (`twin_law` in
  `04_refine/refine_summary.csv` shows it).
- **phenix:** `xray_data.twin_law=<law>` is added to every `phenix.refine` run, which refines
  that law's twin fraction (a bare `twin_law=` is ambiguous to PHENIX, which also has
  `neutron_data.twin_law`). phenix.refine keeps the law for every dataset; an untwinned one
  refines to a twin fraction near 0 (an exact 0 is counted in the refine stage's message and the
  report). Checked with PHENIX 1.21.2 and 2.2.1 on 48 synthetic twinned datasets (twin fractions
  0-0.49): every refined fraction was within 0.008 of the true one, and R-free was 0.08-0.09
  whatever the twin fraction (without the twin law it rose from 0.08 to 0.28; on the untwinned
  ones, below 0.05, it was 0.090 with the law against 0.080 without). PHENIX's twin-refined
  map coefficients can be missing for part of the reflections (there: 81-91 % of the
  reflections had them in 8 of the 48 datasets, twin fractions 0.33-0.43; all of them without
  twin refinement); valdo flags such datasets `low_phase_coverage`.
- **dimple:** dimple has no twin-law option. valdo passes `--refmac-key twin`, which dimple
  gives to its jelly-body and final restrained REFMAC runs (not to the rigid-body run or to
  molecular replacement). REFMAC then finds the twin laws itself, keeps those with a twin
  fraction of at least 0.07 and refines their fractions; valdo compares the law(s) REFMAC kept
  with `phases.twin_law`. A dataset for which REFMAC kept no twin domain was refined as
  untwinned (`twin_fraction` 0, with a note).
- **Results:** `twin_law`, `twin_fraction` and `twin_note` per dataset in `datasets.csv` (for
  the kept indexing variant) and in `04_refine/refine_summary.csv` (every variant); the
  report's "Refinement" section gives the median and range. A law other than the configured
  one, or several laws, are noted in `twin_note`. The twin fraction is read from the refined
  model (REFMAC: REMARK 3 "TWIN DETAILS" and the mmCIF; phenix.refine: REMARK 3 "TWINNING
  INFORMATION", the mmCIF `_pdbx_reflns_twin` items, else the last `twin_fraction=` line of its
  log). Twin laws in `advanced.refine.extra_args` are refused: set `phases.twin_law`, so that
  valdo also handles the indexing check and the twin fraction.
- **Which indexing variant is kept:** with twin refinement, both indexings of a dataset fit
  about equally well (the twin fraction becomes 1 - f in the other one), so R-free cannot
  choose between them. When a dataset has two variants, the filter keeps the one with the lower
  twin fraction (lower by at least 0.1), otherwise the one that correlates best with the
  reference.
- **What it does not do:** only the phases and refined models come from twin refinement. The
  amplitudes the VAE learns from are the measured, twinned ones (valdo does no detwinning and
  no intensity conversion), so strongly twinned datasets still correlate less well with the
  reference (and may be dropped by filter), and a ligand bound in one domain looks weaker.

Without `phases.twin_law` (the default), refinement and every result are exactly as before.
Twinned datasets refined that way can fit valdo's check of the refined MTZ's indexing under
two operators (their amplitudes look alike under the twin law). With dimple, which can
re-index the data, such a dataset is dropped at refine with "could not relate the refined MTZ
to the input data ...; both indexings fit, as for twinned data" (seen at a twin fraction of
0.25). phenix.refine never re-indexes, so its result is kept in the input indexing and its
`reason` in `04_refine/refine_summary.csv` carries the same hint (seen on one of 39 PTP1B
datasets, CC 0.921 under -h,-k,l, with PHENIX 1.21.2 and 2.2.1; and on synthetic data with twin
fractions above 0.27, where R-free without the twin law rose to 0.28). `valdo run` counts these
datasets in either case. Twin refinement avoids this, because the indexing is then taken from
the program itself.

## Batch jobs on a cluster

On a cluster, `valdo run` belongs in a batch job, not on a login node (where `valdo check`
warns: `login_node`). `valdo jobscript -o run_valdo.sh` writes the script, for SGE or SLURM
(detected automatically); submit it with `qsub run_valdo.sh` or `sbatch run_valdo.sh`.
`valdo jobscript` takes `--slots` (cores; default `compute.ncpu`), `--hours`, `--mem-gb` (per
core; default 4), `--gpu` (see [GPUs](#gpus)) and `--env-setup 'shell line'` (by default the
script activates the environment you run `valdo jobscript` in; lines without `activate` are
added after that, lines with `activate` replace it; PHENIX and CCP4 need no line when
`programs.phenix`/`programs.ccp4` is set in `valdo.yaml`). Without `--hours` it sets the time
limit from the number of data files: about 3 times the time measured on Wynton plus 15 minutes
(e.g. 1 h for 300 datasets on 16 cores with phases in the data, 2 h with dimple refinement); it
prints the value and why. Look at the script before you submit it; [WYNTON.md](WYNTON.md) has
a worked Wynton example.

## Run times

| step | ~300 datasets | ~1,700 datasets |
|---|---|---|
| `valdo init` | 2.5 s (measured, 300 PTP1B files) | 8 s (measured, 1,680 PTP1B files) |
| `valdo check` | 2 s (measured) | a few seconds (headers only) |
| `valdo run`, 16 CPU cores, no GPU, phases in the data | 5 min (measured: 298 PTP1B datasets) | 51 min (measured: 1,677 PTP1B datasets) |
| `valdo run` with `phases.source: refine` (dimple) | add about 1 minute per dataset per core (measured: 39 PTP1B datasets refined in 3 min on 16 cores, 44-65 s each), e.g. ~20 min for 300 datasets on 16 cores | add ~2 h on 16 cores |
| `valdo run` with `phases.source: refine` (phenix) | add about 6-11 minutes per dataset per core (measured: 39 PTP1B datasets refined in 20 min on 16 cores with PHENIX 2.2.1, 33 min with 1.21.2), e.g. ~2-3.5 h for 300 datasets on 16 cores | add ~11-20 h on 16 cores |

How these numbers were obtained: the full PTP1B screen (1,680 files, 1,677 usable, 500
training epochs) was run with `valdo run` as an SGE job on one 16-core Wynton CPU node on
2026-10-24: 50 min 46 s in total. Per stage: standardize 22 s, reindex 1.7 min, scale 9.7 min,
filter 5 s, preprocess 4.6 min, train 22 min, reconstruct 2 s, rescale 1.3 min,
add_phases_and_blobs 5.3 min, tag_blobs 5.0 min, report 14 s. A random 300-file subset of the
same screen (298 usable) took 5 min 9 s on 16 cores, following [QUICKSTART.md](QUICKSTART.md)
step by step: standardize 9 s, reindex 16 s, scale 45 s, filter 1 s, preprocess 42 s, train
2.0 min, reconstruct 0.4 s, rescale 10 s, add_phases_and_blobs 37 s, tag_blobs 22 s, report
6 s. Most stages use all `compute.ncpu` cores; training uses them through PyTorch's threads.
Your times depend on the node, resolution and file sizes.

## Reading the report

The last stage of `valdo run` writes the report; `valdo report` rebuilds it (for example
`valdo report --top 50` for maps of more hits). Besides the hits, `report.html` has a summary,
the dataset funnel (how many datasets reached each stage, and how many were dropped where),
quality-control plots (reindexing, scaling, VAE training) and, if you gave known bound models,
how many of their ligands VALDO found and at which rank (descriptive only).

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

1. Open it in Coot or PyMOL (in `<workdir>/report`: `coot --script view_top_hits.py` or
   `pymol view_top_hits.pml`). Each hit opens with its model, the VALDO difference map
   (contoured at +3 sigma, green, and -3 sigma, red) and, when available, a 2mFo-DFc map,
   centred on the blob.
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
  inconsistently and are often dropped by filter. With `phases.source: refine`, valdo can
  refine them as twinned (`phases.twin_law`, see [Twinned crystals](#twinned-crystals)), which
  improves their phases; the amplitudes the VAE compares stay twinned, so check strongly
  twinned datasets separately.
- **One crystal form per project.** Datasets in another space group or with a very different
  cell are excluded or dropped; run them as a separate project.
- **Focal residue.** Blobs near `focal_residue.seqid` are removed from the hit list on
  purpose (PTP1B's reactive Cys215 changes in many datasets). Set it to `null` to see them.
- **Known-ligand validation is descriptive.** The section on known bound models reports how
  VALDO ranked those ligands; it is not a calibrated error rate.

## What success looks like

- `valdo run` ends with `Report: .../report/report.html` and exit code 0.
- Most datasets went through (`valdo status`: `Datasets: N total, M ok`). Losing a few percent
  at `filter` (poor correlation with the reference) is normal.
- The top hits show compact positive (green) density away from the focal residue. On the
  synthetic toy screen all 4 planted ligands were the top blob of their dataset. On the full
  PTP1B screen (1,626 of 1,680 datasets analysed; 3 excluded as another crystal form, 51
  dropped by the filter), 106 of the 110 datasets with a known bound model had a blob on the
  ligand, and for 103 of them it was the top blob of the dataset (descriptive check,
  `--bound-models`).
- Then judge each hit by eye (see [Reading the report](#reading-the-report)).

## Where the results are

Everything is in the work folder (`workdir` in `valdo.yaml`, default `valdo_run` next to
`valdo.yaml`):

| path | what |
|---|---|
| `report/report.html` | start here |
| `report/blobs.csv`, `report/top_hits.csv` | all kept blobs, ranked; the top hits with residues and map values |
| `report/datasets.csv` | one row per dataset: original name, status, best blob |
| `report/hits/`, `report/view_top_hits.py`, `report/view_top_hits.pml` | maps and models of the top hits; viewer scripts |
| `datasets.csv` | what happened to every dataset: ID, original name, kept or dropped (where and why), flags |
| `logs/<stage>.log`, `logs/run.log` | full output of each stage; the console output of every `valdo run` |
| `configs/<stage>.yaml` | the exact upstream settings used (rerun any stage with `valdo.pipeline <stage> configs/<stage>.yaml --force`) |
| `01_standardize/` ... `11_tag/` | intermediate files of each stage |

## Resuming and rerunning

- `valdo status` (any time, also while a job is running) shows what has run, what is out of
  date and where datasets were dropped. A stage marked `out of date` would run again on the
  next `valdo run` (you changed a setting or an input).
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
