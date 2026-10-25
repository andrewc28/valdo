# VALDO quickstart

**For:** crystallographers who want to run VALDO on a fragment screen. No programming needed.
**You need first:** a Linux or macOS computer with conda (or a cluster account, e.g. UCSF
Wynton: see [WYNTON.md](WYNTON.md)); your screen's MTZ files; a model (PDB file) of the
unbound protein, or one model per dataset.

VALDO looks at every dataset of a screen at once and points to the places where a dataset
differs from what an unbound crystal of the same protein looks like. Those places are
candidate ligand sites. This page takes you from your files to a ranked list of hits in five
commands: `valdo init`, `valdo check`, `valdo run`, `valdo status`, `valdo report`.

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

## 1. Install (once)

```bash
git clone -b valdo-easy <URL of the lab's valdo repository> valdo
cd valdo
conda env create -f environment.yml      # creates the environment 'valdo' (a few minutes)
conda activate valdo
pip install -e .                         # installs the 'valdo' command
valdo --version
```

Expected output:

```
valdo 0.0.6+easy.1 (upstream VALDO 0.0.6, commit 1ec32c6)
```

On UCSF Wynton, follow [WYNTON.md](WYNTON.md) instead (the environment needs to be built in a
particular way there).

## 2. Describe your project: `valdo init`

Give `valdo init` the folder with your MTZ files and your model(s):

```bash
mkdir my_screen && cd my_screen
valdo init /path/to/mtz_folder --apo /path/to/apo.pdb
```

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

`valdo init` only reads the file headers, so it is fast. It writes `valdo.yaml`, a
commented text file with every setting, and tells you what it decided and why. Real output on
the PTP1B screen (1,680 datasets; Wynton login node; 8 seconds; `$X` is the folder holding the
PTP1B files):

```
$ valdo init $X/mtzs/mtzs --models $X/pandda_input_models/pandda_input_models \
      --apo $X/pandda_input_models/pandda_input_models/PTP1B-y0001-pandda-input.pdb \
      --focal-seqid 215 --bound-models $X/bound_state_models/bound_state_models --ncpu 16
Reading 1680 MTZ headers ...
Read 1680 MTZ headers in 3.5 s: 1677 usable, P 31 2 1, cell 89.98 89.98 106.63 90.00 90.00 120.00
  columns        F-obs, SIGF-obs  (observed amplitudes with sigmas in 1680/1680 files)
  phases.source  input_mtz  (map coefficients PHI2FOFCWT/PHIFOFCWT (xchem names) in 1680/1680 data files)
  focal_residue  215 (CYS)  (given with --focal-seqid; blobs within the radius are not reported as hits)
  reference      PTP1B-y0001 (0001)  (first dataset (by ID) that passed the input checks (upstream convention))
  reindex        auto -> on  (P 31 2 1 with this cell allows 1 alternative indexing operator(s) (-h,-k,l); datasets may be indexed inconsistently, so they are reindexed to the reference)
  compute.ncpu   16  (given with --ncpu; use your batch job's slot count)
  warning: 3 file(s) cannot be used ('valdo check' lists them)
Wrote /wynton/scratch/achoe/valdo-easy/main/ptp1b_full/valdo.yaml
Next: 'valdo check' (explains any problem), then 'valdo run' (or 'valdo jobscript -o run_valdo.sh' to run it as a batch job).
```

The most important decision is **where the phases come from** (`phases.source`):

| `phases.source` | when | what you need |
|---|---|---|
| `input_mtz` | your MTZ files already contain map coefficients (typical for XChem / PanDDA inputs) | nothing more |
| `refine` | your MTZ files hold only data | `--phases refine --program dimple --ccp4 /path/to/ccp4-9` (CCP4) or `--program phenix --phenix /path/to/phenix-2.2-6143` (your own PHENIX licence) |
| `existing` | you refined every dataset yourself | `--phases existing --existing '/path/refined/{name}.mtz'` |

With `refine`, the installation folder ends up in `valdo.yaml` (`programs.ccp4` or
`programs.phenix`; init fills it in from PATH when you leave out `--ccp4`/`--phenix`). That one
line is all valdo needs: every command and batch job sets the program up from it (see
[USER_GUIDE.md](USER_GUIDE.md#refinement-programs)). For twinned crystals add
`--twin-law -h,-k,l`, or `--twin-law auto` when the lattice allows only one twin law
(`valdo check` lists the possible laws; see [USER_GUIDE.md](USER_GUIDE.md#twinned-crystals)).

Open `valdo.yaml` in a text editor and read part 1 ("your project") and part 2 ("detected by
valdo init"). Part 3 ("advanced") holds the settings validated on the PTP1B screen; leave it
alone unless you have a reason. Every setting is described in
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

## 3. Check everything: `valdo check`

```bash
valdo check
```

`valdo check` looks at your computer (packages, CPUs, disk space), your files (readable,
columns, space group, cell, resolution, a model and phases for every dataset) and your
settings. Every message says what is wrong, why it matters and how to fix it. **Errors**
stop `valdo run`; **warnings** do not. Excerpt from the PTP1B screen:

```
WARN  3 file(s) cannot be used: space group P 32 2 1 differs from the other datasets
      3 of 1680 files will be skipped (space group P 32 2 1 differs from the other datasets).
      Why: They are listed in datasets.csv with dropped_stage 'input'.
      Fix: Datasets in another crystal form cannot share one VAE: remove them, or run them as a separate project.
        - PTP1B-y0162 (0139)
        - PTP1B-y0422 (0365)
        - PTP1B-y1447 (1242)
...
valdo check: 0 error(s), 4 warning(s); 1677 of 1680 datasets usable.
Ready: 'valdo jobscript -o run_valdo.sh' or 'valdo run'.
```

`valdo check --deep` also reads all reflection data (slower; finds missing values and
duplicated reflections). A misspelled setting is caught here too:

```
ERROR: /wynton/scratch/achoe/valdo-easy/main/toy_console_5393479/typo.yaml has 1 problem(s).
  - advanced.train.epoch: unknown setting. Did you mean 'advanced.train.epochs'?
```

## 4. Run: `valdo run`

On your own workstation, or inside an interactive cluster session:

```bash
valdo run
```

On a cluster, write a batch script and submit it (do not run long jobs on a login node):

```bash
valdo jobscript -o run_valdo.sh          # SGE (qsub) or SLURM (sbatch), detected automatically
qsub run_valdo.sh                         # on SLURM clusters: sbatch run_valdo.sh
```

`valdo jobscript` takes `--slots` (cores), `--hours`, `--mem-gb` (per core) and
`--env-setup 'shell line'` (by default the script activates the environment you run
`valdo jobscript` in; extra lines are added after it; PHENIX and CCP4 need no line when
`programs.phenix`/`programs.ccp4` is set in `valdo.yaml`). Without
`--hours` it sets the time limit from the number of data files: about 3 times the time measured
on Wynton plus 15 minutes (e.g. 1 h for 300 datasets on 16 cores with phases in the data,
2 h with dimple refinement); it prints the value and why. Look at the script before you submit
it; see [WYNTON.md](WYNTON.md) for a worked Wynton example.

`valdo run` prints one line per stage. Real output on the synthetic toy screen used by the tests
(24 small datasets plus one damaged file, 4 CPUs, 1 minute):

```
valdo run: .../toy_console_5393479/valdo.yaml -> .../toy_console_5393479/valdo_run  (started 2026-10-24T01:43:02, 4 CPU(s))
11 of 12 stages apply to this project (not: refine, phases.source is input_mtz)
[1/11] standardize ... done in 10 s: 24/24 datasets
[2/11] reindex ... done in 11 s: 24/24 datasets
      note: 1 dataset(s) have ambiguous indexing (CCs within cc_min_dif); the best one is used after filtering
[3/11] scale ... done in 12 s: 24/24 datasets
[4/11] filter ... done in 0.1 s: 23/24 datasets; 1 dropped, see datasets.csv
[5/11] preprocess ... done in 1 s: 23/23 datasets
[6/11] train ... done in 17 s: 23/23 datasets
[7/11] reconstruct ... done in 0.1 s: 23/23 datasets
[8/11] rescale ... done in 0.3 s: 23/23 datasets
[9/11] add_phases_and_blobs ... done in 2 s: 23/23 datasets
[10/11] tag_blobs ... done in 0.7 s: 23/23 datasets
[11/11] report ... done in 1 s: 23/23 datasets

Datasets: 23 of 25 went through; 2 dropped (reasons in .../valdo_run/datasets.csv).
Report: .../toy_console_5393479/valdo_run/report/report.html
```

(Each stage also prints a "running" line with the path of its log; they are left out here. The
"stages apply" line was added after this run was recorded.)

**It is safe to stop and restart.** `valdo run` remembers which stages are finished. If the
job is killed (time limit, Ctrl-C, node crash), run the same command again: it continues
with the first unfinished stage. If you change a setting, only the stages it affects (and the
ones after them) run again. `valdo run --dry-run` shows what would run, without running it.

### How long does it take?

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
same screen (298 usable) took 5 min 9 s on 16 cores, following this page step by step: standardize
9 s, reindex 16 s, scale 45 s, filter 1 s, preprocess 42 s, train 2.0 min, reconstruct 0.4 s,
rescale 10 s, add_phases_and_blobs 37 s, tag_blobs 22 s, report 6 s. Most stages
use all `compute.ncpu` cores; training uses them through PyTorch's threads. Your times depend
on the node, resolution and file sizes.

## 5. Follow progress: `valdo status`

```bash
valdo status
```

Real output after the toy run:

```
valdo status: .../toy_console_5393479/valdo.yaml -> .../toy_console_5393479/valdo_run
   #  stage                 status           time  datasets
   1  standardize           done             10 s  24 -> 24
   2  reindex               done             11 s  24 -> 24
   3  scale                 done             12 s  24 -> 24
   4  refine                n/a
   5  filter                done            0.1 s  24 -> 23 (1 dropped)
   6  preprocess            done              1 s  23 -> 23
   7  train                 done             17 s  23 -> 23
   8  reconstruct           done            0.1 s  23 -> 23
   9  rescale               done            0.3 s  23 -> 23
  10  add_phases_and_blobs  done              2 s  23 -> 23
  11  tag_blobs             done            0.7 s  23 -> 23
  12  report                done              1 s  23 -> 23
Datasets: 25 total, 23 ok, 2 dropped (filter 1, input 1)
Last run: started 2026-10-24T01:43:02, finished 2026-10-24T01:43:59, ok
Next: open .../toy_console_5393479/valdo_run/report/report.html
```

`valdo status` can be run at any time, also while a job is running. A stage marked
`out of date` would run again on the next `valdo run` (you changed a setting or an input).

## 6. Look at the results: `valdo report`

The last stage writes the report; `valdo report` rebuilds it (for example
`valdo report --top 50` for maps of more hits). Open `<workdir>/report/report.html` in a web
browser. It contains:

- a summary and a **dataset funnel** (how many datasets reached each stage, and how many were
  dropped where);
- the **top hits**: blobs ranked by score, with the dataset's original name, the nearest
  residue and the difference density at the blob;
- quality-control plots (reindexing, scaling, VAE training);
- if you gave known bound models: how many of their ligands VALDO found, and at which rank
  (descriptive only).

To look at the hits in Coot or PyMOL:

```bash
cd <workdir>/report
coot --script view_top_hits.py           # or: pymol view_top_hits.pml
```

Each hit opens with its model, the VALDO difference map (green positive, red negative) and,
when available, a 2mFo-DFc map, centred on the blob.

### What success looks like

- `valdo run` ends with `Report: .../report/report.html` and exit code 0.
- Most datasets went through (`valdo status`: `Datasets: N total, M ok`). Losing a few percent
  at `filter` (poor correlation with the reference) is normal.
- The top hits show compact positive (green) density away from the focal residue. On the
  synthetic toy screen all 4 planted ligands were the top blob of their dataset. On the full
  PTP1B screen (1,626 of 1,680 datasets analysed; 3 excluded as another crystal form, 51
  dropped by the filter), 106 of the 110 datasets with a known bound model had a blob on the
  ligand, and for 103 of them it was the top blob of the dataset (descriptive check,
  `--bound-models`).
- Then judge each hit by eye (see "Reading the report" in [USER_GUIDE.md](USER_GUIDE.md)).

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
| `logs/<stage>.log` | full output of each stage |
| `configs/<stage>.yaml` | the exact upstream settings used (rerun any stage with `valdo.pipeline <stage> configs/<stage>.yaml --force`) |
| `01_standardize/` ... `11_tag/` | intermediate files of each stage |

Datasets are renamed to numbers (`0001`, `0002`, ...) inside the pipeline; `datasets.csv` and
the report always show the original names next to them.

## Next

- [USER_GUIDE.md](USER_GUIDE.md): what each stage does, how to judge hits, limitations.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): every error message and what to do.
- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md): every setting in `valdo.yaml`.
