# Running VALDO on UCSF Wynton

**For:** lab members running VALDO on the UCSF Wynton HPC cluster (SGE batch system).
**You need first:** a Wynton account and an SSH login; the general workflow from
[QUICKSTART.md](QUICKSTART.md).

Everything on this page was used on Wynton on 2026-10-24 (Rocky Linux 8.10). Paths under
`/wynton/home/jo/achoe/...` or `/wynton/scratch/achoe/...` are from those test runs; use your
own home and scratch folders.

## Where to run what

| command | where |
|---|---|
| `valdo init`, `valdo check`, `valdo status`, `valdo report`, `valdo jobscript` | login node is fine (seconds; `valdo init` read 1,680 PTP1B headers in 8 s) |
| `valdo run` | always as a batch job (`qsub`). Login nodes are shared and long processes on them are killed. `valdo check` warns (`login_node`) when it runs on one. |

## 1. Python environment

valdo needs Python 3.11 with the packages in `environment.yml` (torch 2.5.1, gemmi 0.6.7,
reciprocalspaceship 1.0.3, ...). The `module` command does not work on login nodes, so use
your own conda installation:

```bash
# once: Miniforge in your home folder (if you have no conda yet)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh

# once: the VALDO environment and the valdo command
cd ~/valdo                         # your clone of the valdo-easy branch
conda env create -f environment.yml
conda activate valdo
pip install -e .
valdo --version                    # valdo 0.0.6+easy.1 (upstream VALDO 0.0.6, commit 1ec32c6)
```

The environment takes a few GB of your home quota (torch is large).

The validated test runs used a Python 3.11 environment with exactly these package versions
and an editable install of valdo in a virtual environment on top of it
(`python -m venv --system-site-packages ...; pip install --no-deps -e .`). Either way works;
what matters is that `python -c "import valdo; print(valdo.__file__)"` points to the copy you
expect.

## 2. Data and storage

- **Home** (`/wynton/home/...`) has a hard quota (500 GiB). Keep code, environments, the
  project file and the final `report/` there.
- **Scratch** (`/wynton/scratch/$USER`, create it with `mkdir -p /wynton/scratch/$USER`) is
  large but **files older than 14 days are deleted**. Put the work folder there: set
  `workdir: /wynton/scratch/<you>/my_screen/valdo_run` in `valdo.yaml` (or
  `valdo init ... --workdir ...`).
- A run needs about 12 MB per dataset (measured on PTP1B with `du --apparent-size`: 3.7 GB
  for 300 datasets, 20 GB for the full 1,680-dataset screen).
- After the run, copy what you want to keep to home:
  `cp -a /wynton/scratch/<you>/my_screen/valdo_run/report ~/my_screen_report`.
  If scratch was purged while a project was unfinished, the next stage that needs a deleted
  file stops with the `file_missing` explanation; start again with
  `valdo run --from standardize`.
- To look at the results, copy the `report` folder to your own computer (for example
  `scp -r <you>@log1.wynton.ucsf.edu:/wynton/scratch/<you>/my_screen/valdo_run/report .`) and open
  `report.html` in a browser; `view_top_hits.py` (Coot) and `view_top_hits.pml` (PyMOL) open the
  maps in `report/hits/` from there.
- Wynton's `du` shows compressed sizes; use `du -sh --apparent-size` to see real sizes.

## 3. The SGE job script

`valdo jobscript` writes a script that runs `valdo run` with the slots it was given. By
default the script activates the environment you run `valdo jobscript` in (a venv or a conda
environment): look at its `--- environment ---` block. To choose the lines yourself, give them
with `--env-setup`:

```bash
cd /wynton/scratch/<you>/my_screen
valdo jobscript --slots 16 --hours 12 --mem-gb 4 \
    --env-setup "source ~/miniforge3/etc/profile.d/conda.sh" \
    --env-setup "conda activate valdo" \
    -o run_valdo.sh
qsub run_valdo.sh
```

This is the script written for the full PTP1B test run (with the test environment's
activation lines):

```bash
#!/bin/bash
#$ -S /bin/bash
#$ -N valdo
#$ -wd /wynton/scratch/achoe/valdo-easy/main/ptp1b_full
#$ -j y
#$ -o /wynton/scratch/achoe/valdo-easy/main/ptp1b_full/valdo_run/logs/
#$ -pe smp 16
#$ -l mem_free=4G
#$ -l h_rt=12:00:00

# --- environment ---
source ~/valdo-twin/env.sh
source ~/valdo-easy/venv-main/bin/activate

set -o pipefail
if ! command -v valdo >/dev/null 2>&1; then
  ...
fi

# VALDO_NCPU (the cores granted to this job) overrides compute.ncpu in valdo.yaml.
export VALDO_NCPU="${NSLOTS:-16}"
export OMP_NUM_THREADS="$VALDO_NCPU" MKL_NUM_THREADS="$VALDO_NCPU" OPENBLAS_NUM_THREADS="$VALDO_NCPU"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""   # CPU job: hide GPUs (deterministic results)

CONFIG=/wynton/scratch/achoe/valdo-easy/main/ptp1b_full/valdo.yaml
...
valdo run -c "$CONFIG"
```

What the lines mean:

- `-pe smp 16`: 16 cores on one node. valdo uses exactly the cores granted (`VALDO_NCPU`).
- `-l mem_free=4G`: memory **per core** (16 x 4 GB = 64 GB). This was enough for the full PTP1B
  run (its peak was not recorded; a 120-dataset run peaked at 1.8 GB).
- `-l h_rt=12:00:00`: wall-clock limit (given here with `--hours 12`). Without `--hours`,
  `valdo jobscript` estimates it from the number of data files: about 3 times the time measured
  on Wynton plus 15 minutes, rounded up to half hours (1 h for 300 datasets on 16 cores with
  phases in the data; 2 h with dimple). Jobs up to 30 minutes can use the short queue and
  start sooner; long jobs wait longer. If the limit is reached, simply `qsub` the script
  again: `valdo run` continues where it stopped. (If you resubmit within 5 minutes of a
  killed job, the new job may stop at once with `locked`, because the old job's lock has not
  expired yet: submit it again a few minutes later.)
- `-o .../logs/`: the job's own log goes to `<workdir>/logs/` (`valdo jobscript -o` creates the
  folder). Each stage's detailed log is `<workdir>/logs/<stage>.log`; the console lines of
  every run are collected in `<workdir>/logs/run.log`.

Follow the job with `qstat` (state `qw` = waiting, `r` = running), `valdo status` and
`tail -f <workdir>/logs/run.log`.

## 4. CPU or GPU

CPU nodes are enough: on 16 cores the whole PTP1B screen (1,677 datasets, 500 training
epochs) took 51 minutes, of which training took 22 minutes. A GPU only speeds up training.
To use one, `valdo jobscript --gpu` adds `-q gpu.q` and the GPU request to the script (this
variant has not been tested on Wynton yet). A model trained on a GPU must also be
reconstructed on a GPU, or retrained on the CPU (`valdo run --from train`).

## 5. Refinement on Wynton

Only needed with `phases.source: refine`. Install the program once in your home (or use your
lab's installation), then give its folder in `valdo.yaml`; that one line is enough for
`valdo check`, `valdo run` and the batch job (valdo sources the setup script itself, see
[USER_GUIDE.md](USER_GUIDE.md#refinement-programs)):

```yaml
programs:
  ccp4: ~/ccp4-9                  # for phases.program: dimple
  phenix: ~/phenix-2.2.1-6174     # for phases.program: phenix
```

(or `valdo init ... --ccp4 ~/ccp4-9`). No `--env-setup` line is needed for them, and
`valdo check` on the login node already tests the installation the job will use.

- **CCP4 / dimple**: install CCP4 (ccp4.ac.uk) in your home. The tests used CCP4 9 (dimple
  2.7): about one minute per PTP1B dataset per core. `valdo check` reports
  `refine_program_missing` if `programs.ccp4` is not set and dimple is not on PATH, and
  `program_not_found` / `program_setup_failed` if the folder is wrong.
- **PHENIX**: PHENIX is not a Wynton module, and valdo does not provide it: it needs a licence
  (free for academic users, phenix-online.org). Download the Linux installer there under your
  licence, copy it to Wynton, and install into a folder of your home (or your lab's folder)
  whose path has no spaces, for example

  ```bash
  bash Phenix-2.2.1-6174-Linux-x86_64.sh -b -p ~/phenix-2.2.1-6174
  ```

  then set `programs.phenix: ~/phenix-2.2.1-6174`. Run the installer as a batch job with a long
  time limit (e.g. `-l h_rt=12:00:00`), not on a login node: Wynton's network file systems make
  it slow (PHENIX 1.21.2 took 100 minutes into `/wynton/scratch`; PHENIX 2.2.1, about 420,000
  files, did not finish within 3 hours straight into a home folder). `/wynton/scratch` is purged
  after two weeks, so it is no place for an installation. Tested: phenix.refine 1.21.2-5419 and
  2.2.1-6174 through valdo with the packaged `.eff`: about 4 to 12 minutes per dataset per core
  in batch runs (median 6.4 minutes with 2.2.1 and 11.3 with 1.21.2, 39 PTP1B datasets refined
  16 at a time).
  Look at the first datasets' logs in `04_refine/runs/` early.
- **Without `programs.*`** you can still set the program up in the job script, as before:
  `--env-setup "source ~/ccp4-9/bin/ccp4.setup-sh"`. A line without the word `activate` is
  added after the activation of the environment you run `valdo jobscript` in; a line with
  `activate` replaces that automatic activation (then give your conda lines too, as in
  section 3). Run `valdo check` after sourcing the same line.
- **Twinned crystals**: set `phases.twin_law` to the law (or `auto`; see
  [USER_GUIDE.md](USER_GUIDE.md#twinned-crystals)).

## 6. Good practice on a shared cluster

- One `valdo run` per work folder (valdo enforces this with a lock file; `valdo status` shows
  whether a run is active, and its `--json` result then says to wait).
- Run `valdo check` and `valdo status` on a login node if you like, but `valdo run` belongs in
  a batch job: on a login node `valdo check` suggests `valdo jobscript` and `qsub` instead.
- Do not change the input files while a run is going.
- Several lab members can use the same installation; each project has its own `valdo.yaml`
  and work folder.
