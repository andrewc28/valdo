# VALDO quickstart

**For:** crystallographers running VALDO on a fragment screen for the first time. No
programming needed.
**You need first:** a Linux or macOS computer with conda, or a cluster account (UCSF Wynton:
see [WYNTON.md](WYNTON.md)); the screen's MTZ files; a model (PDB file) of the unbound
protein, or one model per dataset.

VALDO learns from the whole screen what an unbound crystal looks like and ranks the places
where a dataset differs from it ("blobs"): candidate ligand sites. Terms are explained in
[USER_GUIDE.md](USER_GUIDE.md#words-you-will-meet).

## 1. Install (once)

```bash
git clone -b valdo-easy https://github.com/andrewc28/valdo.git valdo
cd valdo
conda env create -f environment.yml   # creates the environment 'valdo' (a few minutes)
conda activate valdo                  # again in every new terminal
pip install -e .                      # installs the 'valdo' command
valdo --version                       # valdo 0.0.6+easy.1 (upstream VALDO 0.0.6, commit 1ec32c6)
```

On UCSF Wynton, build the environment as described in [WYNTON.md](WYNTON.md) instead.

## 2. Set up: `valdo init`

```bash
mkdir my_screen && cd my_screen       # one folder per screen; run every valdo command in it
valdo init /path/to/mtz_folder --apo /path/to/apo.pdb
```

`valdo init` reads the MTZ headers (seconds), writes `valdo.yaml` (every setting, commented)
and says what it decided and why. Add the options that apply:

- `--models '/path/to/models/{name}.pdb'`: one model per dataset (`{name}` is the dataset's
  name); `--apo` is then the fallback. A single `--apo` model must be in the indexing of the
  reference (first) dataset: use that dataset's own model.
- **Phases.** Nothing to add if the MTZ files contain map coefficients (XChem/PanDDA inputs
  do); then give each dataset the model its phases came from (`--models`). Data only:
  `--phases refine --program dimple --ccp4 /path/to/ccp4-9`, or with your own PHENIX licence
  `--phases refine --program phenix --phenix /path/to/phenix`. Every dataset already refined:
  `--phases existing --existing '/path/to/refined/{name}.mtz'`.
- `--focal-seqid N`: no hits around a residue that changes in many datasets (PTP1B: Cys215).
- `--bound-models FOLDER` (optional): known ligand-bound models; the report then says how many
  of their ligands VALDO found.

More options and details: [USER_GUIDE.md](USER_GUIDE.md#setting-up-a-project).

## 3. Check: `valdo check`

Fix every `ERROR` (it stops `valdo run`), then check again; read the warnings.

## 4. Run

On a workstation: `valdo run`. On a cluster, submit a batch job instead of running on the
login node:

```bash
valdo jobscript -o run_valdo.sh       # SGE or SLURM, detected automatically
qsub run_valdo.sh                     # SLURM: sbatch run_valdo.sh
```

On 16 cores this took 5 minutes for 300 PTP1B datasets and 51 minutes for 1,680 with phases
in the MTZ files; refinement adds about 1 minute (dimple) or 6-11 minutes (PHENIX) per dataset
per core. `valdo status` shows progress at any time. If a run stops (time limit, Ctrl-C,
crash), start it again: finished stages are kept.

## 5. Results

Open `valdo_run/report/report.html` in a web browser: how many datasets got through each stage,
and the top hits, ranked. To look at the hits in Coot or PyMOL:

```bash
cd valdo_run/report
coot --script view_top_hits.py        # or: pymol view_top_hits.pml
```

A blob means "something changed here", not "a ligand": judge every hit by eye
([USER_GUIDE.md](USER_GUIDE.md#reading-the-report)).

**Next:** [USER_GUIDE.md](USER_GUIDE.md) (stages, choices, run times, output files),
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) (every message),
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) (every setting).
