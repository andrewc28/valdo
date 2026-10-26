# valdo-easy

**New here? Follow [docs/easy/QUICKSTART.md](docs/easy/QUICKSTART.md).**

valdo-easy runs [VALDO](https://github.com/Hekstra-Lab/valdo) (VAE-assisted ligand discovery,
Hekstra Lab) on a crystallographic fragment screen with one command, `valdo`, and one commented
project file, and ends with a report of ranked candidate ligand sites. The method is upstream
VALDO 0.0.6 (commit `1ec32c6`); every change is listed in
[CHANGES_FROM_UPSTREAM.md](docs/easy/CHANGES_FROM_UPSTREAM.md).

## Install

```bash
git clone -b valdo-easy https://github.com/andrewc28/valdo.git valdo
cd valdo
conda env create -f environment.yml   # creates the environment 'valdo'
conda activate valdo
pip install -e .
valdo --version
```

On UCSF Wynton, install as described in [docs/easy/WYNTON.md](docs/easy/WYNTON.md).

## Run

```bash
mkdir my_screen && cd my_screen
valdo init /path/to/mtz_folder --apo /path/to/apo.pdb   # writes valdo.yaml
valdo check                                             # fix every ERROR it lists
valdo run                                               # resumable: after any stop, run it again
valdo status                                            # progress, at any time
```

On a cluster, do not use `valdo run` on the login node; submit it as a batch job instead:
`valdo jobscript -o run_valdo.sh`, then `qsub run_valdo.sh` (SLURM: `sbatch run_valdo.sh`).
The ranked hits are in `valdo_run/report/report.html`.

## Documentation

[Quickstart](docs/easy/QUICKSTART.md) · [user guide](docs/easy/USER_GUIDE.md) ·
[troubleshooting](docs/easy/TROUBLESHOOTING.md) · [settings](docs/easy/CONFIG_REFERENCE.md) ·
[UCSF Wynton](docs/easy/WYNTON.md) · for AI agents: [AGENTS.md](AGENTS.md).
Upstream VALDO's own README (Python API, notebooks, the per-stage `valdo.pipeline` command):
[UPSTREAM_README.md](UPSTREAM_README.md).
