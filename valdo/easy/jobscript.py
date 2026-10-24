"""Batch job scripts that run ``valdo run`` on a cluster (SGE or SLURM).

:func:`render` returns the text of a job script; ``valdo jobscript`` prints or
writes it. The script

* asks the scheduler for ``slots`` CPU cores, ``mem_gb`` GB of memory per
  core, ``hours`` of wall-clock time and, optionally, one GPU;
* activates the user's Python environment (the ``env_setup`` lines; when none
  are given, a commented placeholder that must be edited, and a check that
  stops the job with a clear message if ``valdo`` is not on PATH);
* passes the core count to VALDO (``VALDO_NCPU``, also used for the
  OpenMP/BLAS thread counts) and hides GPUs on CPU jobs so the run is
  deterministic;
* runs ``valdo run -c <config>`` and exits with its exit code. ``valdo run``
  is resumable: if the job hits its time limit, submit the same script again.

Scheduler notes
---------------
SGE (UCSF Wynton and similar): ``-pe smp N`` for N cores, ``-l mem_free`` is
per core, ``-l h_rt`` is the run-time limit, ``-q gpu.q`` selects Wynton's GPU
queue and the GPU index is in ``$SGE_GPU``. Verified on Wynton: smp, mem_free,
h_rt, -j/-o, NSLOTS. Not verified here: the GPU lines (Wynton documents
``-q gpu.q`` and ``CUDA_VISIBLE_DEVICES=$SGE_GPU``; other SGE sites differ).
SLURM: ``--cpus-per-task``, ``--mem-per-cpu``, ``--time``, ``--gres=gpu:1``;
written from the SLURM documentation, not tested on a SLURM cluster (sites
often also need ``--partition``/``--account``; placeholders are included).
"""

import math
import os
import shlex

SCHEDULERS = ("sge", "slurm")

_ENV_PLACEHOLDER = """\
# --- EDIT ME: make the Python environment that has valdo installed active ---
# (one of these, or whatever your site uses)
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate valdo
# source /path/to/venv/bin/activate
# module load ...   # e.g. CCP4/PHENIX, if the refine stage runs here"""


def render(scheduler, config_path, slots=16, hours=24, mem_gb=4, gpu=False, env_setup=None, log_dir=None,
           job_name="valdo"):
    """Return the text of a batch script that runs ``valdo run -c <config_path>``.

    Parameters
    ----------
    scheduler : {"sge", "slurm"}
        ("wynton" is accepted as an alias for "sge").
    config_path : str
        The project file (valdo.yaml); written into the script as an absolute path.
    slots : int
        CPU cores to request; passed to VALDO as ``VALDO_NCPU``.
    hours : float
        Wall-clock limit in hours.
    mem_gb : float
        Memory per core in GB.
    gpu : bool
        Request one GPU (SGE: Wynton's ``gpu.q``; SLURM: ``--gres=gpu:1``).
    env_setup : str or list of str, optional
        Shell lines that activate the environment (e.g.
        ``"source ~/miniconda3/etc/profile.d/conda.sh && conda activate valdo"``).
    log_dir : str, optional
        Folder for the scheduler's log file (default: ``<config folder>/joblogs``).
        It must exist before the job starts; ``valdo jobscript`` creates it.
    job_name : str
        Scheduler job name.
    """
    sched = str(scheduler).strip().lower()
    if sched == "wynton":
        sched = "sge"
    if sched not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {', '.join(SCHEDULERS)} (got {scheduler!r})")
    slots = int(slots)
    if slots < 1:
        raise ValueError("slots must be at least 1")
    if not (hours and float(hours) > 0):
        raise ValueError("hours must be > 0")
    if not (mem_gb and float(mem_gb) > 0):
        raise ValueError("mem_gb must be > 0")
    config = os.path.abspath(os.path.expanduser(str(config_path)))
    log_dir = os.path.abspath(os.path.expanduser(log_dir)) if log_dir else os.path.join(
        os.path.dirname(config), "joblogs")
    name = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in str(job_name)) or "valdo"
    if name[0].isdigit():
        name = "v" + name
    walltime = _walltime(hours)
    mem = _mem(mem_gb)

    if sched == "sge":
        head = [
            "#!/bin/bash",
            "#$ -S /bin/bash",
            f"#$ -N {name}",
            f"#$ -wd {_q(os.path.dirname(config))}",
            "#$ -j y",
            f"#$ -o {_q(log_dir + '/')}",
            f"#$ -pe smp {slots}",
            f"#$ -l mem_free={mem}",
            f"#$ -l h_rt={walltime}",
        ]
        if gpu:
            head += ["#$ -q gpu.q",
                     "# (Wynton GPU queue; on other SGE clusters ask your admins how to request a GPU)"]
        ncpu_line = 'export VALDO_NCPU="${NSLOTS:-%d}"' % slots
        gpu_lines = (['# Wynton: SGE_GPU holds the GPU index assigned to this job (not verified by valdo-easy)',
                      'export CUDA_VISIBLE_DEVICES="${SGE_GPU:-}"'] if gpu else
                     ['export CUDA_VISIBLE_DEVICES=""   # CPU job: hide GPUs (deterministic results)'])
        submit = f"qsub {_q('<this file>')}"
    else:
        head = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --chdir={_q(os.path.dirname(config))}",
            f"#SBATCH --output={_q(os.path.join(log_dir, name + '-%j.log'))}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={slots}",
            f"#SBATCH --mem-per-cpu={mem}",
            f"#SBATCH --time={walltime}",
            "##SBATCH --partition=<partition>   # remove one '#' and edit if your site needs it",
            "##SBATCH --account=<account>",
        ]
        if gpu:
            head += ["#SBATCH --gres=gpu:1"]
        ncpu_line = 'export VALDO_NCPU="${SLURM_CPUS_PER_TASK:-%d}"' % slots
        gpu_lines = ([] if gpu else
                     ['export CUDA_VISIBLE_DEVICES=""   # CPU job: hide GPUs (deterministic results)'])
        submit = f"sbatch {_q('<this file>')}"

    if env_setup is None:
        env_block = _ENV_PLACEHOLDER
    else:
        lines = env_setup.splitlines() if isinstance(env_setup, str) else [str(x) for x in env_setup]
        env_block = "\n".join(["# --- environment ---"] + lines)

    mkdir_note = "" if os.path.isdir(log_dir) else f"# The log directory must exist first:  mkdir -p {_q(log_dir)}\n"
    body = f"""
# valdo-easy job script ({sched.upper()}), generated by 'valdo jobscript'.
# Submit with:  {submit}
{mkdir_note}# 'valdo run' is resumable: if this job runs out of time, submit it again and it
# continues from the first unfinished stage (if it was killed less than 5 minutes ago, the new
# job may stop with 'locked' because the old lock has not expired yet: submit it again).

{env_block}
# Use only the packages of the environment above, not ones installed with 'pip install --user'
export PYTHONNOUSERSITE=1

set -o pipefail
if ! command -v valdo >/dev/null 2>&1; then
  echo "ERROR: 'valdo' is not on PATH. Edit the environment section of this script so that" >&2
  echo "       the Python environment with valdo installed is active." >&2
  exit 1
fi

# VALDO_NCPU (the cores granted to this job) overrides compute.ncpu in valdo.yaml.
{ncpu_line}
export OMP_NUM_THREADS="$VALDO_NCPU" MKL_NUM_THREADS="$VALDO_NCPU" OPENBLAS_NUM_THREADS="$VALDO_NCPU"
export MPLBACKEND=Agg
{chr(10).join(gpu_lines) if gpu_lines else '# GPU: the scheduler sets CUDA_VISIBLE_DEVICES'}

CONFIG={_q(config)}
echo "[valdo-job] host=$(hostname) start=$(date '+%Y-%m-%d %H:%M:%S') cores=$VALDO_NCPU config=$CONFIG"
valdo --version
valdo run -c "$CONFIG"
rc=$?
echo "[valdo-job] finished $(date '+%Y-%m-%d %H:%M:%S') with exit code $rc"
exit $rc
"""
    return "\n".join(head) + "\n" + body


def _walltime(hours):
    total = int(math.ceil(float(hours) * 3600))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _mem(mem_gb):
    v = float(mem_gb)
    return f"{int(v)}G" if v == int(v) else f"{int(math.ceil(v * 1024))}M"


def _q(text):
    return shlex.quote(str(text))
