"""Tests for valdo.easy.jobscript (SGE/SLURM batch scripts for `valdo run`).

The rendered scripts are checked with `bash -n` and then executed with a fake
`valdo` on PATH, so the environment and arguments the real `valdo run` would
get are verified without a scheduler.
"""

import os
import shutil
import stat
import subprocess

import pytest

from valdo.easy import jobscript

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


def _fake_valdo(bindir, rc=0):
    """A `valdo` that prints what it received and exits with `rc`."""
    os.makedirs(bindir, exist_ok=True)
    path = os.path.join(bindir, "valdo")
    with open(path, "w") as fh:
        fh.write(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "valdo 0.0.0-fake"; exit 0; fi\n'
            'echo "ARGS:$#"\n'
            'for a in "$@"; do echo "ARG:$a"; done\n'
            'echo "NCPU:$VALDO_NCPU OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS CUDA:[$CUDA_VISIBLE_DEVICES] '
            'MPL:$MPLBACKEND PWD:$(pwd)"\n'
            f"exit {int(rc)}\n"
        )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run(script_text, tmp_path, extra_env=None, path_dirs=()):
    script = tmp_path / "job.sh"
    script.write_text(script_text)
    env = {"PATH": os.pathsep.join(list(path_dirs) + ["/usr/bin", "/bin"]), "HOME": str(tmp_path)}
    env.update(extra_env or {})
    return subprocess.run([BASH, str(script)], cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=60)


def test_sge_header_and_defaults(tmp_path):
    cfg = tmp_path / "proj" / "valdo.yaml"
    text = jobscript.render("sge", str(cfg))
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#$ -S /bin/bash" in lines
    assert "#$ -N valdo" in lines
    assert f"#$ -wd {tmp_path / 'proj'}" in lines
    assert "#$ -j y" in lines
    assert f"#$ -o {tmp_path / 'proj' / 'joblogs'}/" in lines
    assert "#$ -pe smp 16" in lines
    assert "#$ -l mem_free=4G" in lines
    assert "#$ -l h_rt=24:00:00" in lines
    assert not any(ln.startswith("#$ -q") for ln in lines)  # CPU job: no queue request
    assert 'export VALDO_NCPU="${NSLOTS:-16}"' in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert f"CONFIG={cfg}" in text
    assert 'valdo run -c "$CONFIG"' in text
    assert "EDIT ME" in text  # no env_setup given: placeholder to edit
    # all scheduler directives come before the first command
    first_cmd = next(i for i, ln in enumerate(lines) if ln and not ln.startswith("#"))
    assert all(i < first_cmd for i, ln in enumerate(lines) if ln.startswith("#$"))


def test_wynton_alias_and_case():
    a = jobscript.render("sge", "/x/valdo.yaml", slots=4)
    b = jobscript.render("Wynton", "/x/valdo.yaml", slots=4)
    assert a == b


def test_sge_gpu_and_options(tmp_path):
    text = jobscript.render("sge", "/data/run/valdo.yaml", slots=8, hours=1.5, mem_gb=2.5, gpu=True,
                            env_setup=["source ~/env.sh", "module load cuda"], log_dir=str(tmp_path / "logs"),
                            job_name="1 bad name!")
    assert "#$ -pe smp 8" in text
    assert "#$ -l h_rt=01:30:00" in text
    assert "#$ -l mem_free=2560M" in text
    assert "#$ -q gpu.q" in text
    assert 'export CUDA_VISIBLE_DEVICES="${SGE_GPU:-}"' in text
    assert "#$ -N v1_bad_name_" in text
    assert f"#$ -o {tmp_path / 'logs'}/" in text
    assert "source ~/env.sh\nmodule load cuda" in text
    assert "EDIT ME" not in text


def test_slurm_render(tmp_path):
    text = jobscript.render("slurm", "/data/run/valdo.yaml", slots=12, hours=48, mem_gb=3, gpu=True,
                            env_setup="conda activate valdo")
    for line in ("#SBATCH --job-name=valdo", "#SBATCH --chdir=/data/run",
                 "#SBATCH --output=/data/run/joblogs/valdo-%j.log", "#SBATCH --cpus-per-task=12",
                 "#SBATCH --mem-per-cpu=3G", "#SBATCH --time=48:00:00", "#SBATCH --gres=gpu:1"):
        assert line in text.splitlines(), line
    assert 'export VALDO_NCPU="${SLURM_CPUS_PER_TASK:-12}"' in text
    assert "conda activate valdo" in text
    cpu = jobscript.render("slurm", "/data/run/valdo.yaml")
    assert "--gres" not in cpu and 'export CUDA_VISIBLE_DEVICES=""' in cpu


@pytest.mark.parametrize("kwargs", [
    dict(scheduler="pbs"), dict(slots=0), dict(hours=0), dict(mem_gb=0), dict(hours=-1),
])
def test_bad_arguments(kwargs):
    args = dict(scheduler="sge", config_path="/x/valdo.yaml")
    args.update(kwargs)
    with pytest.raises(ValueError):
        jobscript.render(**args)


@needs_bash
@pytest.mark.parametrize("sched", ["sge", "slurm"])
@pytest.mark.parametrize("gpu", [False, True])
def test_bash_syntax(tmp_path, sched, gpu):
    text = jobscript.render(sched, str(tmp_path / "my project" / "valdo.yaml"), gpu=gpu,
                            env_setup="true")
    script = tmp_path / "job.sh"
    script.write_text(text)
    res = subprocess.run([BASH, "-n", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stderr


@needs_bash
def test_runs_valdo_with_config_and_cores(tmp_path):
    proj = tmp_path / "my project"  # a space in the path must survive quoting
    proj.mkdir()
    cfg = proj / "valdo.yaml"
    cfg.write_text("{}\n")
    _fake_valdo(str(tmp_path / "bin"), rc=0)
    text = jobscript.render("sge", str(cfg), slots=6, env_setup="true")
    res = _run(text, tmp_path, path_dirs=[str(tmp_path / "bin")])
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "ARGS:3" in out and "ARG:run" in out and "ARG:-c" in out and f"ARG:{cfg}" in out
    assert "NCPU:6 OMP:6 MKL:6 CUDA:[] MPL:Agg" in out  # NSLOTS unset: the requested slots
    assert "valdo 0.0.0-fake" in out
    # under SGE, NSLOTS (the cores actually granted) wins
    res = _run(text, tmp_path, extra_env={"NSLOTS": "3"}, path_dirs=[str(tmp_path / "bin")])
    assert "NCPU:3 OMP:3" in res.stdout


@needs_bash
def test_exit_code_is_valdos(tmp_path):
    _fake_valdo(str(tmp_path / "bin"), rc=7)
    text = jobscript.render("slurm", str(tmp_path / "valdo.yaml"), slots=2, env_setup="true")
    res = _run(text, tmp_path, extra_env={"SLURM_CPUS_PER_TASK": "5"}, path_dirs=[str(tmp_path / "bin")])
    assert res.returncode == 7
    assert "NCPU:5" in res.stdout
    assert "with exit code 7" in res.stdout


@needs_bash
def test_missing_valdo_fails_clearly(tmp_path):
    text = jobscript.render("sge", str(tmp_path / "valdo.yaml"))  # placeholder env block: nothing activated
    res = _run(text, tmp_path, path_dirs=[str(tmp_path / "empty")])
    assert res.returncode == 1
    assert "'valdo' is not on PATH" in res.stderr


def test_user_site_packages_off_and_log_dir_note(tmp_path):
    """R1-10: packages from 'pip install --user' must not shadow the environment's; the
    'mkdir the log directory' comment only appears while the folder does not exist."""
    cfg = tmp_path / "valdo.yaml"
    logs = tmp_path / "logs"
    text = jobscript.render("sge", str(cfg), log_dir=str(logs), env_setup="true")
    assert "export PYTHONNOUSERSITE=1" in text.splitlines()
    assert text.index("export PYTHONNOUSERSITE=1") > text.index("# --- environment ---")
    assert "The log directory must exist first" in text
    logs.mkdir()
    text = jobscript.render("sge", str(cfg), log_dir=str(logs), env_setup="true")
    assert "The log directory must exist first" not in text
    assert "'locked'" in text   # a resubmission within 5 minutes of a killed job is explained


def test_suggested_wall_time():
    """R1-10: 'valdo jobscript' sizes h_rt from the number of datasets instead of a flat 24 h:
    3x the measured core time plus 15 minutes, in half hours."""
    from valdo.easy.cli import suggest_hours
    assert suggest_hours(300, 16, "input_mtz") == 1.0           # 3*300*30/16 + 900 s = 2588 s
    assert suggest_hours(24, 16, "input_mtz") == 0.5
    assert suggest_hours(300, 16, "refine", "dimple") == 2.0    # + 60 core-s per dataset
    assert suggest_hours(300, 16, "refine", "phenix") == 6.5    # + 360 core-s per dataset
    assert suggest_hours(2000, 4, "input_mtz") == 13.0
    assert suggest_hours(2000, 16, "input_mtz") > suggest_hours(2000, 32, "input_mtz")
