"""Shared fixtures for the valdo-easy tests (tests/easy).

The toy screen (synth.py) and the end-to-end run are built once per session. Tests that need
gemmi/reciprocalspaceship (or torch, for `valdo run`) are skipped where those are missing.
The end-to-end run calls `valdo` in a subprocess (python -m valdo.easy.cli) with
CUDA_VISIBLE_DEVICES='' so it is deterministic and independent of pytest's output capture.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _path in (HERE, REPO):        # synth.py; this checkout of valdo even when it is not installed
    if _path not in sys.path:
        sys.path.insert(0, _path)


def have(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


needs_science = pytest.mark.skipif(not have("gemmi", "reciprocalspaceship"),
                                   reason="needs gemmi and reciprocalspaceship")
needs_torch = pytest.mark.skipif(not have("gemmi", "reciprocalspaceship", "torch", "sklearn"),
                                 reason="needs the full VALDO environment (torch, gemmi, reciprocalspaceship)")


def valdo_check_accepts_environment():
    """True where `valdo check` finds no environment error (check._check_environment): the
    packages it requires are installed and gemmi is 0.6.7 or older."""
    from valdo.easy import check
    versions = check.package_versions()
    required = ("numpy", "pandas", "gemmi", "reciprocalspaceship", "torch", "scipy", "matplotlib", "pyyaml")
    return all(versions.get(n) for n in required) and check._version_tuple(versions["gemmi"]) <= (0, 6, 7)


# for tests that expect `valdo check` (or `valdo run --dry-run`) to pass, e.g. not on a laptop
# without torch or with a newer gemmi
needs_valdo_env = pytest.mark.skipif(not valdo_check_accepts_environment(),
                                     reason="valdo check needs the VALDO environment (torch, gemmi <= 0.6.7)")

# Small but complete settings for the toy screen: a tiny, quick VAE
TOY_ADVANCED = {"epochs": 30, "batch_size": 100}


def valdo_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("VALDO_NCPU", None)
    env["MPLBACKEND"] = "Agg"
    return env


def run_valdo(args, cwd, timeout=1800, env=None):
    """Run `valdo ARGS --json` in a subprocess; returns (CompletedProcess, [json objects])."""
    cmd = [sys.executable, "-m", "valdo.easy.cli"] + list(args)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env or valdo_env())
    objs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                objs.append(json.loads(line))
            except ValueError:
                pass
    return proc, objs


def set_yaml_value(path, key, value):
    """Change one indented 'key: value' line in valdo.yaml (keeps the comments)."""
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            comment = line[line.index("#"):] if "#" in line else ""
            lines[i] = f"{indent}{key}: {value}".ljust(44) + (f" {comment}" if comment else "")
            break
    else:
        raise KeyError(key)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


@pytest.fixture(scope="session")
def toy_screen(tmp_path_factory):
    pytest.importorskip("gemmi")
    pytest.importorskip("reciprocalspaceship")
    import synth
    folder = str(tmp_path_factory.mktemp("toy"))
    truth = synth.make_toy_screen(folder, n_datasets=24, n_bound=4, n_flipped=3, n_broken=1, n_bad=1)
    return folder, truth


@pytest.fixture(scope="session")
def toy_project(toy_screen):
    """valdo init on the toy screen (per-dataset models, apo fallback, bound models, focal residue)."""
    folder, truth = toy_screen
    proc, objs = run_valdo(["init", "data", "--models", "models/{name}-pandda-input.pdb", "--apo", "apo.pdb",
                            "--focal-seqid", str(truth["focal_seqid"]), "--bound-models", "bound",
                            "--ncpu", "2", "--json"], cwd=folder, timeout=300)
    assert proc.returncode == 0, proc.stderr
    path = os.path.join(folder, "valdo.yaml")
    for key, value in TOY_ADVANCED.items():
        set_yaml_value(path, key, value)
    return path, objs[-1]


@pytest.fixture(scope="session")
def toy_run(toy_project):
    """A complete `valdo run` on the toy screen (needs torch)."""
    if not have("torch", "sklearn"):
        pytest.skip("valdo run needs torch and scikit-learn")
    path, _ = toy_project
    folder = os.path.dirname(path)
    proc, objs = run_valdo(["run", "--json"], cwd=folder)
    return path, proc, objs
