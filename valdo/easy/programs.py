"""Refinement programs named in valdo.yaml: ``programs.phenix`` and ``programs.ccp4``.

Setting ``programs.phenix: /path/to/phenix-2.2.1-6174`` (or ``programs.ccp4: /path/to/ccp4-9``) once
in valdo.yaml is enough for every valdo command: ``valdo check`` tests that installation, the
refine stage of ``valdo run`` runs it (on a workstation or in a batch job), and the script written
by ``valdo jobscript`` needs no setup line for it.

valdo builds the program's environment itself. It sources the installation's own setup script
(PHENIX: ``phenix_env.sh``; CCP4: ``bin/ccp4.setup-sh``) in a separate, clean ``bash`` (no
profile or rc files; BASH_ENV and ENV are removed), reads back the environment that script made,
and gives that environment only to the refinement processes. valdo's own process is not changed,
and valdo's Python settings (PYTHONPATH, PYTHONHOME) are not passed on to PHENIX or CCP4. The
setting may name the installation folder, its setup script, its ``bin`` folder or the program
itself. Layouts seen in real installations: PHENIX 2.x ``<root>/phenix_env.sh`` and
``<root>/phenix_bin/phenix.refine`` (a link to ``<root>/bin/phenix.refine``; ``<root>/bin`` also holds
PHENIX's own python, so it is never put on PATH when ``phenix_bin`` exists), PHENIX 1.x
``<root>/phenix_env.sh`` and ``<root>/build/bin/phenix.refine``, CCP4 ``<root>/bin/ccp4.setup-sh``.

Sourcing a setup script runs it: point these settings only at installations you trust (your own or
your lab's PHENIX/CCP4 installation). Only files named like a shell setup script (``*.sh``,
``*.bash``, ``ccp4.setup-sh``) are sourced.

With the setting null (the default) nothing changes: the program must be on PATH, set up in the
shell or job script, as before.
"""

import os
import re
import shutil
import subprocess

FAMILY = {"phenix": "phenix", "dimple": "ccp4"}          # phases.program -> programs.<family>
EXECUTABLES = {"phenix": "phenix.refine", "dimple": "dimple"}
LABELS = {"phenix": "PHENIX", "ccp4": "CCP4"}
SETUP_SCRIPTS = {"phenix": "phenix_env.sh", "ccp4": "ccp4.setup-sh"}
# the variable each setup script sets to its installation folder (used to spot a moved installation)
ROOT_VARIABLES = {"phenix": "PHENIX", "ccp4": "CCP4"}
SOURCE_TIMEOUT_S = 60
VERSION_TIMEOUT_S = 60
# set by bash itself, or meaningless outside the shell that sourced the script
_SHELL_VARIABLES = ("_", "PWD", "OLDPWD", "SHLVL")
# valdo's Python settings, and the files a non-interactive bash would read before the script
_DROPPED = ("PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV")
_ENV_MARKER = "VALDO-ENV-START\0"
_SCRIPT_SUFFIXES = (".sh", ".bash", "setup-sh")

_CACHE = {}


def setting_name(program):
    """'programs.phenix' for phenix, 'programs.ccp4' for dimple."""
    return f"programs.{FAMILY[program]}"


def clear_cache():
    _CACHE.clear()


def clean_setting(value):
    """The path in a programs.* setting, also when the whole shell line was pasted
    ('source /opt/phenix-2.2-6143/phenix_env.sh', with or without quotes, a trailing ';' or a
    following '&& ...'). Returns (path text, note or ""). Raises ValueError for another shell
    command (export, module, setenv), which names no installation folder."""
    raw = str(value).strip()
    text, note = raw, ""
    m = re.match(r"^(?:source|\.)\s+(.+)$", text)
    if m:
        text = re.split(r"\s*(?:&&|;|\|\|)", m.group(1).strip(), maxsplit=1)[0].strip()
    elif re.match(r"^(?:export|setenv|module|ml|eval)\b", text):
        raise ValueError(f"'{raw}' is a shell command; give the installation folder (PHENIX: the one with "
                         "phenix_env.sh; CCP4: the one with bin/ccp4.setup-sh) or its setup script")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    if text != raw:
        note = f"read {raw!r} as the path {text}"
    return text, note


# ---------------------------------------------------------------------------
# Finding the installation
# ---------------------------------------------------------------------------

def _root_of_script(family, script):
    folder = os.path.dirname(script)
    return os.path.dirname(folder) if family == "ccp4" and os.path.basename(folder) == "bin" else folder


def _root_of_exe_dir(folder):
    """<root> of <root>/phenix_bin, <root>/bin or <root>/build/bin."""
    if os.path.basename(folder) in ("bin", "phenix_bin"):
        parent = os.path.dirname(folder)
        return os.path.dirname(parent) if os.path.basename(parent) == "build" else parent
    return folder


def _is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def find_installation(family, path, _resolved=False):
    """Locate an installation from a folder, setup script, bin folder or executable.

    Returns {"root", "script", "exe_dir", "problem"}: ``script`` is the setup script to source (or
    None), ``exe_dir`` a folder with the executable when there is no script (PHENIX only), and
    ``problem`` a plain-language reason when nothing usable was found. Nothing is run or sourced."""
    out = {"root": None, "script": None, "exe_dir": None, "problem": None}
    script_name = SETUP_SCRIPTS[family]
    exe_name = "phenix.refine" if family == "phenix" else "dimple"
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        out["problem"] = f"{path} does not exist"
        return out
    given_file = os.path.isfile(path)
    if given_file:
        base = os.path.basename(path)
        if re.search(r"[._-]t?csh$", base):      # ccp4.setup-csh, phenix_env.csh
            out["problem"] = (f"{path} is a csh script, which valdo cannot source; give the sh version "
                              f"({script_name}) or the installation folder")
            return out
        if base.endswith(_SCRIPT_SUFFIXES):     # a setup script: only files named like one are sourced
            out["script"] = path
            out["root"] = _root_of_script(family, path)
            return out
        start = os.path.dirname(path)            # the program or another file of the installation
    else:
        start = path
    candidates = [start, os.path.dirname(start), os.path.dirname(os.path.dirname(start))]
    for folder in candidates:
        for rel in ((script_name,) if family == "phenix" else (os.path.join("bin", script_name), script_name)):
            script = os.path.join(folder, rel)
            if os.path.isfile(script):
                out["script"] = script
                out["root"] = _root_of_script(family, script)
                return out
    # a link (e.g. /usr/local/bin/phenix.refine -> <root>/phenix_bin/phenix.refine): the installation
    # with a setup script that the link points into comes before the folder of the link itself
    real = os.path.realpath(path)
    linked = find_installation(family, real, _resolved=True) if not _resolved and real != path else None
    if linked and linked["script"]:
        return linked
    if family == "phenix":
        # PHENIX 2.x: phenix_bin before bin (bin also holds PHENIX's python and pip)
        for folder in [start] + [os.path.join(start, sub) for sub in ("phenix_bin", "bin", os.path.join("build", "bin"))]:
            if _is_executable(os.path.join(folder, exe_name)):
                out["exe_dir"] = folder
                out["root"] = _root_of_exe_dir(folder)
                return out
    if linked and not linked["problem"]:
        return linked
    if given_file and os.path.basename(path) != exe_name:
        out["problem"] = (f"{path} is not {LABELS[family]}'s setup script; give the installation folder "
                          f"(the one with {script_name if family == 'phenix' else os.path.join('bin', script_name)})")
        return out
    inside = _installations_below(family, path) if not given_file else []
    if len(inside) == 1:
        out["problem"] = (f"{path} is not a {LABELS[family]} installation itself, but it holds one: set "
                          f"programs.{family} to {inside[0]}")
    elif inside:
        out["problem"] = (f"{path} is not a {LABELS[family]} installation itself, but it holds {len(inside)}: "
                          f"{', '.join(os.path.basename(p) for p in inside)}; set programs.{family} to the folder "
                          "of the one to use")
    elif family == "phenix":
        out["problem"] = (f"no {script_name} and no {exe_name} found in or near {path}; give the PHENIX "
                          f"installation folder (the one with {script_name})")
    else:
        out["problem"] = (f"no CCP4 setup script ({os.path.join('bin', script_name)}) found in or near {path}; "
                          "give the CCP4 installation folder (e.g. /opt/ccp4-9) or the script itself")
    return out


def _installations_below(family, folder):
    """Installations one level below ``folder`` (a lab's software folder), by their setup script."""
    rel = SETUP_SCRIPTS[family] if family == "phenix" else os.path.join("bin", SETUP_SCRIPTS[family])
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names if os.path.isfile(os.path.join(folder, n, rel))]


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

def base_environment(environ=None):
    """The environment the setup script starts from: valdo's, without its Python settings (and
    without BASH_ENV/ENV, which bash would otherwise read before the script)."""
    env = dict(os.environ if environ is None else environ)
    for key in _DROPPED:
        env.pop(key, None)
    return env


def source_script(script, base_env, timeout_s=SOURCE_TIMEOUT_S):
    """The environment after sourcing ``script`` in a clean bash: (env dict, problem or None).

    The script is bash's $0, so it sees no arguments, and what it prints goes to stderr; the
    environment follows a marker on stdout, so stray output cannot be mistaken for it."""
    bash = shutil.which("bash", path=base_env.get("PATH")) or shutil.which("bash") or "/bin/bash"
    code = 'source "$0" 1>&2; printf "VALDO-ENV-START\\0"; command env -0'
    try:
        proc = subprocess.run([bash, "--noprofile", "--norc", "-c", code, script],
                              env=base_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout_s, cwd=os.path.dirname(script) or None)
    except subprocess.TimeoutExpired:
        return None, f"sourcing {script} did not finish within {timeout_s} s"
    except OSError as exc:
        return None, f"could not run bash to source {script}: {exc}"
    raw = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    why = f" ({err[-1][:200]})" if err else ""
    if _ENV_MARKER not in raw:
        return None, f"sourcing {script} in bash gave no environment back (the script stopped the shell){why}"
    raw = raw.split(_ENV_MARKER, 1)[1]
    if "\0" not in raw:
        return None, (f"after sourcing {script}, 'env' could not be run (does the script replace PATH?){why}")
    env = {}
    for item in raw.split("\0"):
        if "=" in item:
            key, value = item.split("=", 1)
            if key:
                env[key] = value
    for key in _SHELL_VARIABLES:
        env.pop(key, None)
        if key in base_env:
            env[key] = base_env[key]
    return env, None


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def _run_text(cmd, env, timeout_s=VERSION_TIMEOUT_S):
    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, timeout=timeout_s, text=True)
        return res.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def phenix_version(exe, env, root=None, probe=True):
    """'2.2.1-6174' style version: PHENIX_VERSION (set by phenix_env.sh), the folder name, or phenix.version."""
    if env and env.get("PHENIX_VERSION"):
        return env["PHENIX_VERSION"]
    for text in (root or "", exe or "", os.path.realpath(exe) if exe else ""):
        m = re.search(r"phenix-(\d+\.\d+(?:\.\d+)?(?:[-.]\d+)?)", text)
        if m:
            return m.group(1)
    if probe and exe:
        ver_exe = os.path.join(os.path.dirname(exe), "phenix.version")
        if os.path.isfile(ver_exe):
            out = _run_text([ver_exe], env)
            ver = re.search(r"Version\s*:\s*(\S+)", out)
            tag = re.search(r"Release tag\s*:\s*(\S+)", out)
            if ver:
                return ver.group(1) + (f"-{tag.group(1)}" if tag and tag.group(1) not in ver.group(1) else "")
    return None


def ccp4_version(exe, env, probe=True):
    """'CCP4 9.0.017, dimple 2.7' style version ($CCP4/lib/ccp4/MAJOR_MINOR and dimple --version)."""
    parts = []
    ccp4 = (env or {}).get("CCP4")
    if ccp4:
        try:
            with open(os.path.join(ccp4, "lib", "ccp4", "MAJOR_MINOR")) as handle:
                text = handle.read().strip()
            if text:
                parts.append(f"CCP4 {text}")
        except OSError:
            pass
    if probe and exe:
        m = re.search(r"dimple\s+([0-9][\w.\-]*)", _run_text([exe, "--version"], env))
        if m:
            parts.append(f"dimple {m.group(1)}")
    return ", ".join(parts) or None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve(program, configured=None, probe_version=True, environ=None):
    """Where ``program`` (phenix or dimple) comes from and the environment to run it in.

    ``configured``: the path from valdo.yaml (programs.phenix / programs.ccp4), or None to use PATH.
    Returns a dict: program, family, setting, configured, source ("valdo.yaml" or "PATH"), root,
    setup_script, exe, version, env (None = valdo's own environment, the PATH behaviour), error
    (None or {"code", "what", "fix"}) and notes. Results are cached per process."""
    if program not in FAMILY:
        raise ValueError(f"unknown refinement program {program!r}; use 'phenix' or 'dimple'")
    key = (program, configured, os.environ.get("PATH") if environ is None else tuple(sorted(environ.items())))
    if configured and key in _CACHE:      # sourcing a setup script is done once per process
        return _CACHE[key]
    family = FAMILY[program]
    info = {"program": program, "family": family, "setting": setting_name(program), "configured": configured,
            "source": "valdo.yaml" if configured else "PATH", "root": None, "setup_script": None, "exe": None,
            "version": None, "env": None, "error": None, "notes": []}
    exe_name = EXECUTABLES[program]
    label = LABELS[family]
    if not configured:
        env = dict(os.environ if environ is None else environ)
        exe = shutil.which(exe_name, path=env.get("PATH"))
        info["exe"] = exe
        if exe is None:
            info["error"] = {"code": "refine_program_missing",
                             "what": f"{exe_name} is not on PATH and {info['setting']} is not set in valdo.yaml",
                             "fix": f"Set {info['setting']} in valdo.yaml to your {label} installation "
                                    f"(e.g. {info['setting']}: {example(family)}), or set {label} up in this "
                                    "shell and in the job script."}
        else:
            info["version"] = (phenix_version(exe, env, probe=probe_version) if family == "phenix"
                               else ccp4_version(exe, env, probe=probe_version))
            if family == "ccp4" and not env.get("CCP4"):
                info["notes"].append("dimple is on PATH but the CCP4 variable is not set: source "
                                     "<ccp4>/bin/ccp4.setup-sh first, or set programs.ccp4 in valdo.yaml")
        return info
    place = find_installation(family, configured)
    info["root"], info["setup_script"] = place["root"], place["script"]
    if place["problem"]:
        info["error"] = {"code": "program_not_found", "what": f"{info['setting']}: {place['problem']}",
                         "fix": f"Set {info['setting']} in valdo.yaml to the {label} installation folder "
                                f"(e.g. {example(family)})."}
        _CACHE[key] = info
        return info
    base = base_environment(environ)
    test = f"bash -c 'source \"{place['script']}\" && which {exe_name}'" if place["script"] else ""
    if place["script"]:
        env, problem = source_script(place["script"], base)
        if problem:
            info["error"] = {"code": "program_setup_failed", "what": f"{info['setting']}: {problem}",
                             "fix": f"Check that the setup script works in a plain shell: {test}"}
            _CACHE[key] = info
            return info
    else:
        env = dict(base)
        env["PATH"] = place["exe_dir"] + (os.pathsep + base["PATH"] if base.get("PATH") else "")
        info["notes"].append(f"no {SETUP_SCRIPTS[family]} found; using {place['exe_dir']} on PATH")
        version = phenix_version(None, None, root=place["root"], probe=False)
        if version and not env.get("PHENIX_VERSION"):    # as phenix_env.sh would; PHENIX 2.x writes it into
            env["PHENIX_VERSION"] = version                # the output model's header (else 'PHENIX (???)')
    exe = shutil.which(exe_name, path=env.get("PATH", ""))
    if exe is None:
        moved = env.get(ROOT_VARIABLES[family])
        if place["script"] and moved and not os.path.isdir(moved):
            what = (f"{info['setting']}: {os.path.basename(place['script'])} sets {ROOT_VARIABLES[family]}={moved}, "
                    "a folder that does not exist: the installation was moved or copied after it was installed")
            fix = (f"Install {label} again in its final place, or point {info['setting']} at the original "
                   f"installation. Test: {test}")
        else:
            what = (f"{info['setting']}: {exe_name} is not on PATH after sourcing "
                    f"{place['script'] or place['exe_dir']}")
            fix = (f"Check that this {label} installation has {exe_name} "
                   + ("(CCP4 7.1 or newer includes dimple)" if family == "ccp4" else
                      "and that its setup script is the one in the installation folder")
                   + (f". Test: {test}" if test else "."))
        info["error"] = {"code": "program_setup_failed", "what": what, "fix": fix}
        _CACHE[key] = info
        return info
    if family == "ccp4" and not shutil.which("refmac5", path=env.get("PATH", "")):
        info["notes"].append("refmac5 is not on PATH after sourcing the CCP4 setup script")
    info["exe"], info["env"] = exe, env
    info["version"] = (phenix_version(exe, env, root=place["root"], probe=probe_version) if family == "phenix"
                       else ccp4_version(exe, env, probe=probe_version))
    _CACHE[key] = info
    return info


def for_project(project, probe_version=True):
    """resolve() for the project's refinement program, or None when phases.source is not refine."""
    if project["phases"]["source"] != "refine" or not project["phases"]["program"]:
        return None
    program = project["phases"]["program"]
    configured = (project.get("programs") or {}).get(FAMILY[program])
    return resolve(program, configured, probe_version=probe_version)


def public(info):
    """The JSON-safe part of a resolve() result (no environment)."""
    if info is None:
        return None
    return {k: v for k, v in info.items() if k != "env"}


def short(info):
    """The program and its version only: 'PHENIX 2.2.1-6174', 'CCP4 9.0.017, dimple 2.7'."""
    if info is None:
        return ""
    if info["family"] == "phenix":
        return f"PHENIX {info['version']}" if info.get("version") else "PHENIX"
    return info.get("version") or "dimple"


def describe(info):
    """One line: 'PHENIX 2.2.1-6174 (phenix.refine) at /opt/phenix-2.2.1-6174 (programs.phenix in valdo.yaml)'."""
    if info is None:
        return ""
    name = "phenix.refine" if info["program"] == "phenix" else "dimple"
    where = info.get("root") if info.get("source") == "valdo.yaml" and info.get("root") else info.get("exe")
    origin = f"{info['setting']} in valdo.yaml" if info.get("source") == "valdo.yaml" else "found on PATH"
    if info.get("error"):
        return f"{name}: {info['error']['what']}"
    return f"{short(info)} ({name}) at {where} ({origin})"


def version_differs(recorded, now):
    """True if ``now`` (a resolve() result) is another version of the program ``recorded`` describes.

    Only the version counts: the same version in another folder, a switch to the other program
    (the per-dataset fingerprint then refines every dataset again anyway) or an unknown version is
    no change. A version that is the start of the other ('CCP4 9.0.017' against 'CCP4 9.0.017,
    dimple 2.7', when ``dimple --version`` was not run) is the same version."""
    if not recorded or not now or recorded.get("program") != now.get("program"):
        return False
    return _versions_differ(recorded.get("version"), now.get("version"))


def _versions_differ(a, b):
    if not a or not b:
        return False
    return not (a == b or a.startswith(b) or b.startswith(a))


def installation_on_path(program):
    """For `valdo init`: the installation folder of ``program`` found on PATH, if it has the setup
    script valdo needs (else None). Returns (folder or None, note)."""
    exe = shutil.which(EXECUTABLES[program])
    if not exe:
        return None, f"{EXECUTABLES[program]} is not on PATH"
    family = FAMILY[program]
    place = find_installation(family, exe)
    if place["script"]:
        return place["root"], f"{EXECUTABLES[program]} on PATH ({exe}) belongs to {place['root']}"
    return None, (f"{EXECUTABLES[program]} is on PATH ({exe}) but no {SETUP_SCRIPTS[family]} was found next to "
                  f"it; {setting_name(program)} is left empty (the program on PATH is used)")


def example(family):
    """An example installation folder for messages."""
    return "/path/to/phenix-2.2.1-6174" if family == "phenix" else "/path/to/ccp4-9"
