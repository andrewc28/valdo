"""User documentation and LLM harness files stay in sync with the code (fast, no torch)."""

import json
import os
import re
import shutil

import yaml

from conftest import REPO, needs_science, run_valdo
from valdo.easy import SCHEMA_ID, cli, docgen

DOCS = os.path.join(REPO, "docs", "easy")


def read(*parts):
    with open(os.path.join(REPO, *parts)) as handle:
        return handle.read()


def test_generated_docs_are_up_to_date():
    """CONFIG_REFERENCE.md (schema) and the hint table of TROUBLESHOOTING.md (hints.py)."""
    for path, expected in docgen.expected_files(REPO).items():
        with open(path) as handle:
            assert handle.read() == expected, f"{path} is stale: run python -m valdo.easy.docgen"


def test_config_reference_lists_every_setting():
    from valdo.easy.project import iter_fields
    text = read("docs", "easy", "CONFIG_REFERENCE.md")
    for dotted, _ in iter_fields():
        assert f"| `{dotted}` |" in text


def _codes(module_file, pattern):
    return set(re.findall(pattern, read("valdo", "easy", module_file)))


def test_troubleshooting_explains_every_check_finding_and_error_code():
    text = read("docs", "easy", "TROUBLESHOOTING.md")
    findings = _codes("check.py", r'finding\(\s*[^,()]+,\s*"([a-z_0-9]+)"')
    errors = set()
    for name in ("cli.py", "runner.py", "stages.py"):
        errors |= _codes(name, r'(?:CliError|RunError)\(\s*"([a-z_0-9]+)"')
    errors.discard("internal")          # init's round-trip guard; same text as internal_error
    assert len(findings) > 30 and len(errors) > 15
    missing = sorted(c for c in findings | errors if f"`{c}`" not in text)
    assert not missing, f"TROUBLESHOOTING.md does not mention: {missing}"


def test_every_doc_page_says_who_it_is_for():
    for name in ("QUICKSTART.md", "USER_GUIDE.md", "TROUBLESHOOTING.md", "CONFIG_REFERENCE.md", "WYNTON.md",
                 "CHANGES_FROM_UPSTREAM.md"):
        head = read("docs", "easy", name)[:1500]
        assert "**For:**" in head and "**You need first:**" in head, name


def test_doc_links_resolve():
    pages = [os.path.join(DOCS, n) for n in os.listdir(DOCS) if n.endswith(".md")]
    pages += [os.path.join(REPO, n) for n in ("README.md", "AGENTS.md", "CHANGELOG.md")]
    pages.append(os.path.join(REPO, "integrations", "claude-code", "skills", "valdo", "SKILL.md"))
    bad = []
    for page in pages:
        text = open(page).read()
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not os.path.exists(os.path.join(os.path.dirname(page), target)):
                bad.append(f"{os.path.relpath(page, REPO)} -> {target}")
    assert not bad, bad


def test_version_changelog_and_environment_agree():
    from valdo.easy import UPSTREAM_BASE, __version__
    assert __version__ == "0.0.6+easy.1"
    changelog = read("CHANGELOG.md")
    assert f"## [{__version__}] - 2026-10-24" in changelog and UPSTREAM_BASE in changelog
    env = yaml.safe_load(read("environment.yml"))
    flat = [d for d in env["dependencies"] if isinstance(d, str)]
    assert any(d.startswith("python=3.11") for d in flat)
    pip = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict))
    assert "gemmi==0.6.7" in pip and "reciprocalspaceship==1.0.3" in pip and any(p.startswith("torch==") for p in pip)


def test_changes_from_upstream_names_every_changed_upstream_file():
    text = read("docs", "easy", "CHANGES_FROM_UPSTREAM.md")
    for path in ("valdo/__init__.py", "valdo/VERSION", "valdo/helper.py", "valdo/scaling.py", "valdo/tag.py",
                 "valdo/commandline/pipeline.py", "valdo/commandline/refine.py", "setup.py", "MANIFEST.in",
                 "README.md"):
        assert f"`{path}`" in text, path


def test_agent_harness_files():
    agents = read("AGENTS.md")
    skill = read("integrations", "claude-code", "skills", "valdo", "SKILL.md")
    assert skill.startswith("---\n")
    front = yaml.safe_load(skill.split("---\n")[1])
    assert front["name"] == "valdo" and len(front["description"]) > 80
    for text in (agents, skill):
        assert SCHEMA_ID in text and "--json" in text and "valdo check" in text and '"next"' in text
        for cmd in ("init", "check", "run", "status", "report", "jobscript"):
            assert f"valdo {cmd}" in text, cmd


def test_readme_points_new_users_to_the_quickstart():
    head = read("README.md")[:1200]
    assert "docs/easy/QUICKSTART.md" in head


# ---------------------------------------------------------------------------
# JSON result conventions shared by every command
# ---------------------------------------------------------------------------

def test_next_commands_name_a_non_default_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli._remember_project(str(tmp_path / "valdo.yaml"))
    assert cli._with_project(["valdo run", "open x.html"]) == ["valdo run", "open x.html"]
    other = tmp_path / "my project" / "screen.yaml"
    cli._remember_project(str(other))
    got = cli._with_project(["valdo run", "valdo jobscript -o run_valdo.sh", "qsub x.sh", "valdo stages"])
    assert got == [f"valdo run -c '{other}'", f"valdo jobscript -o run_valdo.sh -c '{other}'", "qsub x.sh",
                   "valdo stages"]
    cli._remember_project(str(tmp_path / "valdo.yaml"))


@needs_science
def test_check_errors_give_an_error_object_and_project_aware_next(toy_project, tmp_path):
    path, _ = toy_project
    folder = os.path.dirname(path)
    small = tmp_path / "two"
    small.mkdir()
    for name in sorted(os.listdir(os.path.join(folder, "data")))[:2]:
        shutil.copy(os.path.join(folder, "data", name), small / name)
    project = yaml.safe_load(open(path))
    text = open(path).read().replace(json.dumps(project["data"]), json.dumps(str(small)), 1)
    other = tmp_path / "small.yaml"
    other.write_text(text)
    proc, objs = run_valdo(["check", "-c", str(other), "--json"], cwd=str(tmp_path / "two"))
    last = objs[-1]
    assert proc.returncode == 1 and last["schema"] == SCHEMA_ID and last["ok"] is False
    assert last["error"]["code"] == "check_failed" and "Too few datasets" in last["error"]["what"]
    assert last["next"] == [f"valdo check -c {other}"]
    assert any(f["code"] == "too_few_datasets" for f in last["findings"])
