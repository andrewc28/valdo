"""valdo.yaml: schema, strict validation, did-you-mean, coercion, paths, writer round trip."""

import os

import pytest

pytest.importorskip("yaml")

from valdo.easy import project as P  # noqa: E402


def write(tmp_path, text, name="valdo.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


MINIMAL = "data: mtz\nmodels:\n  apo: apo.pdb\n"


def test_minimal_project_gets_every_default(tmp_path):
    proj, notes = P.load_project(write(tmp_path, MINIMAL))
    assert notes == []
    assert proj["data"] == str(tmp_path / "mtz")                  # relative to the project file
    assert proj["models"]["apo"] == str(tmp_path / "apo.pdb")
    assert proj["workdir"] == str(tmp_path / "valdo_run")
    assert proj["_path"] == str(tmp_path / "valdo.yaml") and proj["_dir"] == str(tmp_path)
    # validated PTP1B / upstream template values, all explicit
    t = proj["advanced"]["train"]
    assert (t["latent_dim"], t["n_hidden_layers"], t["epochs"], t["stdof"], t["random_seed"], t["activation"]) == \
        (7, [3, 6], 500, 128, 11231, "relu")
    assert proj["advanced"]["reindex"] == {"wcorr": True, "cc_min_dif": 0.2}
    assert proj["advanced"]["scale"]["when_opt"] == "all"
    assert proj["advanced"]["filter"] == {"max_rfree": 0.45, "min_cc": 0.55}
    assert proj["reindex"] == "auto" and proj["reference"] == "auto"


def test_every_schema_field_documented():
    for dotted, spec in P.iter_fields():
        assert spec.doc, dotted


def test_unknown_key_did_you_mean(tmp_path):
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, MINIMAL + "advanced:\n  train:\n    epoch: 10\n"))
    msg = "\n".join(err.value.messages)
    assert "advanced.train.epoch: unknown setting" in msg and "Did you mean 'advanced.train.epochs'?" in msg


def test_misplaced_key_points_to_its_section(tmp_path):
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, MINIMAL + "latent_dim: 5\n"))
    assert "Did you mean 'advanced.train.latent_dim'?" in err.value.messages[0]


def test_numeric_text_is_converted_with_a_note(tmp_path):
    proj, notes = P.load_project(write(tmp_path, MINIMAL + "advanced:\n  train:\n    learning_rate: 1e-3\n"))
    assert proj["advanced"]["train"]["learning_rate"] == pytest.approx(0.001)
    assert any("learning_rate" in n for n in notes)


def test_int_where_float_expected_is_fine(tmp_path):
    proj, notes = P.load_project(write(tmp_path, MINIMAL + "advanced:\n  train:\n    w_kl: 1\n"))
    assert proj["advanced"]["train"]["w_kl"] == 1.0 and isinstance(proj["advanced"]["train"]["w_kl"], float)
    assert notes == []


@pytest.mark.parametrize("text, fragment", [
    ("advanced:\n  train:\n    train_fraction: 1.0\n", "train_fraction"),
    ("advanced:\n  train:\n    include_errors: false\n", "include_errors"),
    ("advanced:\n  reconstruct:\n    repeats: 3\n", "repeats"),
    ("advanced:\n  train:\n    activation: elu\n", "must be one of relu, tanh, sigmoid"),
    ("advanced:\n  train:\n    n_hidden_layers: [3]\n", "n_hidden_layers"),
    ("advanced:\n  scale:\n    when_opt: sometimes\n", "when_opt"),
    ("advanced:\n  reindex:\n    cc_min_dif: 0\n", "cc_min_dif"),
    ("reindex: maybe\n", "auto, true or false"),
    ("phases:\n  source: refine\n", "phases.program"),
    ("phases:\n  source: existing\n", "phases.existing"),
    ("compute:\n  ncpu: 0\n", "at least 1"),
    ("advanced:\n  train:\n    epochs: 2.5\n", "whole number"),
])
def test_invalid_settings_are_explained(tmp_path, text, fragment):
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, MINIMAL + text))
    assert fragment in "\n".join(err.value.messages)


def test_models_required(tmp_path):
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, "data: mtz\n"))
    assert "models" in err.value.messages[0]


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(P.ProjectError) as err:
        P.load_project(str(tmp_path / "nope.yaml"))
    assert "valdo init" in err.value.messages[0]
    with pytest.raises(P.ProjectError):
        P.load_project(write(tmp_path, "data: [unclosed\n"))
    with pytest.raises(P.ProjectError):
        P.load_project(write(tmp_path, "- a list\n"))


def test_user_and_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("VALDO_TEST_DATA", str(tmp_path / "somewhere"))
    proj, _ = P.load_project(write(tmp_path, "data: $VALDO_TEST_DATA/mtz\nmodels:\n  apo: ~/apo.pdb\n"))
    assert proj["data"] == str(tmp_path / "somewhere" / "mtz")
    assert proj["models"]["apo"] == os.path.join(os.path.expanduser("~"), "apo.pdb")


def test_when_opt_number_and_bool_words(tmp_path):
    proj, notes = P.load_project(write(tmp_path, MINIMAL + "advanced:\n  scale:\n    when_opt: 0\n"
                                       "  report:\n    maps: 'no'\n"))
    assert proj["advanced"]["scale"]["when_opt"] == 0.0 and isinstance(proj["advanced"]["scale"]["when_opt"], float)
    assert proj["advanced"]["report"]["maps"] is False
    assert len(notes) == 2


def test_writer_round_trip(tmp_path):
    proj = P.default_project()
    proj["data"] = "/data/*.mtz"
    proj["models"]["apo"] = "/models/apo.pdb"
    proj["advanced"]["train"]["learning_rate"] = 1e-5
    proj["focal_residue"]["seqid"] = 215
    proj["focal_residue"]["tag"] = "cys215"
    path = str(tmp_path / "valdo.yaml")
    P.write_project(proj, path, header_lines=["detected: something"])
    text = open(path).read()
    assert "# detected: something" in text and "1. YOUR PROJECT" in text and "3. ADVANCED" in text
    loaded, notes = P.load_project(path)
    assert notes == []
    for dotted, _ in P.iter_fields():
        if dotted == "workdir":
            continue
        assert P.get(loaded, dotted) == P.get(proj, dotted), dotted


def test_format_scalar_keeps_floats_float():
    import yaml
    for value in (1.0, 1e-5, 0.001, 2.5e10, 100.0):
        assert isinstance(yaml.safe_load(P.format_scalar(value)), float)
    assert P.format_scalar("a: b") == '"a: b"'
    assert P.format_scalar([2, 4.0]) == "[2, 4.0]"


def test_yaml_syntax_error_names_the_line(tmp_path):
    """Review R1-08: say where the YAML is broken and what usually causes it."""
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, "data: mtz\nmodels:\n  apo: apo.pdb\n   per_dataset: x\n"))
    assert err.value.kind == "syntax"
    msg = err.value.messages[0]
    assert "is not valid YAML near line" in msg and "4" in msg and "indented by 2 spaces" in msg
    assert err.value.details and err.value.details[0].startswith("YAML parser:")
    with pytest.raises(P.ProjectError) as err:
        P.load_project(str(tmp_path / "missing.yaml"))
    assert err.value.kind == "missing"
    with pytest.raises(P.ProjectError) as err:
        P.load_project(write(tmp_path, MINIMAL + "advanced:\n  train:\n    epoch: 10\n"))
    assert err.value.kind == "settings"
