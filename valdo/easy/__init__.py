"""valdo-easy: a one-command front end to the VALDO pipeline.

``valdo init`` writes a project file (valdo.yaml), ``valdo check`` explains what is wrong
with the inputs or the environment, ``valdo run`` runs the upstream VALDO stages in order
(resumable), ``valdo status`` and ``valdo report`` summarize the results. Every command
takes ``--json`` for use by scripts and LLM agents. See docs/easy/DESIGN.md.

This package only orchestrates: the science is done by the upstream stage functions in
``valdo.commandline.pipeline.STAGE_REGISTRY``. Importing it never imports torch.
"""

from valdo import __version__  # noqa: F401  (re-exported)

SCHEMA_ID = "valdo-easy/1"
UPSTREAM_BASE = "1ec32c6"
UPSTREAM_VERSION = "0.0.6"


def version_string():
    """Human-readable version line, e.g. 'valdo 0.0.6+easy.1 (upstream VALDO 0.0.6, commit 1ec32c6)'."""
    return f"valdo {__version__} (upstream VALDO {UPSTREAM_VERSION}, commit {UPSTREAM_BASE})"
