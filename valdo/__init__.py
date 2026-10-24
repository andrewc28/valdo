"""VALDO: VAE-assisted ligand discovery for crystallographic fragment screens.

valdo-easy changes to this file (see docs/easy/DESIGN.md, section 10):

* ``__version__`` is read from ``valdo/VERSION`` next to this file, so it is always the
  version of *this* copy (``pkg_resources.require`` failed for an uninstalled checkout,
  needed setuptools < 82, and reported another copy's version if one was installed).
* The public API is unchanged (``valdo.VAE``, ``valdo.Scaler``, ``valdo.Scaler_pool`` and the
  submodules ``preprocessing``, ``reindex``, ``blobs``, ``tag``, ``helper``, ``knn_tools``),
  but those members are imported on first use (PEP 562), so ``import valdo`` no longer
  imports torch. ``valdo --help`` and ``valdo check`` start in a fraction of a second.
"""

import importlib as _importlib
import os as _os


def getVersionNumber():
    """Return the version string from valdo/VERSION (e.g. '0.0.6+easy.1')."""
    here = _os.path.dirname(_os.path.abspath(__file__))
    try:
        with open(_os.path.join(here, "VERSION")) as vfile:
            return vfile.read().strip()
    except OSError:
        try:
            from importlib.metadata import version
            return version("rs-valdo")
        except Exception:
            return "unknown"


__version__ = getVersionNumber()

# Top level API, loaded lazily: name -> (submodule, attribute)
_LAZY_ATTRIBUTES = {
    "Scaler": ("scaling", "Scaler"),
    "Scaler_pool": ("scaling", "Scaler_pool"),
    "VAE": ("vae_networks", "VAE"),
}

# Submodules that upstream imported eagerly (plus the other package modules), loaded lazily
_LAZY_SUBMODULES = (
    "preprocessing", "reindex", "blobs", "tag", "helper", "knn_tools",
    "scaling", "vae_networks", "vae_basics", "commandline", "easy",
)

__all__ = ["VAE", "Scaler", "Scaler_pool", "preprocessing", "reindex", "blobs", "tag",
           "helper", "knn_tools", "__version__"]


def __getattr__(name):
    if name in _LAZY_ATTRIBUTES:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
        value = getattr(_importlib.import_module("." + module_name, __name__), attribute)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES) | set(_LAZY_SUBMODULES))
