"""The kit, as the published figures need it: every name resolves to the published study code."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from .core.book import PART_NAMES
from .core.engine import MatchingEngine

_HERE = Path(__file__).resolve().parent
_REPOSITORY = _HERE.parent.parent
_PARTS = json.loads((_HERE / "parts.json").read_text())      # name -> the published file that defines it
_modules = {}


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    if name not in _PARTS:
        raise AttributeError(f"This figure runs on '{name}', which has not been published yet.")
    path = _REPOSITORY / _PARTS[name]
    if path not in _modules:
        spec = importlib.util.spec_from_file_location("published_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _modules[path] = module
    return getattr(_modules[path], name)


def engine_parts():
    """The engine's book operations, every one of them from the published study code.

    This copy holds no book operations of its own, so the engine cannot run until all of them are published.
    """
    return SimpleNamespace(**{name: __getattr__(name) for name in PART_NAMES})


def new_engine(**options):
    return MatchingEngine(engine_parts(), **options)


def fingerprint():
    return ""
