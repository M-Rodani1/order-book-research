"""The experiment runner.

An experiment is a function ``experiment(config, seed)`` that returns a
dictionary of numbers and arrays.  :func:`run_experiment` runs it once per seed
and saves everything in a folder named after the configuration, so the same
configuration always lands in the same place.

A figure is a function ``figure(results)`` that returns SVG text.  Each saved
figure sits next to a small file recording the configuration and seeds it came
from; :func:`reproduce_figure` rebuilds the figure from that file alone.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import HOME

EXPERIMENTS: dict = {}
FIGURES: dict = {}


def experiment(name: str):
    def register(function):
        EXPERIMENTS[name] = function
        return function
    return register


def figure(name: str):
    def register(function):
        FIGURES[name] = function
        return function
    return register


def _lookup(table: dict, name: str, kind: str):
    if name not in table:
        try:
            importlib.import_module("lattice.studies.experiments")
        except ModuleNotFoundError:
            pass
    if name not in table:
        raise KeyError(f"No {kind} is registered under the name '{name}'.")
    return table[name]


def canonical(config: dict) -> str:
    """The configuration as text, the same every time for the same content."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def folder_name(config: dict) -> str:
    """A readable name plus a short digest of the whole configuration."""
    label = str(config.get("name") or config.get("experiment") or "experiment")
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "experiment"
    digest = hashlib.sha1(canonical(config).encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


@dataclass
class Results:
    folder: Path | None
    config: dict
    seeds: list
    runs: list = field(default_factory=list)
    fingerprint: str = ""

    def stack(self, key: str) -> np.ndarray:
        """One entry per seed, stacked into an array."""
        return np.stack([np.asarray(run[key]) for run in self.runs])

    def concatenate(self, key: str) -> np.ndarray:
        return np.concatenate([np.atleast_1d(np.asarray(run[key])) for run in self.runs])

    def mean(self, key: str) -> np.ndarray:
        return self.stack(key).mean(axis=0)


def _compute(config: dict, seeds, progress=None) -> list:
    function = _lookup(EXPERIMENTS, config["experiment"], "experiment")
    runs = []
    for number, seed in enumerate(seeds):
        output = function(config, int(seed))
        runs.append({key: np.asarray(value) for key, value in output.items()})
        if progress is not None:
            progress(number + 1, len(seeds))
    return runs


def run_experiment(config: dict, seeds, root: Path | None = None, force: bool = False,
                   fingerprint: str = "", progress=None) -> Results:
    """Run ``config`` once per seed and save the results under ``root``.

    A finished run with the same configuration, seeds and code fingerprint is
    loaded instead of being run again, unless ``force`` is set.
    """
    if "experiment" not in config:
        raise ValueError("A configuration must name its experiment.")
    seeds = [int(seed) for seed in seeds]
    root = Path(root) if root is not None else HOME / "results"
    folder = root / folder_name(config)
    summary_path = folder / "summary.json"
    if not force and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("seeds") == seeds and summary.get("fingerprint", "") == fingerprint:
            return load_results(folder)
    runs = _compute(config, seeds, progress)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for seed, run in zip(seeds, runs):
        np.savez_compressed(folder / f"seed_{seed}.npz", **run)
    scalars = {}
    for key in runs[0]:
        if all(run[key].ndim == 0 for run in runs):
            scalars[key] = [float(run[key]) for run in runs]
    summary_path.write_text(json.dumps({
        "config": config, "seeds": seeds, "fingerprint": fingerprint,
        "numpy": np.__version__, "scalars": scalars,
    }, indent=2, sort_keys=True) + "\n")
    return Results(folder, config, seeds, runs, fingerprint)


def load_results(folder) -> Results:
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    runs = []
    for seed in summary["seeds"]:
        with np.load(folder / f"seed_{seed}.npz") as data:
            runs.append({key: data[key] for key in data.files})
    return Results(folder, summary["config"], summary["seeds"], runs, summary.get("fingerprint", ""))


def render(results: Results, figure_name: str) -> str:
    return _lookup(FIGURES, figure_name, "figure")(results)


def make_figure(config: dict, seeds, figure_name: str, root: Path | None = None,
                force: bool = False, fingerprint: str = "", progress=None) -> dict:
    """Run (or load) the experiment, draw one figure and save it beside its recipe."""
    results = run_experiment(config, seeds, root, force, fingerprint, progress)
    svg = render(results, figure_name)
    figures = results.folder / "figures"
    figures.mkdir(exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "-", figure_name.lower()).strip("-")
    path = figures / f"{stem}.svg"
    path.write_text(svg)
    recipe = {"figure": figure_name, "config": config, "seeds": results.seeds,
              "fingerprint": fingerprint, "sha1": hashlib.sha1(svg.encode()).hexdigest()}
    recipe_path = figures / f"{stem}.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    return {"svg": svg, "path": path, "recipe": recipe_path, "results": results}


def reproduce_figure(recipe_path) -> dict:
    """Rebuild a figure from its saved recipe, computing everything afresh.

    Returns the new SVG and whether it is identical to the one first saved.
    """
    recipe = json.loads(Path(recipe_path).read_text())
    runs = _compute(recipe["config"], recipe["seeds"])
    results = Results(None, recipe["config"], recipe["seeds"], runs, recipe.get("fingerprint", ""))
    svg = render(results, recipe["figure"])
    return {"svg": svg, "identical": hashlib.sha1(svg.encode()).hexdigest() == recipe.get("sha1")}
