"""Hawkes processes: arrivals that make further arrivals more likely for a while.

With exponential kernels, the intensity of event type ``i`` at time ``t`` is

    lambda_i(t) = baseline_i + sum over past events (type j at time s) of
                  excitation[i][j] * exp(-decay[i][j] * (t - s)).

Fitted parameters are read from a file with this layout::

    {
      "event_types": ["limit_buy", "limit_sell", "market_buy", "market_sell", "cancel_buy", "cancel_sell"],
      "baseline":   [ ... one number per event type ... ],
      "excitation": [[ ... ], ... one row per event type ... ],
      "decay":      [[ ... ], ... one row per event type ... ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PARAMETER_FILE = "data/hawkes_parameters.json"


@dataclass
class HawkesParameters:
    event_types: list
    baseline: np.ndarray
    excitation: np.ndarray
    decay: np.ndarray

    def validate(self) -> None:
        n = len(self.event_types)
        if self.baseline.shape != (n,):
            raise ValueError(f"baseline needs {n} numbers, one per event type.")
        if self.excitation.shape != (n, n) or self.decay.shape != (n, n):
            raise ValueError(f"excitation and decay must each be {n} rows of {n} numbers.")
        if (self.baseline < 0).any() or (self.excitation < 0).any() or (self.decay <= 0).any():
            raise ValueError("baseline and excitation must not be negative, and decay must be positive.")
        radius = max(abs(np.linalg.eigvals(self.excitation / self.decay)))
        if radius >= 1:
            raise ValueError(
                f"These parameters describe a process that explodes (branching ratio {radius:.3f}, must be below 1).")

    @property
    def branching_ratio(self) -> float:
        return float(max(abs(np.linalg.eigvals(self.excitation / self.decay))))


def load_parameters(path) -> HawkesParameters:
    data = json.loads(Path(path).read_text())
    missing = [key for key in ("event_types", "baseline", "excitation", "decay") if key not in data]
    if missing:
        raise ValueError(f"The parameter file is missing: {', '.join(missing)}.")
    parameters = HawkesParameters(
        list(data["event_types"]),
        np.asarray(data["baseline"], dtype=float),
        np.asarray(data["excitation"], dtype=float),
        np.asarray(data["decay"], dtype=float),
    )
    parameters.validate()
    return parameters


def intensity(parameters: HawkesParameters, time: float, events: list) -> np.ndarray:
    """The intensity of every event type at ``time`` given past ``(time, type_index)`` events."""
    rates = parameters.baseline.astype(float).copy()
    for when, kind in events:
        if when < time:
            rates += parameters.excitation[:, kind] * np.exp(-parameters.decay[:, kind] * (time - when))
    return rates


