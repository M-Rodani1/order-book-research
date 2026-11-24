"""Study 2 experiments and figures.  Each runs on the kit, so on the learner's code once it has passed."""

from __future__ import annotations

import numpy as np

from lattice import HOME, kit
from lattice.core import figures, hawkes
from lattice.core.runner import experiment, figure

from .markets import build_market, depth_by_distance, measure_rng


def _measure(spec: dict, config: dict, seed: int, extra=()) -> dict:
    """Warm one market up, run the large orders through it and return every order's size and impact."""
    market, _ = build_market(spec, seed, extra=extra)
    empty_before = market.empty_side_steps
    sizes, impacts = kit.measure_impact(market, list(config["sizes"]), int(config["orders_per_size"]),
                                        int(config["gap"]), measure_rng(seed))
    return {"sizes": sizes, "impacts": impacts, "steps": market.time,
            "empty_side_steps": market.empty_side_steps - empty_before}


def _fit(results, prefix: str = "") -> dict:
    """The pooled impact curve of every seed, its fitted power law and the bootstrap interval."""
    sizes = results.concatenate(f"{prefix}sizes")
    impacts = results.concatenate(f"{prefix}impacts")
    curve = kit.impact_curve(sizes, impacts)
    exponent, prefactor = kit.fit_exponent(sizes, impacts)
    settings = results.config["bootstrap"]
    low, high = kit.bootstrap_interval(sizes, impacts, np.random.default_rng(int(settings["seed"])),
                                       resamples=int(settings["resamples"]), level=float(settings["level"]))
    return {"curve": curve, "exponent": exponent, "prefactor": prefactor, "low": low, "high": high,
            "level": float(settings["level"])}


def _fit_series(fit: dict, label: str, colour: str) -> list:
    """The points of one impact curve and the straight line fitted through them, for a figure with logarithmic axes."""
    curve = fit["curve"]
    ends = np.array([curve["size"].min(), curve["size"].max()], dtype=float)
    return [
        {"x": curve["size"], "y": curve["impact"], "label": label, "colour": colour, "points": True, "line": False},
        {"x": ends, "y": fit["prefactor"] * ends ** fit["exponent"], "colour": colour, "dashed": True},
    ]


def _seed_exponents(results, prefix: str = "") -> list:
    """The exponent fitted to each seed's own orders (None where a seed's curve cannot be fitted)."""
    fitted = []
    for run in results.runs:
        try:
            fitted.append(float(kit.fit_exponent(run[f"{prefix}sizes"], run[f"{prefix}impacts"])[0]))
        except (ValueError, IndexError):
            fitted.append(None)
    return fitted


def _strip_row(results, fit: dict, label: str, colour: str, prefix: str = "") -> dict:
    return {"label": label, "colour": colour, "pooled": fit["exponent"], "low": fit["low"], "high": fit["high"],
            "seeds": _seed_exponents(results, prefix)}


def _interval_text(fit: dict) -> str:
    return f"{fit['exponent']:.2f} ({fit['low']:.2f} to {fit['high']:.2f})"


# -- steps 4 and 5: impact in the market of random traders ---------------------------

@experiment("study2.random_market_impact")
def random_market_impact(config: dict, seed: int) -> dict:
    return _measure(config["market"], config, seed)


@figure("study2.impact_curve")
def impact_curve_figure(results) -> str:
    curve = kit.impact_curve(results.concatenate("sizes"), results.concatenate("impacts"))
    series = [{"x": curve["size"], "y": curve["impact"], "error": curve["error"], "label": "mean impact",
               "colour": figures.LIME, "points": True}]
    orders = int(np.min(curve["count"])) if len(curve["count"]) else 0
    return figures.line_figure(series, "Impact against size", "size of the large order in shares",
                               "price moved in ticks", zero_line=True,
                               note=f"{orders} large orders of each size")


@figure("study2.impact_exponent")
def impact_exponent_figure(results) -> str:
    fit = _fit(results)
    series = _fit_series(fit, "mean impact", figures.LIME)
    series[1]["label"] = "fitted power law"
    return figures.line_figure(series, "Fitting the exponent", "size of the large order in shares",
                               "price moved in ticks", log_x=True, log_y=True,
                               note=f"exponent {fit['exponent']:.2f}, {100 * fit['level']:.0f}% interval "
                                    f"{fit['low']:.2f} to {fit['high']:.2f}",
                               strip=[_strip_row(results, fit, "random traders only", figures.LIME)])


# -- step 6: where the hidden liquidity sits -------------------------------------------

@experiment("study2.hidden_profile")
def hidden_profile(config: dict, seed: int) -> dict:
    market, hidden = build_market(config["market"], seed)
    levels = int(config["levels"])
    shown, unseen, spreads = np.zeros(levels), np.zeros(levels), []
    for _ in range(int(config["samples"])):
        market.run(int(config["sample_every"]))
        sample = depth_by_distance(market, hidden, levels)
        if sample is not None:
            shown += sample[0]
            unseen += sample[1]
            spreads.append(market.spread)
    count = max(len(spreads), 1)
    return {"shown": shown / count, "hidden": unseen / count, "samples": len(spreads),
            "mean_spread": float(np.mean(spreads)) if spreads else np.nan}


@figure("study2.hidden_profile")
def hidden_profile_figure(results) -> str:
    distance = np.arange(results.runs[0]["shown"].shape[0])
    shown, unseen = results.stack("shown"), results.stack("hidden")
    total = shown + unseen
    series = [
        {"x": distance, "y": total.mean(axis=0), "error": total.std(axis=0), "label": "shown and hidden together",
         "colour": figures.LIME, "points": True},
        {"x": distance, "y": shown.mean(axis=0), "label": "shown in the book", "colour": figures.BLUE, "points": True},
        {"x": distance, "y": unseen.mean(axis=0), "label": "hidden", "colour": figures.ORANGE, "dashed": True},
    ]
    spread = float(results.stack("mean_spread").mean())
    return figures.line_figure(series, "Where the liquidity sits", "ticks behind the best price",
                               "average shares at one price", note=f"mean spread {spread:.2f} ticks")


# -- step 7: three markets side by side ------------------------------------------------

MARKETS = (("random", "random traders only", figures.BLUE),
           ("both", "random traders and hidden liquidity", figures.LIME),
           ("hidden", "hidden liquidity only", figures.ORANGE))


@experiment("study2.three_markets")
def three_markets(config: dict, seed: int) -> dict:
    output = {}
    for label, _, _ in MARKETS:
        for key, value in _measure(config[f"{label}_market"], config, seed).items():
            output[f"{label}_{key}"] = value
    return output


def three_fits(results) -> dict:
    return {label: _fit(results, f"{label}_") for label, _, _ in MARKETS}


@figure("study2.three_markets")
def three_markets_figure(results) -> str:
    fits = three_fits(results)
    series, strip = [], []
    for label, name, colour in MARKETS:
        series += _fit_series(fits[label], f"{name}: {_interval_text(fits[label])}", colour)
        strip.append(_strip_row(results, fits[label], name, colour, f"{label}_"))
    return figures.line_figure(series, "Impact in three markets", "size of the large order in shares",
                               "price moved in ticks", log_x=True, log_y=True, strip=strip)


# -- step 9 (optional): Hawkes-driven market orders ------------------------------------

def _market_order_rate(parameters) -> float:
    """Long-run market orders per unit of Hawkes time, from the stationary rates of the process."""
    kinds = list(parameters.event_types)
    missing = [kind for kind in ("market_buy", "market_sell") if kind not in kinds]
    if missing:
        raise ValueError(f"The parameter file has no event type called {' or '.join(missing)}.")
    stationary = np.linalg.solve(np.eye(len(kinds)) - parameters.excitation / parameters.decay, parameters.baseline)
    return float(stationary[kinds.index("market_buy")] + stationary[kinds.index("market_sell")])


@experiment("study2.hawkes_impact")
def hawkes_impact(config: dict, seed: int) -> dict:
    parameters = hawkes.load_parameters(HOME / config["parameter_file"])
    rate = float(config["market_orders_per_step"])
    time_per_step = rate / _market_order_rate(parameters)
    # Enough Hawkes time for the warm-up and every large order, with a margin.
    orders = len(config["sizes"]) * int(config["orders_per_size"])
    steps = int(config["market"]["warm_up"]) + int(sum(config["sizes"])) * int(config["orders_per_size"]) \
        + orders * (int(config["gap"]) + 3)
    horizon = 1.05 * steps * time_per_step
    output = {}
    # The same providers of liquidity and the same long-run rate of market orders,
    # once arriving in bursts (Hawkes) and once evenly at random (Poisson).
    for label in ("hawkes", "poisson"):
        if label == "hawkes":
            taker = kit.HawkesTaker(parameters, horizon, time_per_step=time_per_step, size=1)
        else:
            taker = kit.RandomTrader(limit_rate=0.0, market_rate=rate, cancel_probability=0.0, band=1, size=1,
                                     name="poisson")
        for key, value in _measure(config["market"], config, seed, extra=[taker]).items():
            output[f"{label}_{key}"] = value
    output["time_per_step"] = time_per_step
    return output


@figure("study2.hawkes_impact")
def hawkes_impact_figure(results) -> str:
    bursts, even = _fit(results, "hawkes_"), _fit(results, "poisson_")
    series = (_fit_series(bursts, "Hawkes arrivals", figures.LIME)
              + _fit_series(even, "Poisson arrivals", figures.BLUE))
    return figures.line_figure(series, "Impact with Hawkes arrivals", "size of the large order in shares",
                               "price moved in ticks", log_x=True, log_y=True,
                               note=f"exponents {_interval_text(bursts)} and {_interval_text(even)}",
                               strip=[_strip_row(results, bursts, "Hawkes arrivals", figures.LIME, "hawkes_"),
                                      _strip_row(results, even, "Poisson arrivals", figures.BLUE, "poisson_")])
