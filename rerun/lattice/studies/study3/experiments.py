"""Study 3 experiments and figures.  Each runs on the kit, so on the learner's code once it has passed."""

from __future__ import annotations

import numpy as np

from lattice import kit
from lattice.core import Marketplace, figures
from lattice.core.runner import experiment, figure

REGIME_CODES = {"collapsed": -1.0, "frozen": 0.0, "continuous": 1.0}
REGIME_COLOURS = {"continuous": figures.BLUE, "collapsed": figures.ORANGE, "frozen": "#5B6370"}


def _quoters(config: dict, count: int, depth: int) -> list:
    """Fresh learning quoters with the settings written in the configuration."""
    return [kit.LearningQuoter(depth=int(depth), learning_rate=config["learning_rate"],
                               exploration=config["exploration"], discount=config["discount"],
                               penalty=config["penalty"], name=f"quoter {number + 1}")
            for number in range(int(count))]


# -- step 4: the learning curve ----------------------------------------------------

def _fixed_quoters(config: dict, count: int) -> list:
    """Quoters that never learn: each always quotes one tick from the midpoint on both sides."""
    quoters = _quoters(config, count, 1)
    for quoter in quoters:
        quoter.choose = lambda state: (1, 1)
        quoter.learning = False
    return quoters


@experiment("study3.learning_curve")
def learning_curve(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    learners = _quoters(config, config["quoters"], config["depth"])
    # The yardstick: a quoter in the same market that picks its distances at random at every step.
    yardstick = kit.LearningQuoter(depth=config["depth"], exploration=1.0, penalty=config["penalty"],
                                   name="yardstick")
    curve = np.asarray(kit.train_quoters(learners + [yardstick], config["sessions"], config["session_length"],
                                         rng, taker_rate=config["taker_rate"]), dtype=float)
    # Group the sessions into blocks that grow longer and longer, so the early sessions can be seen.
    edges = np.unique(np.round(np.geomspace(1, config["sessions"], config["points"])).astype(int))
    starts = np.concatenate([[0], edges[:-1]])
    learned = np.array([curve[a:b, :-1].mean() for a, b in zip(starts, edges)])
    random_one = np.array([curve[a:b, -1].mean() for a, b in zip(starts, edges)])
    # A second yardstick, in a market of its own: quoters that always quote one tick from the midpoint
    # on both sides.  The random quoter trades there too.
    fixed = _fixed_quoters(config, config["quoters"])
    random_two = kit.LearningQuoter(depth=config["depth"], exploration=1.0, penalty=config["penalty"],
                                    name="yardstick")
    fixed_curve = np.asarray(kit.train_quoters(fixed + [random_two], config["fixed_sessions"],
                                               config["session_length"], rng, taker_rate=config["taker_rate"]),
                             dtype=float)
    return {"steps": edges * config["session_length"], "learners": learned, "yardstick": random_one,
            "first": learned[0], "last": learned[-1], "yardstick_last": random_one[-1],
            "fixed": fixed_curve[:, :-1].mean()}


@figure("study3.learning_curve")
def learning_curve_figure(results) -> str:
    steps = results.runs[0]["steps"]
    learned, random_one = results.stack("learners"), results.stack("yardstick")
    series = [{"x": steps, "y": run["learners"], "colour": figures.MUTED} for run in results.runs]
    series[0]["label"] = "learners, each seed"
    series.append({"x": steps, "y": learned.mean(axis=0), "label": "learners, average", "colour": figures.LIME,
                   "points": True})
    series.append({"x": steps, "y": random_one.mean(axis=0), "label": "random quoter", "colour": figures.ORANGE,
                   "points": True, "dashed": True})
    fixed = float(results.stack("fixed").mean())
    series.append({"x": steps, "y": np.full(len(steps), fixed), "label": "fixed at one tick",
                   "colour": figures.WHITE, "dashed": True})
    note = (f"last block: learners {learned[:, -1].mean():+.3f}, random {random_one[:, -1].mean():+.3f}, "
            f"fixed {fixed:+.3f}")
    return figures.line_figure(series, "Quoters learning together", "training steps so far",
                               "average reward per step in ticks", log_x=True, zero_line=True, note=note)


# -- step 5: impact with learning liquidity providers ---------------------------------

def _random_market(config: dict, seed: int):
    """A market whose liquidity comes from zero-intelligence traders, warmed up."""
    traders = [kit.RandomTrader(limit_rate=config["random_limit_rate"], market_rate=config["random_market_rate"],
                                cancel_probability=config["random_cancel_probability"],
                                band=config["random_band"], size=1) for _ in range(config["random_traders"])]
    market = Marketplace(traders, seed=seed, engine=kit.new_engine(keep_trades=False))
    market.run(config["random_warm_up"])
    return market


@experiment("study3.impact_comparison")
def impact_comparison(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sizes = [int(size) for size in config["sizes"]]
    # Trained quoters.  Every large order meets a fresh market in which the quoters hold nothing.
    quoters = _quoters(config, config["quoters"], config["depth"])
    kit.train_quoters(quoters, config["training_sessions"], config["session_length"], rng,
                      taker_rate=config["taker_rate"])
    quoter_sizes, quoter_impacts, emptied = [], [], 0
    for _ in range(config["orders_per_size"]):
        for size in sizes:
            market = kit.quoter_market(quoters, int(rng.integers(0, 2 ** 31)), taker_rate=config["taker_rate"],
                                       warm_up=config["warm_up"])
            done, impacts = kit.measure_impact(market, [size], 1, 0, rng)
            quoter_sizes.extend(np.asarray(done, dtype=float))
            quoter_impacts.extend(np.asarray(impacts, dtype=float))
            emptied += market.empty_side_steps > 0
    # Random liquidity providers, measured in exactly the same way: a fresh market for every large order.
    random_sizes, random_impacts, random_emptied = [], [], 0
    for _ in range(config["orders_per_size"]):
        for size in sizes:
            market = _random_market(config, int(rng.integers(0, 2 ** 31)))
            before = market.empty_side_steps
            done, impacts = kit.measure_impact(market, [size], 1, 0, rng)
            random_sizes.extend(np.asarray(done, dtype=float))
            random_impacts.extend(np.asarray(impacts, dtype=float))
            random_emptied += market.empty_side_steps > before
    orders = len(sizes) * config["orders_per_size"]
    return {"quoter_sizes": np.asarray(quoter_sizes), "quoter_impacts": np.asarray(quoter_impacts),
            "random_sizes": np.asarray(random_sizes), "random_impacts": np.asarray(random_impacts),
            "quoter_emptied_fraction": emptied / orders, "random_emptied_fraction": random_emptied / orders}


def impact_summary(results, which: str) -> dict:
    """The pooled impact curve of one market with its fitted exponent and interval (None where no fit is possible)."""
    sizes, impacts = results.concatenate(f"{which}_sizes"), results.concatenate(f"{which}_impacts")
    curve = kit.impact_curve(sizes, impacts)
    summary = {"curve": curve, "exponent": None, "prefactor": None, "low": None, "high": None}
    try:
        summary["exponent"], summary["prefactor"] = kit.fit_exponent(sizes, impacts)
        rng = np.random.default_rng(int(sum(results.seeds)) + (1 if which == "quoter" else 2))
        summary["low"], summary["high"] = kit.bootstrap_interval(sizes, impacts, rng,
                                                                 resamples=results.config["resamples"])
    except (ValueError, IndexError):
        pass                                 # too few sizes with a positive average impact to fit a power law
    return summary


@figure("study3.impact_comparison")
def impact_comparison_figure(results) -> str:
    series, strip = [], []
    for which, name, colour in (("quoter", "quoters", figures.LIME), ("random", "random", figures.BLUE)):
        summary = impact_summary(results, which)
        seeds = []
        for run in results.runs:                 # each seed's own exponent, beside the pooled one
            try:
                seeds.append(float(kit.fit_exponent(run[f"{which}_sizes"], run[f"{which}_impacts"])[0]))
            except (ValueError, IndexError):
                seeds.append(None)
        strip.append({"label": "trained quoters" if which == "quoter" else "random providers", "colour": colour,
                      "pooled": summary["exponent"], "low": summary["low"], "high": summary["high"],
                      "seeds": seeds})
        curve = summary["curve"]
        label = f"{name}: no fit"
        if summary["exponent"] is not None:
            label = f"{name} {summary['exponent']:.2f}"
            if summary["low"] is not None:
                label += f" ({summary['low']:.2f} to {summary['high']:.2f})"
            fitted = summary["prefactor"] * curve["size"] ** summary["exponent"]
            series.append({"x": curve["size"], "y": fitted, "colour": colour, "dashed": True})
        series.append({"x": curve["size"], "y": curve["impact"], "error": curve["error"], "label": label,
                       "colour": colour, "points": True, "line": False})
        if which == "random" and np.any(curve["impact"] > 0):
            # A guide with the slope of the square-root law, drawn through the smallest size.
            first = int(np.argmax(curve["impact"] > 0))
            guide = curve["impact"][first] * (curve["size"] / curve["size"][first]) ** 0.5
            series.append({"x": curve["size"], "y": guide, "colour": figures.MUTED, "label": "slope one half"})
    return figures.line_figure(series, "Impact with learning liquidity providers", "order size in lots",
                               "impact in ticks", log_x=True, log_y=True,
                               note="fitted exponent (95% interval)", strip=strip)


# -- steps 6 and 7: volatility and collapse against agents and depth ----------------------

@experiment("study3.agents_and_depth")
def agents_and_depth(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    counts, depths = list(config["quoter_counts"]), list(config["depths"])
    volatility = np.zeros((len(depths), len(counts)))
    collapse = np.zeros((len(depths), len(counts)))
    for row, depth in enumerate(depths):
        for column, count in enumerate(counts):
            quoters = _quoters(config, count, depth)
            kit.train_quoters(quoters, config["training_sessions"], config["session_length"], rng,
                              taker_rate=config["taker_rate"])
            volatility[row, column], collapse[row, column] = kit.volatility_and_collapse(
                quoters, config["runs"], config["run_length"], rng, taker_rate=config["taker_rate"])
    return {"volatility": volatility, "collapse": collapse, "counts": np.array(counts, dtype=float),
            "depths": np.array(depths, dtype=float)}


@figure("study3.volatility_map")
def volatility_map_figure(results) -> str:
    grid = results.mean("volatility")
    run = results.runs[0]
    return figures.heatmap_figure(grid, run["counts"], run["depths"], "Volatility against agents and depth",
                                  "number of quoters", "depth in ticks",
                                  note=f"{grid.min():.2f} to {grid.max():.2f} ticks per step",
                                  legend=[("brighter is more volatile", figures.BLUE)])


@figure("study3.collapse_map")
def collapse_map_figure(results) -> str:
    grid = results.mean("collapse")
    run = results.runs[0]
    return figures.heatmap_figure(grid, run["counts"], run["depths"], "How often one side of the book emptied",
                                  "number of quoters", "depth in ticks",
                                  note=f"fraction of runs, {grid.min():.2f} to {grid.max():.2f}",
                                  legend=[("brighter is more often", figures.BLUE)])


def regime_grid(results) -> list:
    """The regime of every cell, as a grid of names, from the averages over the seeds."""
    volatility, collapse = results.mean("volatility"), results.mean("collapse")
    config = results.config
    return [[kit.classify_regime(float(volatility[r, c]), float(collapse[r, c]), config["collapse_threshold"],
                                 config["frozen_threshold"]) for c in range(volatility.shape[1])]
            for r in range(volatility.shape[0])]


@figure("study3.regime_map")
def regime_map_figure(results) -> str:
    names = regime_grid(results)
    for row in names:
        for name in row:
            if name not in REGIME_CODES:
                raise ValueError(f"classify_regime returned {name!r}, which is not one of the three regimes.")
    grid = np.array([[REGIME_CODES[name] for name in row] for row in names])
    run = results.runs[0]
    return figures.heatmap_figure(grid, run["counts"], run["depths"], "The regime of each cell",
                                  "number of quoters", "depth in ticks",
                                  legend=[(name, REGIME_COLOURS[name]) for name in ("continuous", "collapsed", "frozen")],
                                  colours={REGIME_CODES[name]: colour for name, colour in REGIME_COLOURS.items()})
