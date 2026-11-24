"""Study 1 experiments and figures.  Each runs on the kit, so on the learner's code once it has passed."""

from __future__ import annotations

import numpy as np

from lattice import HOME, kit
from lattice.core import figures, hawkes, moallemi_yuan
from lattice.core.queue_model import QueueModel, value_grid
from lattice.core.runner import experiment, figure

from .flow import RandomFlow, depth_at_distance

NEVER = np.nan


def _model(config: dict, **changes) -> QueueModel:
    return QueueModel(**{**config.get("model", {}), **changes})


# -- step 5: the shape of a random book ---------------------------------------

def _profile(engine, step, events: int, warm_up: int, sample_every: int, levels: int) -> dict:
    bid_total = np.zeros(levels)
    ask_total = np.zeros(levels)
    spreads = []
    for number in range(events):
        step(number)
        if number >= warm_up and number % sample_every == 0:
            sample = depth_at_distance(engine, levels)
            if sample is not None:
                bid_total += sample[0]
                ask_total += sample[1]
                spreads.append(sample[2])
    samples = max(len(spreads), 1)
    return {"bid_depth": bid_total / samples, "ask_depth": ask_total / samples,
            "mean_spread": float(np.mean(spreads)) if spreads else np.nan, "samples": len(spreads)}


@experiment("study1.book_shape")
def book_shape(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    engine = kit.new_engine(keep_trades=False)
    flow = RandomFlow(engine, rng, kit.next_arrival, **config.get("flow", {}))
    flow.seed_book()
    result = _profile(engine, lambda n: flow.step(), config["events"], config["warm_up"],
                      config["sample_every"], config["levels"])
    result["traded_volume"] = engine.traded_volume
    return result


@figure("study1.book_shape")
def book_shape_figure(results) -> str:
    distance = np.arange(results.runs[0]["bid_depth"].shape[0])
    bids, asks = results.stack("bid_depth"), results.stack("ask_depth")
    spread = float(results.stack("mean_spread").mean())
    series = [
        {"x": distance, "y": bids.mean(axis=0), "error": bids.std(axis=0), "label": "buyers",
         "colour": figures.BLUE, "points": True},
        {"x": distance, "y": asks.mean(axis=0), "error": asks.std(axis=0), "label": "sellers",
         "colour": figures.ORANGE, "points": True},
    ]
    return figures.line_figure(series, "The shape of a random book", "ticks behind the best price",
                               "average size resting", note=f"mean spread {spread:.2f} ticks")


# -- step 6: exact values ----------------------------------------------------------

@experiment("study1.exact_values")
def exact_values(config: dict, seed: int) -> dict:
    model = _model(config)
    values = kit.solve_values(model)
    return {"grid": value_grid(model, values), "premiums": model.premiums,
            "positions": np.arange(model.max_position + 1)}


@figure("study1.exact_values")
def exact_values_figure(results) -> str:
    run = results.runs[0]
    return figures.heatmap_figure(run["grid"], run["positions"], run["premiums"],
                                  "The value of a place in the queue, solved exactly",
                                  "lots ahead of you", "premium in ticks",
                                  note=f"largest value {run['grid'].max():.2f} ticks")


# -- step 7: learning by trial -----------------------------------------------------

def _values_from_q(q: dict) -> dict:
    return {state: max(actions.values()) for state, actions in q.items()}


def _learn(model: QueueModel, schedule, rng, exact: dict | None = None):
    q = None
    done = 0
    episodes, errors, agreement = [], [], []
    for count, learning_rate, exploration in schedule:
        q = kit.learn_values(model, int(count), rng, learning_rate=learning_rate, exploration=exploration, q=q)
        done += int(count)
        if exact is not None:
            learned = _values_from_q(q)
            episodes.append(done)
            errors.append(float(np.mean([abs(learned[s] - exact[s]) for s in model.states])))
            agreement.append(float(np.mean([(learned[s] > 1e-9) == (exact[s] > 1e-9) for s in model.states])))
    return q, np.array(episodes), np.array(errors), np.array(agreement)


@experiment("study1.learned_values")
def learned_values(config: dict, seed: int) -> dict:
    model = _model(config)
    exact = kit.solve_values(model)
    q, episodes, errors, agreement = _learn(model, config["schedule"], np.random.default_rng(seed), exact)
    return {"episodes": episodes, "error": errors, "agreement": agreement,
            "grid": value_grid(model, _values_from_q(q)), "exact": value_grid(model, exact)}


@figure("study1.learned_values")
def learned_values_figure(results) -> str:
    episodes = results.runs[0]["episodes"]
    errors = results.stack("error")
    agreement = float(results.stack("agreement")[:, -1].mean())
    series = [{"x": episodes, "y": run["error"], "colour": figures.MUTED} for run in results.runs]
    series.append({"x": episodes, "y": errors.mean(axis=0), "label": "average over seeds",
                   "colour": figures.LIME, "points": True})
    series[0]["label"] = "each seed"
    return figures.line_figure(series, "Learning the value by trial", "episodes of trial so far",
                               "average error in ticks", log_x=True, log_y=True,
                               note=f"same choice as the exact answer in {100 * agreement:.0f}% of states")


# -- step 8: the cancel boundary and the paper's model ----------------------------

def _boundary_array(model: QueueModel, values: dict) -> np.ndarray:
    boundary = kit.cancel_boundary(model, values)
    return np.array([model.max_position + 1 if b is None else b for b in boundary], dtype=float)


@experiment("study1.boundary")
def boundary(config: dict, seed: int) -> dict:
    model = _model(config)
    exact = kit.solve_values(model)
    q, _, _, _ = _learn(model, config["schedule"], np.random.default_rng(seed))
    alpha, beta = moallemi_yuan.solve_lattice(model)
    linear = moallemi_yuan.lattice_value(model, alpha, beta)
    paper = np.array([next((k for k in range(model.max_position + 1) if linear[i, k] <= 1e-9),
                           model.max_position + 1) for i in range(model.premium_count)], dtype=float)
    return {"grid": value_grid(model, exact), "premiums": model.premiums,
            "positions": np.arange(model.max_position + 1),
            "exact_boundary": _boundary_array(model, exact),
            "learned_boundary": _boundary_array(model, _values_from_q(q)),
            "paper_boundary": paper}


@figure("study1.boundary")
def boundary_figure(results) -> str:
    run = results.runs[0]
    overlays = [{"x": run["exact_boundary"], "label": "cancel beyond here, exact", "colour": figures.LIME}]
    for number, each in enumerate(results.runs):
        overlays.append({"x": each["learned_boundary"], "colour": figures.WHITE, "dashed": True,
                         "label": "learned by trial" if number == 0 else ""})
    overlays.append({"x": run["paper_boundary"], "colour": figures.MUTED,
                     "label": "value zero with no choice to cancel"})
    return figures.heatmap_figure(run["grid"], run["positions"], run["premiums"], "The cancel boundary",
                                  "lots ahead of you", "premium in ticks", overlays=overlays)


@experiment("study1.moallemi_yuan")
def moallemi_yuan_comparison(config: dict, seed: int) -> dict:
    # A wide premium grid, so that its edges are too far away to matter at the premium of interest.
    wide = {"premium_lowest": config["grid_lowest"], "premium_count": config["grid_count"]}
    model = _model(config, **wide)
    no_cancel = _model(config, allow_cancel=False, **wide)
    index = int(np.argmin(np.abs(model.premiums - config["premium"])))
    premium = float(model.premiums[index])
    with_choice = value_grid(model, kit.solve_values(model))[index]
    without_choice = value_grid(no_cancel, kit.solve_values(no_cancel))[index]
    alpha, beta = moallemi_yuan.solve_lattice(no_cancel)
    q, alpha_c, beta_c = moallemi_yuan.continuous_from_model(no_cancel, points=401)
    return {"positions": np.arange(model.max_position + 1), "premium": premium,
            "with_choice": with_choice, "without_choice": without_choice,
            "paper_lattice": alpha * premium - beta, "alpha": alpha, "beta": beta,
            "continuous_positions": q, "paper_continuous": alpha_c * premium - beta_c}


@figure("study1.moallemi_yuan")
def moallemi_yuan_figure(results) -> str:
    run = results.runs[0]
    series = [
        {"x": run["continuous_positions"], "y": run["paper_continuous"], "colour": figures.MUTED,
         "dashed": True, "label": "paper, continuous queue"},
        {"x": run["positions"], "y": run["paper_lattice"], "colour": figures.BLUE,
         "label": "paper, whole lots"},
        {"x": run["positions"], "y": run["without_choice"], "colour": figures.WHITE, "points": True,
         "line": False, "label": "yours, no choice to cancel"},
        {"x": run["positions"], "y": run["with_choice"], "colour": figures.LIME, "points": True,
         "label": "yours, free to cancel"},
    ]
    return figures.line_figure(series, "Your values against Moallemi and Yuan", "lots ahead of you",
                               "value in ticks", zero_line=True,
                               note=f"premium {float(run['premium']):.2f} ticks")


# -- step 10 (optional): Hawkes-driven arrivals --------------------------------------

@experiment("study1.hawkes_book_shape")
def hawkes_book_shape(config: dict, seed: int) -> dict:
    parameters = hawkes.load_parameters(HOME / config["parameter_file"])
    kinds = list(parameters.event_types)
    stationary = np.linalg.solve(np.eye(len(kinds)) - parameters.excitation / parameters.decay,
                                 parameters.baseline)
    horizon = config["events"] / float(stationary.sum())
    output = {}
    # The same long-run rates, once clustered (Hawkes) and once evenly random (Poisson).
    for label in ("hawkes", "poisson"):
        rng = np.random.default_rng(seed)
        engine = kit.new_engine(keep_trades=False)
        flow = RandomFlow(engine, rng, kit.next_arrival)
        flow.seed_book()
        if label == "hawkes":
            arrivals = [kinds[k] for _, k in kit.hawkes_arrivals(parameters, horizon, rng)]
        else:
            arrivals = list(rng.choice(kinds, size=config["events"], p=stationary / stationary.sum()))
        result = _profile(engine, lambda n: flow.apply(arrivals[n]), len(arrivals),
                          config["warm_up"], config["sample_every"], config["levels"])
        for key, value in result.items():
            output[f"{label}_{key}"] = value
    return output


@figure("study1.hawkes_book_shape")
def hawkes_book_shape_figure(results) -> str:
    distance = np.arange(results.runs[0]["hawkes_bid_depth"].shape[0])
    series = [
        {"x": distance, "y": results.mean("hawkes_bid_depth"), "label": "buyers, Hawkes", "colour": figures.BLUE,
         "points": True},
        {"x": distance, "y": results.mean("hawkes_ask_depth"), "label": "sellers, Hawkes",
         "colour": figures.ORANGE, "points": True},
        {"x": distance, "y": results.mean("poisson_bid_depth"), "label": "buyers, Poisson",
         "colour": figures.BLUE, "dashed": True},
        {"x": distance, "y": results.mean("poisson_ask_depth"), "label": "sellers, Poisson",
         "colour": figures.ORANGE, "dashed": True},
    ]
    note = (f"mean spread {float(results.mean('hawkes_mean_spread')):.2f} against "
            f"{float(results.mean('poisson_mean_spread')):.2f} ticks")
    return figures.line_figure(series, "Book shape with Hawkes arrivals", "ticks behind the best price",
                               "average size resting", note=note)
