"""The three markets of Study 2, built from the kit so that they run on the learner's traders."""

from __future__ import annotations

import numpy as np

from lattice import kit
from lattice.core import BUY, SELL, Marketplace

# Four random traders who both provide liquidity (limit orders that live ten steps on average)
# and take it (market orders, one every two steps between them).
RANDOM_TRADERS = {"count": 4, "limit_rate": 2.0, "market_rate": 0.125, "cancel_probability": 0.1,
                  "band": 10, "size": 1}

# The same four traders sending the same market orders and no limit orders at all.
TAKERS_ONLY = {**RANDOM_TRADERS, "limit_rate": 0.0}

# One provider of slow liquidity: a unit of volume lives a thousand steps on average.
HIDDEN_LIQUIDITY = {"refill_probability": 0.1, "cancel_probability": 0.001, "reveal_distance": 3,
                    "span": 40, "size": 1}

RANDOM_MARKET = {"random_traders": RANDOM_TRADERS, "hidden_liquidity": None, "warm_up": 1000}
HIDDEN_MARKET = {"random_traders": TAKERS_ONLY, "hidden_liquidity": HIDDEN_LIQUIDITY, "warm_up": 6000}
# The market in between: the random traders keep their limit orders, and the hidden liquidity is added.
BOTH_MARKET = {"random_traders": RANDOM_TRADERS, "hidden_liquidity": HIDDEN_LIQUIDITY, "warm_up": 6000}


def build_market(spec: dict, seed: int, extra=(), warm_up: bool = True):
    """A market described by ``spec``, warmed up.  Returns ``(market, hidden trader or None)``."""
    random_spec = dict(spec["random_traders"])
    count = int(random_spec.pop("count"))
    traders = [kit.RandomTrader(name=f"random-{number + 1}", **random_spec) for number in range(count)]
    hidden = None
    if spec.get("hidden_liquidity"):
        hidden = kit.HiddenLiquidityTrader(**spec["hidden_liquidity"])
        traders.append(hidden)
    traders.extend(extra)
    market = Marketplace(traders, seed=seed, engine=kit.new_engine(keep_trades=False))
    if warm_up:
        market.run(int(spec["warm_up"]))
    return market, hidden


def measure_rng(seed: int) -> np.random.Generator:
    """The generator that orders and signs the large orders: separate from the traders' own."""
    return np.random.default_rng([int(seed), 2])


def depth_by_distance(market, hidden, levels: int):
    """Shown and hidden volume at 0, 1, 2 ... ticks behind the best price, averaged over the two sides.

    Returns ``(shown, hidden)`` or None when a side of the book is empty.
    """
    bid, ask = market.best_bid, market.best_ask
    if bid is None or ask is None:
        return None
    shown = np.zeros(levels)
    unseen = np.zeros(levels)
    for side, best in ((BUY, bid), (SELL, ask)):
        for price, size, _ in market.depth(side):
            distance = abs(price - best)
            if distance < levels:
                shown[distance] += size
        if hidden is not None:
            for price, size in hidden.hidden_levels(side):
                distance = abs(price - best)
                if distance < levels:
                    unseen[distance] += size
    return shown / 2, unseen / 2
