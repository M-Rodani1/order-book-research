"""Traders and the market that runs them.

A trader is any object with one method, ``decide(market)``, that returns a list
of actions: :class:`Limit`, :class:`Market` (a market order) or :class:`Cancel`.
The :class:`Marketplace` asks each trader in turn, once per step, and sends the
actions to the matching engine.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .book import BUY, SELL
from .engine import MatchingEngine


class Limit(NamedTuple):
    side: str
    price: int
    size: int


class Market(NamedTuple):
    side: str
    size: int


class Cancel(NamedTuple):
    order_id: int


class Trader:
    """Base class.  Subclasses override :meth:`decide`."""

    name = "trader"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        self.rng: np.random.Generator = np.random.default_rng(0)

    def decide(self, market: "Marketplace") -> list:
        return []


class Marketplace:
    """Runs traders against one matching engine, one step at a time."""

    def __init__(self, traders, seed: int = 0, engine: MatchingEngine | None = None,
                 reference_price: int = 10000, shuffle: bool = False) -> None:
        self.engine = engine if engine is not None else MatchingEngine()
        self.traders = list(traders)
        self.time = 0
        self.reference_price = reference_price
        self.shuffle = shuffle
        self.mids: list[float] = []
        self._last_bid = reference_price - 1
        self._last_ask = reference_price + 1
        self.empty_side_steps = 0
        self.last_trades: list = []
        root = np.random.SeedSequence(seed)
        children = root.spawn(len(self.traders) + 1)
        self.rng = np.random.default_rng(children[0])
        names = set()
        for number, (trader, child) in enumerate(zip(self.traders, children[1:])):
            trader.rng = np.random.default_rng(child)
            if trader.name in names:
                trader.name = f"{trader.name}-{number}"
            names.add(trader.name)

    def add_trader(self, trader) -> None:
        """Bring a trader into a market that is already running.

        Its random generator is seeded from the market's own, so the run stays reproducible.
        """
        trader.rng = np.random.default_rng(int(self.rng.integers(0, 2 ** 63 - 1)))
        names = {existing.name for existing in self.traders}
        if trader.name in names:
            trader.name = f"{trader.name}-{len(self.traders)}-{self.time}"
        self.traders.append(trader)

    def remove_trader(self, trader, cancel_orders: bool = True) -> None:
        """Take a trader out of the market, cancelling whatever it left resting."""
        if cancel_orders:
            for order in self.engine.orders_of(trader.name):
                self.engine.cancel(order.id)
        self.traders = [existing for existing in self.traders if existing is not trader]

    # -- what a trader may look at -----------------------------------------

    @property
    def best_bid(self):
        return self.engine.best_bid()

    @property
    def best_ask(self):
        return self.engine.best_ask()

    @property
    def spread(self):
        return self.engine.spread()

    @property
    def mid(self) -> float:
        """The midpoint, using the last seen price for a side that is empty."""
        bid, ask = self.engine.best_bid(), self.engine.best_ask()
        if bid is not None:
            self._last_bid = bid
        if ask is not None:
            self._last_ask = ask
        low = bid if bid is not None else min(self._last_bid, self._last_ask - 1)
        high = ask if ask is not None else max(self._last_ask, low + 1)
        return (low + high) / 2

    def depth(self, side: str, levels: int | None = None):
        return self.engine.depth(side, levels)

    def my_orders(self, trader) -> list:
        return self.engine.orders_of(trader.name)

    def my_fills(self, trader) -> list:
        """Trades against this trader's resting orders since it last asked."""
        return self.engine.take_inbox(trader.name)

    # -- running ------------------------------------------------------------

    def apply(self, trader, action) -> list:
        if isinstance(action, Limit):
            _, trades = self.engine.submit_limit(action.side, int(action.price), int(action.size), trader.name)
            return trades
        if isinstance(action, Market):
            return self.engine.submit_market(action.side, int(action.size), trader.name)
        if isinstance(action, Cancel):
            self.engine.cancel(action.order_id)
            return []
        raise TypeError(f"{trader.name} returned {action!r}, which is not a Limit, Market or Cancel.")

    def step(self) -> float:
        self.time += 1
        self.last_trades = []
        order = self.traders
        if self.shuffle:
            order = [self.traders[i] for i in self.rng.permutation(len(self.traders))]
        for trader in order:
            for action in trader.decide(self) or ():
                self.last_trades.extend(self.apply(trader, action))
        if self.engine.best_bid() is None or self.engine.best_ask() is None:
            self.empty_side_steps += 1
        mid = self.mid
        self.mids.append(mid)
        return mid

    def run(self, steps: int, on_step=None) -> np.ndarray:
        for _ in range(steps):
            self.step()
            if on_step is not None:
                on_step(self)
        return np.asarray(self.mids)


