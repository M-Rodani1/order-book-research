"""The random order flow used by Study 1: its live stage and its book experiments."""

from __future__ import annotations

import numpy as np

from lattice.core import BUY, SELL

DEFAULT_RATES = {"limit_buy": 1.0, "limit_sell": 1.0, "market_buy": 0.22, "market_sell": 0.22}
CANCEL_RATE_PER_ORDER = 0.012


class RandomFlow:
    """Limit orders, market orders and cancellations arriving at random.

    Arrivals are drawn by ``next_arrival(rng, rates)``, which is the piece the
    learner writes in step 5.  The cancellation rate grows with the number of
    resting orders, so the book settles around a steady size.
    """

    def __init__(self, engine, rng, next_arrival, rates=None, cancel_rate=CANCEL_RATE_PER_ORDER,
                 band: int = 6, lot: int = 10, reference_price: int = 10000, owner: str = "flow") -> None:
        self.engine = engine
        self.rng = rng
        self.next_arrival = next_arrival
        self.rates = dict(rates or DEFAULT_RATES)
        self.cancel_rate = cancel_rate
        self.band = band
        self.lot = lot
        self.owner = owner
        self.clock = 0.0
        self.last_bid = reference_price
        self.last_ask = reference_price + 1

    def seed_book(self, per_level: int = 4) -> None:
        """Start with a few orders at each of the nearest prices."""
        for distance in range(self.band):
            for _ in range(per_level):
                size = self.lot * int(self.rng.integers(1, 6))
                self.engine.submit_limit(BUY, self.last_bid - distance, size, self.owner)
                size = self.lot * int(self.rng.integers(1, 6))
                self.engine.submit_limit(SELL, self.last_ask + distance, size, self.owner)

    def _anchors(self):
        bid, ask = self.engine.best_bid(), self.engine.best_ask()
        if bid is not None:
            self.last_bid = bid
        if ask is not None:
            self.last_ask = ask
        if self.last_ask <= self.last_bid:
            self.last_ask = self.last_bid + 1
        return self.last_bid, self.last_ask

    def step(self) -> dict:
        """Draw the next arrival and apply it.  Returns a description of what happened."""
        resting = self.engine.orders_of(self.owner)
        rates = dict(self.rates)
        rates["cancel"] = self.cancel_rate * len(resting)
        wait, kind = self.next_arrival(self.rng, rates)
        self.clock += float(wait)
        return self.apply(kind, resting)

    def apply(self, kind: str, resting=None) -> dict:
        rng, engine = self.rng, self.engine
        bid, ask = self._anchors()
        event = {"kind": kind, "side": None, "price": None, "size": 0, "trades": [], "order": None,
                 "cancelled": None}
        if kind in ("limit_buy", "limit_sell"):
            side = BUY if kind == "limit_buy" else SELL
            distance = int(rng.integers(1, self.band + 1))
            price = ask - distance if side == BUY else bid + distance
            size = self.lot * int(rng.integers(1, 6))
            order, trades = engine.submit_limit(side, price, size, self.owner)
            event.update(side=side, price=price, size=size, order=order, trades=trades)
        elif kind in ("market_buy", "market_sell"):
            side = BUY if kind == "market_buy" else SELL
            size = self.lot * int(min(rng.geometric(0.45), 8))
            trades = engine.submit_market(side, size, self.owner)
            event.update(side=side, size=size, trades=trades)
        elif kind in ("cancel", "cancel_buy", "cancel_sell"):
            if resting is None:
                resting = engine.orders_of(self.owner)
            if kind != "cancel":
                wanted = BUY if kind == "cancel_buy" else SELL
                resting = [order for order in resting if order.side == wanted]
            if resting:
                order = resting[int(rng.integers(len(resting)))]
                event.update(side=order.side, price=order.price, size=order.size, cancelled=order)
                event["ahead_before"] = None
                engine.cancel(order.id)
        else:
            raise ValueError(f"next_arrival returned the kind {kind!r}, which is not one of the rates.")
        return event


def depth_at_distance(engine, levels: int):
    """Resting size at 0, 1, 2 ... ticks behind the best price on each side, or None if a side is empty."""
    bid, ask = engine.best_bid(), engine.best_ask()
    if bid is None or ask is None:
        return None
    bids, asks = engine.book.bids, engine.book.asks
    bid_depth = np.array([sum(o.size for o in bids.get(bid - d, ())) for d in range(levels)], dtype=float)
    ask_depth = np.array([sum(o.size for o in asks.get(ask + d, ())) for d in range(levels)], dtype=float)
    return bid_depth, ask_depth, ask - bid
