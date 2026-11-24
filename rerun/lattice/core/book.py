"""The limit order book and the small functions that operate on it.

Prices are whole numbers of ticks, so 10000 means 100.00 when one tick is 0.01.
Each price holds a queue: a plain list of orders, with the earliest order at
index 0.  The functions below are the "parts" of the matching engine.  The
engine can be assembled from these defaults or from the learner's own versions,
which is how the live order book comes to run on the learner's code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

BUY = "buy"
SELL = "sell"


def opposite(side: str) -> str:
    return SELL if side == BUY else BUY


@dataclass(eq=False)
class Order:
    """One resting limit order.  Two orders are the same only if they are the same object."""

    id: int
    side: str
    price: int
    size: int
    owner: str = ""

    def __repr__(self) -> str:
        return f"Order(id={self.id}, {self.side} {self.size} at {self.price})"


class Fill(NamedTuple):
    """A resting order traded: which order, at what price, and how much."""

    order_id: int
    price: int
    size: int


class Book:
    """Buy orders (bids) and sell orders (asks), each grouped by price into queues."""

    def __init__(self) -> None:
        self.bids: dict[int, list[Order]] = {}
        self.asks: dict[int, list[Order]] = {}

    def levels(self, side: str) -> dict[int, list[Order]]:
        return self.bids if side == BUY else self.asks

    def volume(self, side: str) -> int:
        return sum(order.size for queue in self.levels(side).values() for order in queue)

    def __repr__(self) -> str:
        return f"Book({len(self.bids)} bid prices, {len(self.asks)} ask prices)"


# ---------------------------------------------------------------------------
# The parts.  Study 1 has the learner write each of these.
# ---------------------------------------------------------------------------


PART_NAMES = (
    "rest_order",
    "best_bid",
    "best_ask",
    "fill_from_front",
    "match_market_order",
    "orders_ahead",
    "cancel_order",
)


