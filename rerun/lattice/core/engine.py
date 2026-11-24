"""A price-time priority matching engine.

Price priority: a better price always trades first.  Time priority: at one
price, the earliest order trades first.  The engine does the bookkeeping
(order identifiers, who owns what, the trade record) and leaves the book
operations to the parts in :mod:`lattice.core.book`.
"""

from __future__ import annotations

from typing import NamedTuple

from .book import BUY, SELL, Book, Order, opposite


class Trade(NamedTuple):
    """One execution between a resting order (the maker) and an incoming order (the taker)."""

    sequence: int
    price: int
    size: int
    taker_side: str
    maker_owner: str
    taker_owner: str
    maker_order_id: int


class BookError(AssertionError):
    """The book broke one of its invariants."""


class MatchingEngine:
    def __init__(self, parts=None, keep_trades: bool = True) -> None:
        self.book = Book()
        if parts is None:
            raise ValueError("The engine needs its book operations, which come from the "
                             "published study code.")
        self.parts = parts
        self.index: dict[int, Order] = {}
        self.owned: dict[str, dict[int, Order]] = {}
        self.trades: list[Trade] = []
        self.keep_trades = keep_trades
        self.inbox: dict[str, list[Trade]] = {}
        self.sequence = 0
        self.next_id = 1
        self.last_trade_price = None
        self.traded_volume = 0

    # -- reading the book ---------------------------------------------------

    def best_bid(self):
        return self.parts.best_bid(self.book)

    def best_ask(self):
        return self.parts.best_ask(self.book)

    def spread(self):
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else ask - bid

    def mid(self):
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else (bid + ask) / 2

    def depth(self, side: str, levels: int | None = None):
        """(price, total size, number of orders) for each price, best price first."""
        book_levels = self.book.levels(side)
        prices = sorted(book_levels, reverse=(side == BUY))
        if levels is not None:
            prices = prices[:levels]
        return [(p, sum(o.size for o in book_levels[p]), len(book_levels[p])) for p in prices]

    def orders_of(self, owner: str) -> list[Order]:
        return list(self.owned.get(owner, {}).values())

    def orders_ahead(self, order: Order):
        return self.parts.orders_ahead(self.book, order)

    # -- changing the book --------------------------------------------------

    def submit_market(self, side: str, size: int, owner: str = "") -> list[Trade]:
        """Trade immediately against the other side, as much as is available."""
        if size <= 0:
            return []
        fills = self.parts.match_market_order(self.book, side, size)
        return self._record(fills, side, owner)

    def submit_limit(self, side: str, price: int, size: int, owner: str = ""):
        """Trade whatever crosses the limit price, then rest the remainder.

        Returns ``(order, trades)``; ``order`` is None when nothing was left to rest.
        """
        if size <= 0:
            return None, []
        trades: list[Trade] = []
        resting_side = self.book.levels(opposite(side))
        while size > 0:
            best = self.best_ask() if side == BUY else self.best_bid()
            if best is None or (best > price if side == BUY else best < price):
                break
            available = sum(order.size for order in resting_side[best])
            take = min(size, available)
            new = self._record(self.parts.match_market_order(self.book, side, take), side, owner)
            traded = sum(trade.size for trade in new)
            if traded == 0:
                break
            trades.extend(new)
            size -= traded
        if size <= 0:
            return None, trades
        order = Order(self.next_id, side, price, size, owner)
        self.next_id += 1
        self.parts.rest_order(self.book, order)
        self.index[order.id] = order
        self.owned.setdefault(owner, {})[order.id] = order
        return order, trades

    def cancel(self, order_id: int) -> bool:
        order = self.index.get(order_id)
        if order is None:
            return False
        removed = bool(self.parts.cancel_order(self.book, order))
        if removed:
            self._forget(order)
        return removed

    # -- bookkeeping --------------------------------------------------------

    def _forget(self, order: Order) -> None:
        self.index.pop(order.id, None)
        owned = self.owned.get(order.owner)
        if owned is not None:
            owned.pop(order.id, None)

    def _record(self, fills, taker_side: str, taker_owner: str) -> list[Trade]:
        trades = []
        for fill in fills:
            order_id, price, size = fill[0], fill[1], fill[2]
            maker = self.index.get(order_id)
            maker_owner = maker.owner if maker is not None else ""
            self.sequence += 1
            trade = Trade(self.sequence, price, size, taker_side, maker_owner, taker_owner, order_id)
            trades.append(trade)
            if self.keep_trades:
                self.trades.append(trade)
            self.inbox.setdefault(maker_owner, []).append(trade)
            self.last_trade_price = price
            self.traded_volume += size
            if maker is not None and maker.size <= 0:
                self._forget(maker)
        return trades

    def take_inbox(self, owner: str) -> list[Trade]:
        """Trades in which ``owner`` was the resting side since the last call."""
        return self.inbox.pop(owner, [])

    def check_invariants(self) -> None:
        """Raise :class:`BookError` if the book is in an impossible state."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            raise BookError(f"The best bid {bid} is not below the best ask {ask}.")
        seen = 0
        for side in (BUY, SELL):
            for price, queue in self.book.levels(side).items():
                if not queue:
                    raise BookError(f"An empty queue was left at price {price}.")
                for order in queue:
                    seen += 1
                    if order.size <= 0:
                        raise BookError(f"{order} is resting with no size left.")
                    if order.price != price or order.side != side:
                        raise BookError(f"{order} is resting in the wrong queue.")
                    if self.index.get(order.id) is not order:
                        raise BookError(f"{order} is in the book but not in the index.")
        if seen != len(self.index):
            raise BookError("The index lists orders that are no longer in the book.")
