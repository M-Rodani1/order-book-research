"""A small model of one buy order waiting in the queue at the best bid.

This is the setting of Moallemi and Yuan (2017), put on a grid so that it can
be solved exactly by dynamic programming and learned by tabular Q-learning.

State: ``(position, premium_index)``.

* ``position`` is the number of lots ahead of the order in its queue.
* ``premium_index`` points into :attr:`QueueModel.premiums`.  The premium is the
  efficient price minus the order's price, in ticks: what the order earns, on
  paper, if it were filled right now.

Each step the holder first chooses ``"stay"`` or ``"cancel"``.  Cancelling ends
the episode with reward zero.  Staying lets exactly one event happen:

* a market sell of ``u`` lots arrives: the premium falls by ``impact * u``; if
  ``u`` is larger than the lots ahead the order is filled and the reward is the
  new premium, otherwise the order moves up by ``u``;
* a market buy of ``u`` lots arrives on the other side: the premium rises by
  ``impact * u`` and the position is unchanged;
* the price jumps down through the order: it is filled and the reward is the
  premium minus ``jump_size``;
* the price jumps up and away: the order is cancelled, reward zero;
* a cancellation event: each lot ahead stays with probability ``keep_fraction``.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from math import comb

import numpy as np

STAY = "stay"
CANCEL = "cancel"


@dataclass(frozen=True)
class QueueModel:
    max_position: int = 25
    premium_lowest: float = 0.0
    premium_count: int = 31
    impact: float = 0.04
    sell_probability: float = 0.20
    buy_probability: float = 0.20
    jump_down_probability: float = 0.10
    jump_up_probability: float = 0.10
    cancellation_probability: float = 0.40
    mean_trade_size: float = 3.0
    max_trade_size: int = 15
    jump_size: float = 1.0
    keep_fraction: float = 0.85
    allow_cancel: bool = True

    def __post_init__(self) -> None:
        total = (self.sell_probability + self.buy_probability + self.jump_down_probability
                 + self.jump_up_probability + self.cancellation_probability)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"The five event probabilities must sum to 1, not {total}.")

    # -- the grid -----------------------------------------------------------

    @cached_property
    def premiums(self) -> np.ndarray:
        return self.premium_lowest + self.impact * np.arange(self.premium_count)

    @cached_property
    def states(self) -> list:
        return [(k, i) for k in range(self.max_position + 1) for i in range(self.premium_count)]

    def actions(self, state=None) -> tuple:
        return (STAY, CANCEL) if self.allow_cancel else (STAY,)

    def premium(self, state) -> float:
        return float(self.premiums[state[1]])

    @cached_property
    def trade_sizes(self) -> np.ndarray:
        return np.arange(1, self.max_trade_size + 1)

    @cached_property
    def trade_size_probabilities(self) -> np.ndarray:
        """A geometric distribution with the chosen mean, cut off at ``max_trade_size``."""
        success = 1.0 / self.mean_trade_size
        weights = success * (1.0 - success) ** (self.trade_sizes - 1)
        return weights / weights.sum()

    def thinning_probabilities(self, position: int) -> np.ndarray:
        """Probability that ``j`` of ``position`` lots ahead remain after a cancellation event."""
        keep = self.keep_fraction
        return np.array([comb(position, j) * keep ** j * (1 - keep) ** (position - j)
                         for j in range(position + 1)])

    def _clamp(self, index: int) -> int:
        return min(max(index, 0), self.premium_count - 1)

    # -- exact dynamics, for dynamic programming ----------------------------

    def transitions(self, state, action) -> list:
        """Every outcome of ``action`` in ``state`` as ``(probability, reward, next_state)``.

        ``next_state`` is None when the episode ends.
        """
        return self._transitions[(state, action)] if (state, action) in self._transitions \
            else self._build(state, action)

    @cached_property
    def _transitions(self) -> dict:
        return {}

    def _build(self, state, action) -> list:
        position, index = state
        if action == CANCEL:
            outcomes = [(1.0, 0.0, None)]
            self._transitions[(state, action)] = outcomes
            return outcomes
        outcomes = []
        sizes, weights = self.trade_sizes, self.trade_size_probabilities
        for size, weight in zip(sizes, weights):
            size = int(size)
            probability = self.sell_probability * float(weight)
            new_index = self._clamp(index - size)
            if size > position:
                # Filled.  The reward uses the true premium, not the clamped grid value.
                outcomes.append((probability, float(self.premiums[index] - self.impact * size), None))
            else:
                outcomes.append((probability, 0.0, (position - size, new_index)))
        for size, weight in zip(sizes, weights):
            probability = self.buy_probability * float(weight)
            outcomes.append((probability, 0.0, (position, self._clamp(index + int(size)))))
        outcomes.append((self.jump_down_probability, float(self.premiums[index] - self.jump_size), None))
        outcomes.append((self.jump_up_probability, 0.0, None))
        for remaining, weight in enumerate(self.thinning_probabilities(position)):
            outcomes.append((self.cancellation_probability * float(weight), 0.0, (remaining, index)))
        self._transitions[(state, action)] = outcomes
        return outcomes

    # -- sampled dynamics, for learning by trial ----------------------------

    def sample(self, state, action, rng: np.random.Generator):
        """Draw one outcome: ``(reward, next_state)``, with ``next_state`` None at the end.

        Outcomes are drawn from exactly the probabilities :meth:`transitions` lists.
        """
        if action == CANCEL:
            return 0.0, None
        table = self._samplers.get(state)
        if table is None:
            outcomes = self.transitions(state, STAY)
            cumulative = np.cumsum([probability for probability, _, _ in outcomes]).tolist()
            cumulative[-1] = 1.0
            table = (cumulative, outcomes)
            self._samplers[state] = table
        cumulative, outcomes = table
        _, reward, next_state = outcomes[bisect_right(cumulative, rng.random())]
        return reward, next_state

    @cached_property
    def _samplers(self) -> dict:
        return {}

    def random_state(self, rng: np.random.Generator):
        return (int(rng.integers(0, self.max_position + 1)), int(rng.integers(0, self.premium_count)))

    def to_config(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def value_grid(model: QueueModel, values: dict) -> np.ndarray:
    """Values as an array with one row per premium and one column per queue position."""
    grid = np.zeros((model.premium_count, model.max_position + 1))
    for (position, index), value in values.items():
        grid[index, position] = value
    return grid
