"""The queue position value model of Moallemi and Yuan.

C. C. Moallemi and K. Yuan, "A Model for Queue Position Valuation in a Limit
Order Book", working paper, revision of June 2017.

Their Theorem 1 says the value of an order at queue position ``q`` with
liquidity premium ``delta`` is linear in the premium,

    V(q, delta) = alpha(q) * delta - beta(q),

where ``alpha`` is the probability of being filled and ``beta / alpha`` is the
adverse selection cost.  Two solvers live here:

* :func:`solve_lattice` applies the same recursion (their Lemma 1) to the
  whole-lot :class:`~lattice.core.queue_model.QueueModel`, so that it can be
  compared number for number with dynamic programming on that model.
* :func:`solve_continuous` solves their integral equations (6) and (7) for a
  continuous queue with two-sided exponential trade sizes and a fixed
  cancellation proportion.

Neither solver gives the order holder a choice to cancel: as in the paper, the
order leaves only when it is filled or the price moves away.
"""

from __future__ import annotations

import numpy as np

from .queue_model import QueueModel


def solve_lattice(model: QueueModel):
    """Fill probability ``alpha[k]`` and cost ``beta[k]`` for ``k`` lots ahead."""
    sizes = model.trade_sizes
    weights = model.trade_size_probabilities
    mean_size = float((sizes * weights).sum())
    p_sell, p_buy = model.sell_probability, model.buy_probability
    p_down, p_cancel = model.jump_down_probability, model.cancellation_probability
    impact, jump = model.impact, model.jump_size
    count = model.max_position + 1
    alpha = np.zeros(count)
    beta = np.zeros(count)
    for k in range(count):
        thin = model.thinning_probabilities(k)
        own = 1.0 - p_buy - p_cancel * thin[k]
        a_sum = 0.0
        b_sum = 0.0
        for size, weight in zip(sizes, weights):
            size = int(size)
            if size <= k:
                a_sum += weight * alpha[k - size]
                b_sum += weight * (impact * size * alpha[k - size] + beta[k - size])
            else:
                a_sum += weight
                b_sum += weight * impact * size
        a_thin = float((thin[:k] * alpha[:k]).sum())
        b_thin = float((thin[:k] * beta[:k]).sum())
        alpha[k] = (p_sell * a_sum + p_down + p_cancel * a_thin) / own
        beta[k] = (p_sell * b_sum - p_buy * impact * mean_size * alpha[k]
                   + p_down * jump + p_cancel * b_thin) / own
    return alpha, beta


def lattice_value(model: QueueModel, alpha, beta) -> np.ndarray:
    """``alpha * premium - beta`` on the model's grid: rows are premiums, columns positions."""
    return np.outer(model.premiums, alpha) - beta[None, :]


def solve_continuous(trade_rate: float, jump_rate: float, cancellation_rate: float,
                     mean_trade_size: float, impact: float, jump_up_probability: float,
                     mean_adverse_jump: float, keep_fraction: float,
                     max_position: float, points: int = 801):
    """Solve equations (6) and (7) of the paper by marching up the queue.

    ``trade_rate`` counts trades on both sides.  Trade sizes are two-sided
    exponential with mean absolute size ``mean_trade_size``.  The paper writes
    the model for a sell order; ``jump_up_probability`` is the probability that
    a jump goes *through* the order (their ``p_J^+``) and ``mean_adverse_jump``
    is the mean size of such a jump.  Each cancellation keeps the fixed
    proportion ``keep_fraction`` of the queue ahead.

    Returns ``(positions, alpha, beta)``.
    """
    mu, gamma, eta = trade_rate, jump_rate, cancellation_rate
    p_u = 0.5
    u_bar = mean_trade_size / 2.0            # E[u; u > 0]
    j_bar = jump_up_probability * mean_adverse_jump   # E[J; J > 0]
    denominator = mu * p_u + gamma + eta
    q = np.linspace(0.0, max_position, points)
    h = q[1] - q[0]
    density = np.exp(-q / mean_trade_size) / (2.0 * mean_trade_size)
    alpha = np.zeros(points)
    beta = np.zeros(points)
    alpha[0] = (mu * p_u + gamma * jump_up_probability) / (mu * p_u + gamma)
    beta[0] = (mu * gamma * (1 - jump_up_probability) / (mu * p_u + gamma) ** 2 * impact * u_bar
               + gamma / (mu * p_u + gamma) * j_bar)
    for n in range(1, points):
        # Where does keep_fraction * q[n] fall on the grid?
        target = keep_fraction * q[n] / h
        low = min(int(np.floor(target)), n - 1) if keep_fraction < 1 else n - 1
        weight_high = target - low
        if low + 1 == n:
            own_weight, known = weight_high, (1 - weight_high, low)
        else:
            own_weight, known = 0.0, None
        x = np.arange(1, n)                  # interior points of the trapezoid rule
        f_inner = density[x]
        # --- alpha ---
        integral = h * ((f_inner * (alpha[n - x] - 1.0)).sum() + 0.5 * density[n] * (alpha[0] - 1.0))
        own_integral = 0.5 * h * density[0]          # multiplies (alpha[n] - 1)
        if known is None:
            a_keep = (1 - weight_high) * alpha[low] + weight_high * alpha[low + 1]
        else:
            a_keep = known[0] * alpha[known[1]]
        right = mu * (p_u + integral - own_integral) + gamma * jump_up_probability + eta * a_keep
        left = denominator - mu * own_integral - eta * own_weight
        alpha[n] = right / left
        # --- beta ---
        beta_integral = h * ((f_inner * beta[n - x]).sum() + 0.5 * density[n] * beta[0])
        beta_own = 0.5 * h * density[0]              # multiplies beta[n]
        cost_integral = h * ((f_inner * (alpha[n - x] - 1.0) * q[x]).sum()
                             + 0.5 * density[n] * (alpha[0] - 1.0) * q[n])
        if known is None:
            b_keep = (1 - weight_high) * beta[low] + weight_high * beta[low + 1]
        else:
            b_keep = known[0] * beta[known[1]]
        right = (mu * (beta_integral + impact * cost_integral - impact * u_bar * (alpha[n] - 1.0))
                 + gamma * j_bar + eta * b_keep)
        left = denominator - mu * beta_own - eta * own_weight
        beta[n] = right / left
    return q, alpha, beta


def continuous_from_model(model: QueueModel, points: int = 801):
    """The continuous solution with primitives matched to a :class:`QueueModel`."""
    trade_rate = model.sell_probability + model.buy_probability
    jump_rate = model.jump_down_probability + model.jump_up_probability
    return solve_continuous(
        trade_rate=trade_rate,
        jump_rate=jump_rate,
        cancellation_rate=model.cancellation_probability,
        mean_trade_size=model.mean_trade_size,
        impact=model.impact,
        jump_up_probability=model.jump_down_probability / jump_rate,
        mean_adverse_jump=model.jump_size,
        keep_fraction=model.keep_fraction,
        max_position=float(model.max_position),
        points=points,
    )
