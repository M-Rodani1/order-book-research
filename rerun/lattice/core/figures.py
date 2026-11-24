"""Figures drawn as plain SVG text, with nothing but the standard library and NumPy.

Each function returns a string.  The same inputs always give the same string,
which is what makes a figure reproducible from its saved configuration.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np

BACKGROUND = "#050607"
TEXT = "#F2F4F7"
MUTED = "#9AA3B0"
LINE = "#262B33"
LIME = "#D4FF3A"
BLUE = "#4DA3FF"
ORANGE = "#FF8A3D"
WHITE = "#F2F4F7"
PALETTE = [LIME, BLUE, ORANGE, WHITE, MUTED]

WIDTH, HEIGHT = 760, 440
LEFT, RIGHT, TOP, BOTTOM = 78, 28, 84, 64


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 1e5 else f"{value:.3g}"


def _label(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e7 or magnitude < 0.001:
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    if magnitude >= 1000 and float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.3g}"


def _ticks(low: float, high: float, target: int = 6) -> list:
    if high <= low:
        high = low + 1.0
    raw = (high - low) / target
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * power - raw)) * power
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + 1e-9 * step:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _log_ticks(low: float, high: float) -> list:
    ticks = []
    for power in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for m in (1, 2, 5):
            value = m * 10.0 ** power
            if low * 0.999 <= value <= high * 1.001:
                ticks.append(value)
    return ticks or [low, high]


class _Axis:
    def __init__(self, low, high, start, end, log=False):
        self.log = log
        if log:
            low, high = math.log10(low), math.log10(high)
        if high - low < 1e-12:
            low, high = low - 0.5, high + 0.5
        pad = 0.04 * (high - low)
        self.low, self.high = low - pad, high + pad
        self.start, self.end = start, end

    def __call__(self, value: float) -> float:
        if self.log:
            value = math.log10(value)
        return self.start + (value - self.low) / (self.high - self.low) * (self.end - self.start)

    def ticks(self):
        if self.log:
            return _log_ticks(10 ** self.low, 10 ** self.high)
        return _ticks(self.low, self.high)


def _frame(title: str, x_label: str, y_label: str, note: str = "", extra_height: float = 0) -> list:
    full = HEIGHT + extra_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {full:g}" '
        f'font-family="JetBrains Mono, Menlo, monospace" font-size="12">',
        f'<rect width="{WIDTH}" height="{full:g}" fill="{BACKGROUND}"/>',
        f'<text x="{LEFT}" y="30" fill="{TEXT}" font-family="Chakra Petch, sans-serif" font-size="15" '
        f'font-weight="600" letter-spacing="2">{escape(title.upper())}</text>',
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{HEIGHT - 16}" fill="{MUTED}" text-anchor="middle" '
        f'font-family="Chakra Petch, sans-serif" letter-spacing="2">{escape(x_label.upper())}</text>',
        f'<text transform="translate(20 {(TOP + HEIGHT - BOTTOM) / 2:.1f}) rotate(-90)" fill="{MUTED}" '
        f'text-anchor="middle" font-family="Chakra Petch, sans-serif" letter-spacing="2">'
        f'{escape(y_label.upper())}</text>',
    ]
    if note:
        parts.append(f'<text x="{WIDTH - RIGHT}" y="50" fill="{MUTED}" text-anchor="end">{escape(note)}</text>')
    # Lime corner brackets, as on the stage, at the corners of the whole figure.
    x0, x1, y0, y1, b = 5, WIDTH - 5, 5, full - 5, 16
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        parts.append(f'<path d="M{cx + dx * b} {cy} L{cx} {cy} L{cx} {cy + dy * b}" fill="none" '
                     f'stroke="{LIME}" stroke-width="1.5"/>')
    return parts


def _axes(parts: list, x_axis: _Axis, y_axis: _Axis) -> None:
    bottom, left = HEIGHT - BOTTOM, LEFT
    for tick in y_axis.ticks():
        y = y_axis(tick)
        parts.append(f'<line x1="{left}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" fill="{MUTED}" text-anchor="end">{_label(tick)}</text>')
    for tick in x_axis.ticks():
        x = x_axis(tick)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{bottom}" y2="{bottom + 5}" stroke="{MUTED}"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 20}" fill="{MUTED}" text-anchor="middle">{_label(tick)}</text>')
    parts.append(f'<line x1="{left}" x2="{WIDTH - RIGHT}" y1="{bottom}" y2="{bottom}" stroke="{MUTED}"/>')


def _legend(parts: list, entries: list, note: str = "") -> None:
    """Legend entries in rows under the title, wrapping before the note at the right edge."""
    x, y = LEFT, 50
    edge = WIDTH - RIGHT - (7.3 * len(note) + 16 if note else 0)
    for label, colour, dashed in entries:
        width = 40 + 7.3 * len(label)
        if x + width > edge and x > LEFT:
            x, y = LEFT, y + 17
            edge = WIDTH - RIGHT
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" stroke="{colour}" '
                     f'stroke-width="2.5"{dash}/>')
        parts.append(f'<text x="{x + 28}" y="{y}" fill="{TEXT}">{escape(label)}</text>')
        x += width


STRIP_ROW = 24


def _strip(parts: list, groups: list, caption: str) -> None:
    """Under the plot: for each group, every seed's own estimate beside the pooled one and its interval."""
    values = [v for g in groups for v in [g["pooled"], g["low"], g["high"], *g["seeds"]]
              if v is not None and np.isfinite(v)]
    if not values:
        return
    top = HEIGHT + 4
    left = LEFT + 190
    axis = _Axis(min(values), max(values), left, WIDTH - RIGHT - 10)
    parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{top - 10}" y2="{top - 10}" stroke="{LINE}"/>')
    parts.append(f'<text x="{LEFT}" y="{top + 8}" fill="{MUTED}" font-family="Chakra Petch, sans-serif" '
                 f'letter-spacing="2">{escape(caption.upper())}</text>')
    bottom = top + 22 + STRIP_ROW * len(groups)
    for tick in axis.ticks():
        x = axis(tick)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top + 16}" y2="{bottom}" stroke="{LINE}"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 14}" fill="{MUTED}" text-anchor="middle">{_label(tick)}</text>')
    for row, group in enumerate(groups):
        y = top + 22 + STRIP_ROW * row + STRIP_ROW / 2
        colour = group.get("colour", LIME)
        parts.append(f'<text x="{LEFT}" y="{y + 4:.1f}" fill="{TEXT}">{escape(group["label"])}</text>')
        if group["low"] is not None and group["high"] is not None:
            parts.append(f'<line x1="{axis(group["low"]):.1f}" x2="{axis(group["high"]):.1f}" y1="{y:.1f}" '
                         f'y2="{y:.1f}" stroke="{colour}" stroke-width="3"/>')
        for value in group["seeds"]:               # each seed alone: a hollow square
            if value is not None and np.isfinite(value):
                parts.append(f'<rect x="{axis(value) - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{BACKGROUND}" '
                             f'stroke="{colour}" stroke-width="1.5"/>')
        if group["pooled"] is not None:            # all seeds together: a filled square
            parts.append(f'<rect x="{axis(group["pooled"]) - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
                         f'fill="{colour}"/>')


def strip_height(groups: list) -> float:
    return 22 + STRIP_ROW * len(groups) + 30


def line_figure(series: list, title: str, x_label: str, y_label: str, log_x: bool = False,
                log_y: bool = False, note: str = "", zero_line: bool = False, strip: list | None = None,
                strip_caption: str = "exponent: each seed hollow, all seeds pooled filled") -> str:
    """Lines and points.

    Each series is a dictionary with ``x`` and ``y`` and optionally ``label``,
    ``colour``, ``error`` (bars), ``points`` (draw markers), ``line`` (draw the
    line, default True) and ``dashed``.
    """
    cleaned = []
    for number, item in enumerate(series):
        x = np.asarray(item["x"], dtype=float)
        y = np.asarray(item["y"], dtype=float)
        error = np.asarray(item["error"], dtype=float) if item.get("error") is not None else None
        keep = np.isfinite(x) & np.isfinite(y)
        if log_x:
            keep &= x > 0
        if log_y:
            keep &= y > 0
        cleaned.append({**item, "x": x[keep], "y": y[keep],
                        "error": error[keep] if error is not None else None,
                        "colour": item.get("colour", PALETTE[number % len(PALETTE)])})
    all_x = np.concatenate([s["x"] for s in cleaned]) if cleaned else np.array([0.0, 1.0])
    low_parts, high_parts = [], []
    for s in cleaned:
        if s["error"] is not None and not log_y:
            low_parts.append(s["y"] - s["error"])
            high_parts.append(s["y"] + s["error"])
        else:
            low_parts.append(s["y"])
            high_parts.append(s["y"])
    all_low = np.concatenate(low_parts) if low_parts else np.array([0.0])
    all_high = np.concatenate(high_parts) if high_parts else np.array([1.0])
    if len(all_x) == 0:
        all_x, all_low, all_high = np.array([1.0, 2.0]), np.array([1.0]), np.array([2.0])
    x_axis = _Axis(float(all_x.min()), float(all_x.max()), LEFT, WIDTH - RIGHT, log_x)
    y_axis = _Axis(float(all_low.min()), float(all_high.max()), HEIGHT - BOTTOM, TOP, log_y)
    parts = _frame(title, x_label, y_label, note, strip_height(strip) if strip else 0)
    _axes(parts, x_axis, y_axis)
    if strip:
        _strip(parts, strip, strip_caption)
    if zero_line and not log_y and y_axis.low < 0 < y_axis.high:
        y = y_axis(0.0)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{MUTED}" '
                     f'stroke-dasharray="2 4"/>')
    legend = []
    for s in cleaned:
        colour = s["colour"]
        points = [(x_axis(a), y_axis(b)) for a, b in zip(s["x"], s["y"])]
        if s.get("line", True) and len(points) > 1:
            dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
            path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f} {py:.1f}" for i, (px, py) in enumerate(points))
            parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>')
        if s["error"] is not None:
            for (px, _), value, err in zip(points, s["y"], s["error"]):
                top = y_axis(value + err)
                bottom = y_axis(max(value - err, 10 ** y_axis.low) if log_y else value - err)
                parts.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top:.1f}" y2="{bottom:.1f}" '
                             f'stroke="{colour}" stroke-width="1.2"/>')
        if s.get("points"):
            for px, py in points:
                parts.append(f'<rect x="{px - 3:.1f}" y="{py - 3:.1f}" width="6" height="6" fill="{colour}"/>')
        if s.get("label"):
            legend.append((s["label"], colour, bool(s.get("dashed"))))
    _legend(parts, legend, note)
    parts.append("</svg>")
    return "\n".join(parts)


def _shade(value: float, limit: float) -> str:
    """Flat cell colour: blue for positive, orange for negative, in five steps each."""
    if limit <= 0 or abs(value) < 1e-12:
        return "#0B0D10"
    level = min(5, max(1, math.ceil(abs(value) / limit * 5)))
    blues = ["#10243A", "#1B3F66", "#2A5E96", "#3A80C8", BLUE]
    oranges = ["#3A2010", "#66381B", "#96522A", "#C86E38", ORANGE]
    return (blues if value > 0 else oranges)[level - 1]


def heatmap_figure(grid, x_values, y_values, title: str, x_label: str, y_label: str,
                   overlays: list | None = None, note: str = "", legend: list | None = None,
                   colours: dict | None = None) -> str:
    """A grid of flat cells: rows are ``y_values`` (drawn bottom to top), columns ``x_values``.

    ``overlays`` are stepped lines: dictionaries with ``y`` (one row value per
    entry), ``x`` (column position for each) and optionally ``label``, ``colour``, ``dashed``.

    ``colours`` is for a grid whose numbers stand for categories: it maps each
    cell value to the colour of that category, in place of the blue and orange shading.
    """
    grid = np.asarray(grid, dtype=float)
    rows, columns = grid.shape
    parts = _frame(title, x_label, y_label, note)
    plot_w, plot_h = WIDTH - RIGHT - LEFT, HEIGHT - BOTTOM - TOP
    cell_w, cell_h = plot_w / columns, plot_h / rows
    limit = float(np.abs(grid).max())
    for r in range(rows):
        for c in range(columns):
            x = LEFT + c * cell_w
            y = HEIGHT - BOTTOM - (r + 1) * cell_h
            value = float(grid[r, c])
            fill = colours[value] if colours is not None and value in colours else _shade(value, limit)
            parts.append(f'<rect x="{x + 0.5:.1f}" y="{y + 0.5:.1f}" width="{cell_w - 1:.1f}" '
                         f'height="{cell_h - 1:.1f}" fill="{fill}"/>')
    step_x = max(1, columns // 13)
    for c in range(0, columns, step_x):
        parts.append(f'<text x="{LEFT + (c + 0.5) * cell_w:.1f}" y="{HEIGHT - BOTTOM + 20}" fill="{MUTED}" '
                     f'text-anchor="middle">{_label(float(x_values[c]))}</text>')
    step_y = max(1, rows // 8)
    for r in range(0, rows, step_y):
        parts.append(f'<text x="{LEFT - 8}" y="{HEIGHT - BOTTOM - (r + 0.5) * cell_h + 4:.1f}" fill="{MUTED}" '
                     f'text-anchor="end">{_number(float(y_values[r]))}</text>')
    # ``legend`` replaces the default key: a list of (label, colour) pairs.
    legend = ([(label, colour, False) for label, colour in legend] if legend is not None
              else [("positive", BLUE, False), ("negative", ORANGE, False)])
    for number, overlay in enumerate(overlays or []):
        colour = overlay.get("colour", [LIME, WHITE, MUTED][number % 3])
        dash = ' stroke-dasharray="6 4"' if overlay.get("dashed") else ""
        path = []
        for r, column in enumerate(overlay["x"]):
            if column is None or not np.isfinite(column):
                continue
            x = LEFT + min(max(float(column), 0.0), columns) * cell_w
            y_low = HEIGHT - BOTTOM - r * cell_h
            y_high = y_low - cell_h
            path.append(f"{'M' if not path else 'L'}{x:.1f} {y_low:.1f} L{x:.1f} {y_high:.1f}")
        if path:
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{colour}" stroke-width="2.5"{dash}/>')
        if overlay.get("label"):
            legend.append((overlay["label"], colour, bool(overlay.get("dashed"))))
    _legend(parts, legend, note)
    parts.append("</svg>")
    return "\n".join(parts)


def bar_figure(labels: list, values, title: str, x_label: str, y_label: str, colours: list | None = None,
               note: str = "") -> str:
    """Flat vertical bars, one per label."""
    values = np.asarray(values, dtype=float)
    parts = _frame(title, x_label, y_label, note)
    top = float(max(values.max(), 0)) if len(values) else 1.0
    low = float(min(values.min(), 0)) if len(values) else 0.0
    y_axis = _Axis(low, top if top > low else low + 1, HEIGHT - BOTTOM, TOP)
    for tick in y_axis.ticks():
        y = y_axis(tick)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" fill="{MUTED}" text-anchor="end">{_label(tick)}</text>')
    slot = (WIDTH - RIGHT - LEFT) / max(len(values), 1)
    for i, (label, value) in enumerate(zip(labels, values)):
        colour = (colours[i] if colours else BLUE)
        y0, y1 = y_axis(0.0), y_axis(float(value))
        parts.append(f'<rect x="{LEFT + i * slot + slot * 0.15:.1f}" y="{min(y0, y1):.1f}" '
                     f'width="{slot * 0.7:.1f}" height="{abs(y1 - y0):.1f}" fill="{colour}"/>')
        parts.append(f'<text x="{LEFT + (i + 0.5) * slot:.1f}" y="{HEIGHT - BOTTOM + 20}" fill="{MUTED}" '
                     f'text-anchor="middle">{escape(str(label))}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
