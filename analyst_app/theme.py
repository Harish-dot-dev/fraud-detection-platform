"""Chart styling, in one place.

Two palettes, used for two different jobs:

**Status** (green / amber / red) for the decision outcomes - allow, review,
block. These are semantic: an analyst reads "red" as "we stopped it" without
consulting a legend. Status colours are fixed and deliberately sit outside the
categorical lightness band, so they always ship with a **text label** beside
them - never colour alone.

**Categorical slots 1-3** for everything else (money caught vs missed, latency
percentiles, precision vs recall). Validated as a set: worst adjacent
colour-vision-deficient separation dE 9.2, normal-vision dE 27.6, both clear of
the floors. Aqua sits below 3:1 contrast on a light surface, so every chart
using it also offers the underlying table - the documented relief.

The rules the charts follow, which are easier to keep than to remember:

* never two y-axes - two measures of different scale get two charts;
* a legend whenever there is more than one series;
* recessive grid and axes, thin marks;
* hover tooltips everywhere (Plotly gives these, and they are left on).
"""

from __future__ import annotations

from typing import Any

# Decision outcomes. Semantic, fixed, always paired with a label.
STATUS_COLOURS = {
    "allow": "#0ca30c",
    "review": "#fab219",
    "block": "#d03b3b",
}

# Validated categorical slots, in fixed order. Never cycled, never reordered
# by rank - a filter that drops a series must not repaint the survivors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

GRID = "rgba(0,0,0,0.08)"
AXIS_TEXT = "#52514e"


def style(figure: Any, title: str = "", y_title: str = "", show_legend: bool = True) -> Any:
    """Apply the house style to a Plotly figure."""
    layout = {
        "template": "plotly_white",
        # Generous top margin: bar value labels sit outside the bar and get
        # clipped against a tight one.
        "margin": {"l": 56, "r": 20, "t": 60 if title else 44, "b": 44},
        "height": 340,
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "x": 0,
            "title": {"text": ""},
        },
        "hovermode": "x unified",
        "font": {"color": AXIS_TEXT, "size": 12},
        "bargap": 0.45,
    }
    # Only set a title when there is one. Passing title=None leaves Plotly
    # with a title object whose text is undefined, and it renders the literal
    # string "undefined" above the chart.
    if title:
        layout["title"] = {"text": title}

    figure.update_layout(**layout)
    figure.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID)
    figure.update_yaxes(
        title=y_title or None, gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)"
    )
    return figure
