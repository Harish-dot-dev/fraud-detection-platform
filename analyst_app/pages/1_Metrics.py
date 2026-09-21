"""The fraud metrics page.

Laid out in the order a fraud manager asks the questions:

1. **What happened today** - stat tiles, no charts. A single number is read
   faster as a number than as a bar of length one.
2. **Where the decisions went** - the allow / review / block split over time.
3. **Money** - fraud value caught against fraud value missed. The one chart
   that does not lie about a model with respectable recall and expensive
   misses.
4. **Was it right** - precision and recall, over matured labels only, so the
   most recent days are deliberately absent rather than flattering.
5. **How fast** - p50, p95, p99 latency.
6. **The rules** - how often each fires and how often it is right.

Every chart carries a table underneath it. That is partly the documented relief
for one palette slot that sits below 3:1 contrast, and partly because a fraud
manager will always want the number rather than the shape.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analyst_app import data
from analyst_app.theme import SERIES, STATUS_COLOURS, style

st.set_page_config(page_title="Fraud metrics", page_icon="📊", layout="wide")
st.title("Fraud metrics")


def _lines(frame, x, series: list[tuple[str, str, str]], y_title: str) -> go.Figure:
    """A multi-series chart over time. One y-axis, always.

    With a single day of data a line chart is the wrong form: it draws one
    lonely dot and invents a time axis spanning two milliseconds either side of
    it. Early in a run - or on the first day after a deployment - that is
    exactly the situation, so a single period is drawn as grouped bars instead.
    """
    single_period = len(frame) < 2
    figure = go.Figure()

    for column, label, colour in series:
        if column not in frame.columns:
            continue
        if single_period:
            figure.add_trace(
                go.Bar(
                    x=[label],
                    y=frame[column],
                    name=label,
                    marker={"color": colour},
                    text=frame[column],
                    texttemplate="%{text:,.0f}",
                    textposition="outside",
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=frame[x],
                    y=frame[column],
                    name=label,
                    mode="lines+markers",
                    line={"color": colour, "width": 2},
                    marker={"size": 8},
                )
            )

    # A legend whenever there is more than one series; none for a single one,
    # where the heading already names it.
    styled = style(figure, y_title=y_title, show_legend=len(figure.data) > 1)
    if single_period:
        # The category axis carries the labels; a date axis over one point is
        # noise ("23:59:59.999 ... 00:00:00.001").
        styled.update_xaxes(type="category")
    return styled


try:
    connection = data.connect()
except data.WarehouseUnavailable as error:
    st.error(str(error))
    st.stop()

kpis = data.daily_kpis(connection)
if kpis.empty:
    st.warning("No decisions in the warehouse yet. Run `make warehouse` after scoring payments.")
    st.stop()

latest = kpis.iloc[-1]

# --- 1. Today, as numbers ---------------------------------------------------
st.subheader(f"Latest day · {latest['decision_date']}")
columns = st.columns(6)
columns[0].metric("Payments scored", f"{int(latest['payments_scored']):,}")
columns[1].metric("Review queue", f"{int(latest['review_queue_size']):,}")
columns[2].metric("Blocked", f"{int(latest['blocked']):,}")
columns[3].metric("Fraud value caught", f"{latest['fraud_value_caught']:,.0f}")
# Its own tile rather than a delta on the one above: a delta would colour
# "63 missed" green or red by arithmetic rather than by meaning.
columns[4].metric(
    "Fraud value missed",
    f"{latest['fraud_value_missed']:,.0f}",
    help="Confirmed fraud that was allowed through.",
)
false_positive_rate = latest.get("false_positive_rate")
columns[5].metric(
    "False positive rate",
    "pending" if false_positive_rate is None else f"{false_positive_rate:.2%}",
    help="Null until the chargeback window closes - not the same as zero.",
)

if int(latest["labels_pending"]) > 0:
    st.caption(
        f"{int(latest['labels_pending']):,} of these payments have no confirmed outcome yet - "
        "chargebacks take weeks, so accuracy for recent days is genuinely unknown."
    )

# --- 2. Where the decisions went --------------------------------------------
st.subheader("Decisions")
decisions = _lines(
    kpis,
    "decision_date",
    [
        ("allowed", "Allowed", STATUS_COLOURS["allow"]),
        ("sent_to_review", "Sent to review", STATUS_COLOURS["review"]),
        ("blocked", "Blocked", STATUS_COLOURS["block"]),
    ],
    "payments",
)
st.plotly_chart(decisions, use_container_width=True)
with st.expander("Table"):
    st.dataframe(
        kpis[
            [
                "decision_date",
                "payments_scored",
                "allowed",
                "sent_to_review",
                "blocked",
                "review_rate",
                "block_rate",
            ]
        ],
        use_container_width=True,
    )

# --- 3. Money ---------------------------------------------------------------
st.subheader("Fraud value caught against value missed")
st.caption(
    "Counting frauds caught can flatter a model that misses the expensive ones. "
    "This is the chart that does not."
)
money = _lines(
    kpis,
    "decision_date",
    [
        ("fraud_value_caught", "Caught", SERIES[0]),
        ("fraud_value_missed", "Missed", SERIES[1]),
    ],
    "value",
)
st.plotly_chart(money, use_container_width=True)
with st.expander("Table"):
    st.dataframe(
        kpis[
            [
                "decision_date",
                "fraud_value_caught",
                "fraud_value_missed",
                "fraud_blocked",
                "fraud_to_review",
                "fraud_missed",
            ]
        ],
        use_container_width=True,
    )

# --- 4. Was it right --------------------------------------------------------
st.subheader("Model performance, over matured labels only")
performance = data.model_performance(connection)
if performance.empty:
    st.info("No matured labels yet. Precision and recall appear once chargebacks arrive.")
else:
    for version in sorted(performance["model_version"].unique()):
        subset = performance[performance["model_version"] == version]
        st.plotly_chart(
            _lines(
                subset,
                "decision_date",
                [
                    ("precision_at_block", "Precision at block", SERIES[0]),
                    ("recall_at_block", "Recall at block", SERIES[1]),
                    ("recall_including_review", "Recall including review", SERIES[2]),
                ],
                "rate",
            ).update_layout(title=f"Model version {version}"),
            use_container_width=True,
        )
    with st.expander("Table"):
        st.dataframe(performance, use_container_width=True)

# --- 5. Latency -------------------------------------------------------------
st.subheader("Scoring latency")
latency = _lines(
    kpis,
    "decision_date",
    [
        ("latency_p50_ms", "p50", SERIES[0]),
        ("latency_p95_ms", "p95", SERIES[1]),
        ("latency_p99_ms", "p99", SERIES[2]),
    ],
    "milliseconds",
)
st.plotly_chart(latency, use_container_width=True)

# --- 6. Rules ---------------------------------------------------------------
st.subheader("Rule effectiveness")
st.caption(
    "A rule that fires constantly and is rarely right is not a safety net - "
    "it is a tax on this queue."
)
rules = data.rule_effectiveness(connection)
if rules.empty:
    st.info("No rules have fired yet.")
else:
    figure = go.Figure(
        go.Bar(
            x=rules["times_fired"],
            y=rules["rule_name"],
            orientation="h",
            marker={"color": SERIES[0]},
            text=rules["times_fired"],
            textposition="outside",
        )
    )
    st.plotly_chart(style(figure, y_title="", show_legend=False), use_container_width=True)
    st.dataframe(
        rules[
            ["rule_name", "times_fired", "blocked", "sent_to_review", "caught_fraud", "hit_rate"]
        ],
        use_container_width=True,
    )
