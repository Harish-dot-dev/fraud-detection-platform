"""The analyst review queue.

    make app

What a reviewer needs, in the order they need it: what was decided and why,
what the payment looked like, what the model reacted to, what similar confirmed
cases turned out to be - and then two buttons.

The LLM summary is generated **on demand**, when a case is opened, not for
every payment in the queue. Generation takes seconds; an analyst opens a
handful of cases a minute. If the summary cannot be grounded in the facts it is
not shown at all, and the facts and SHAP reasons stand on their own.
"""

from __future__ import annotations

import streamlit as st

from analyst_app import data
from analyst_app.reviews import VERDICT_FRAUD, VERDICT_LEGITIMATE, Review, ReviewStore
from analyst_app.theme import STATUS_COLOURS
from common.config import get_settings

st.set_page_config(page_title="Fraud review queue", page_icon="🔍", layout="wide")

DECISION_LABELS = {"review": "REVIEW", "block": "BLOCKED", "allow": "ALLOWED"}


@st.cache_resource
def _review_store() -> ReviewStore | None:
    """Postgres-backed review store, with the case store wired in behind it."""
    settings = get_settings()
    try:
        from genai.case_store import CaseStore, connect
        from genai.embeddings import build_embedder

        connection = connect(
            settings.pgvector_host,
            settings.pgvector_port,
            settings.pgvector_db,
            settings.pgvector_user,
            settings.pgvector_password,
        )
        case_store = None
        try:
            case_store = CaseStore(
                connection, build_embedder(settings.embedding_model, allow_stub=True)
            )
            case_store.create_schema()
        except Exception:  # noqa: BLE001 - reviews still work without retrieval
            pass

        store = ReviewStore(connection, case_store)
        store.create_schema()
        return store
    except Exception as error:  # noqa: BLE001
        st.session_state["review_store_error"] = str(error)
        return None


def _decision_badge(decision: str) -> str:
    colour = STATUS_COLOURS.get(decision, "#52514e")
    # Colour plus the word: a status colour never carries meaning alone.
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:4px;font-weight:600;font-size:0.85em'>"
        f"{DECISION_LABELS.get(decision, decision.upper())}</span>"
    )


def render_case(connection, case: dict, store: ReviewStore | None) -> None:
    """One flagged payment, in full."""
    transaction_id = int(case["transaction_id"])

    st.markdown(
        f"### Payment {transaction_id} &nbsp; {_decision_badge(case['decision'])}",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.metric("Amount", f"{case['amount']:,.2f}")
        st.write(f"**Decided by:** {case['triggered_by']}")
        if case.get("rule_name"):
            st.write(f"**Rule:** `{case['rule_name']}`")
        score = case.get("score")
        st.write(f"**Model score:** {score:.4f}" if score is not None else "**Model score:** n/a")
        st.write(f"**Model version:** {case.get('model_version', 'unknown')}")
    with right:
        st.write(f"**Card token:** `{case.get('card_token', '')[:16]}…`")
        st.write(f"**Product:** {case.get('product_cd') or 'unknown'}")
        st.write(f"**Card:** {case.get('card_brand') or 'unknown'}")
        st.write(f"**Device:** {case.get('device_type') or 'no device data'}")
        st.write(f"**Scored in:** {case.get('latency_ms', 0):.1f} ms")

    detail = data.decision_features(data.default_export_dir(), transaction_id)
    features = detail.get("features", {})
    reasons = detail.get("top_reasons", [])

    if reasons:
        st.markdown("#### Why the model flagged it")
        for reason in reasons:
            st.write(
                f"- **{reason['feature']}** = {reason.get('value')} "
                f"(contribution {reason['contribution']:+.3f})"
            )
    else:
        st.info("No model reasons recorded for this decision.")

    with st.expander("All feature values the decision used"):
        st.json(features)

    _render_assistant(case, features, transaction_id)
    _render_verdict_buttons(case, features, transaction_id, store)


def _render_assistant(case: dict, features: dict, transaction_id: int) -> None:
    """The grounded LLM summary, generated only when asked for."""
    st.markdown("#### Analyst assistant")

    if not st.button("Summarise this case", key=f"summarise_{transaction_id}"):
        st.caption("Generation takes a few seconds, so it runs only when you ask.")
        return

    settings = get_settings()
    with st.spinner("Retrieving similar cases and writing a summary…"):
        try:
            from genai.case_store import CaseStore, connect, describe_case
            from genai.embeddings import build_embedder
            from genai.facts import CaseFacts
            from genai.summarise import OllamaGenerator, summarise_case

            store = CaseStore(
                connect(
                    settings.pgvector_host,
                    settings.pgvector_port,
                    settings.pgvector_db,
                    settings.pgvector_user,
                    settings.pgvector_password,
                ),
                build_embedder(settings.embedding_model, allow_stub=True),
            )
            similar = store.find_similar(
                describe_case(float(case["amount"]), case["decision"], features),
                top_k=settings.rag_top_k,
                exclude_transaction_id=transaction_id,
            )

            facts = CaseFacts(
                transaction_id=transaction_id,
                card_token=case.get("card_token", ""),
                decided_at=case["decided_at"],
                decision=case["decision"],
                score=case.get("score"),
                triggered_by=case.get("triggered_by", "model"),
                reason="",
                rule_name=case.get("rule_name"),
                model_version=str(case.get("model_version", "")),
                review_threshold=0.0,
                block_threshold=1.0,
                amount=float(case["amount"]),
                features=features,
                similar_cases=similar,
            )
            result = summarise_case(
                facts, OllamaGenerator(settings.ollama_base_url, settings.ollama_model)
            )
        except Exception as error:  # noqa: BLE001
            st.warning(f"The assistant is unavailable ({error}). The facts above stand alone.")
            return

    if similar:
        st.markdown("**Similar confirmed cases**")
        for past in similar:
            st.write(f"- {past.summary_line()} (similarity {past.similarity:.2f})")

    if not result.usable:
        # Nothing is better than something wrong.
        st.warning(
            "No summary: the model's answer could not be grounded in the facts "
            f"({result.error}). Use the facts and reasons above."
        )
        return

    summary = result.summary
    st.success(summary.summary)
    if summary.key_risk_signals:
        st.write("**Key signals:** " + " · ".join(summary.key_risk_signals))
    st.caption(
        f"Suggested: {summary.recommended_action} · confidence {summary.confidence:.0%} · "
        f"{result.latency_ms:.0f} ms · a human decides"
    )


def _render_verdict_buttons(
    case: dict, features: dict, transaction_id: int, store: ReviewStore | None
) -> None:
    """The two buttons, and the feedback loop behind them."""
    st.markdown("#### Your decision")
    note = st.text_input("Note (optional)", key=f"note_{transaction_id}")

    confirm, clear = st.columns(2)
    verdict = None
    if confirm.button("Confirm fraud", key=f"fraud_{transaction_id}", type="primary"):
        verdict = VERDICT_FRAUD
    if clear.button("Mark legitimate", key=f"legit_{transaction_id}"):
        verdict = VERDICT_LEGITIMATE

    if verdict is None:
        return
    if store is None:
        st.error("Reviews cannot be saved: the review database is unavailable.")
        return

    store.record(
        Review(
            transaction_id=transaction_id,
            verdict=verdict,
            analyst=st.session_state.get("analyst", "analyst"),
            platform_decision=case["decision"],
            note=note,
            model_version=str(case.get("model_version", "")),
        ),
        case_facts={
            "features": features,
            "card_token": case.get("card_token", ""),
            "decided_at": case["decided_at"],
            "product_cd": case.get("product_cd"),
            "device_type": case.get("device_type"),
        },
    )
    # st.rerun() wipes anything written before it, so the confirmation is
    # stashed and rendered at the top of the next run. Without this the
    # analyst clicks and sees nothing happen.
    st.session_state["flash"] = (
        f"Payment {transaction_id} recorded as {verdict}. "
        "It is now a confirmed case for future retrieval."
    )
    st.rerun()


def main() -> None:
    st.title("Fraud review queue")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    st.sidebar.text_input("Analyst", key="analyst", value="analyst")
    store = _review_store()
    if store is None:
        st.sidebar.warning(
            "Review database unavailable - decisions cannot be saved. "
            f"({st.session_state.get('review_store_error', '')[:120]})"
        )

    try:
        connection = data.connect()
    except data.WarehouseUnavailable as error:
        st.error(str(error))
        return

    reviewed = store.reviewed_ids() if store else set()
    queue = data.review_queue(connection, reviewed)

    if queue.empty:
        st.success("The queue is empty. Every flagged payment has been reviewed.")
        return

    st.caption(f"{len(queue)} payments waiting. Newest first.")
    labels = [
        f"{row.transaction_id} · {row.decision} · {row.amount:,.2f}" for row in queue.itertuples()
    ]
    chosen = st.sidebar.radio("Queue", options=range(len(labels)), format_func=lambda i: labels[i])

    render_case(connection, queue.iloc[chosen].to_dict(), store)

    if store:
        rate = store.agreement_rate()
        if rate is not None:
            st.sidebar.metric("Analyst agreement with platform", f"{rate:.0%}")


main()
