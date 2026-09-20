-- One row per payment: the decision the platform actually acted on.
--
-- A payment can be scored more than once - a retried request, a replayed
-- topic - and every one of those attempts is in the audit log by design.
-- For analytics only the latest matters, so the rest are ranked away here
-- rather than silently double-counted in every downstream metric.

with ranked as (

    select
        *,
        row_number() over (
            partition by transaction_id
            order by decided_at desc
        ) as attempt_rank
    from {{ source('platform', 'decisions') }}

)

select
    transaction_id,
    card_token,
    decided_at,
    cast(decided_at as date) as decision_date,
    decision,
    score,
    triggered_by,
    reason,
    rule_name,
    model_version,
    review_threshold,
    block_threshold,
    latency_ms,
    degraded,
    -- The reasons an analyst would see. Empty for allowed payments by design:
    -- SHAP only runs where somebody will read it.
    len(top_reasons) as reason_count,
    case when len(top_reasons) > 0 then top_reasons[1].feature end as top_reason_feature
from ranked
where attempt_rank = 1
