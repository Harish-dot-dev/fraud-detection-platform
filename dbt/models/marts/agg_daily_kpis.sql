-- The numbers a fraud team looks at every morning.
--
-- Split deliberately into two groups: volumes, which are known the moment a
-- payment is scored, and outcomes, which are only known once the chargeback
-- window has closed. Reporting the two as though they arrive together is the
-- most common way a fraud dashboard lies.

with decisions as (
    select * from {{ ref('fct_decisions') }}
)

select
    decision_date,

    -- --- Known immediately: volumes and operational load -------------------
    count(*) as payments_scored,
    count(*) filter (where decision = 'allow') as allowed,
    count(*) filter (where decision = 'review') as sent_to_review,
    count(*) filter (where decision = 'block') as blocked,
    round(count(*) filter (where decision = 'review') * 1.0 / count(*), 6) as review_rate,
    round(count(*) filter (where decision = 'block') * 1.0 / count(*), 6) as block_rate,
    -- The analyst queue this generated: the number that decides whether the
    -- thresholds are affordable.
    count(*) filter (where decision = 'review') as review_queue_size,
    count(*) filter (where triggered_by = 'rule') as decided_by_rule,
    count(*) filter (where triggered_by = 'model') as decided_by_model,
    count(*) filter (where degraded) as decided_without_a_model,

    sum(amount) as payment_value,
    sum(amount) filter (where decision = 'block') as blocked_value,

    -- --- Latency -----------------------------------------------------------
    round(median(latency_ms), 2) as latency_p50_ms,
    round(quantile_cont(latency_ms, 0.95), 2) as latency_p95_ms,
    round(quantile_cont(latency_ms, 0.99), 2) as latency_p99_ms,

    -- --- Known weeks later: was any of it right? ---------------------------
    count(*) filter (where label_is_known) as labels_known,
    count(*) filter (where not label_is_known) as labels_pending,
    count(*) filter (where outcome = 'fraud_blocked') as fraud_blocked,
    count(*) filter (where outcome = 'fraud_to_review') as fraud_to_review,
    count(*) filter (where outcome = 'fraud_missed') as fraud_missed,
    count(*) filter (where outcome = 'false_block') as false_blocks,

    coalesce(sum(amount) filter (where outcome in ('fraud_blocked', 'fraud_to_review')), 0)
        as fraud_value_caught,
    coalesce(sum(amount) filter (where outcome = 'fraud_missed'), 0) as fraud_value_missed,

    -- Share of legitimate payments wrongly blocked. Null rather than zero
    -- while the labels are still pending: "we don't know yet" and "we blocked
    -- nobody" are not the same statement.
    case
        when count(*) filter (where label_is_known and not is_confirmed_fraud) > 0
        then round(
            count(*) filter (where outcome = 'false_block') * 1.0
            / count(*) filter (where label_is_known and not is_confirmed_fraud),
            6
        )
    end as false_positive_rate

from decisions
group by decision_date
order by decision_date
