-- The grain of the whole warehouse: one row per payment, with what the
-- platform decided, what it cost, and - once it arrived - what the truth was.
--
-- The label is deliberately left null until label_available_at has passed.
-- Everything downstream therefore has to be explicit about whether it is
-- measuring decisions (known immediately) or accuracy (known weeks later),
-- which is the distinction fraud reporting most often blurs.

with decisions as (
    select * from {{ ref('stg_decisions') }}
),

payments as (
    select * from {{ ref('stg_payments') }}
),

labels as (
    select * from {{ ref('stg_chargebacks') }}
)

select
    decisions.transaction_id,
    decisions.card_token,
    decisions.decided_at,
    decisions.decision_date,
    decisions.decision,
    decisions.score,
    decisions.triggered_by,
    decisions.rule_name,
    decisions.model_version,
    decisions.latency_ms,
    decisions.degraded,
    decisions.reason_count,
    decisions.top_reason_feature,

    payments.amount,
    payments.product_cd,
    payments.card_brand,
    payments.card_type,
    payments.payer_email_domain,
    payments.device_type,

    labels.label_available_at,
    labels.label_source,
    -- Known only once the dispute window has done its work.
    case
        when labels.label_available_at <= current_timestamp then labels.is_confirmed_fraud
    end as is_confirmed_fraud,
    labels.label_available_at <= current_timestamp as label_is_known,

    -- Outcome categories, used by every aggregate downstream.
    decisions.decision in ('block', 'review') as was_flagged,
    case
        when labels.label_available_at > current_timestamp then 'pending'
        when labels.is_fraud = 1 and decisions.decision = 'block' then 'fraud_blocked'
        when labels.is_fraud = 1 and decisions.decision = 'review' then 'fraud_to_review'
        when labels.is_fraud = 1 and decisions.decision = 'allow' then 'fraud_missed'
        when labels.is_fraud = 0 and decisions.decision = 'block' then 'false_block'
        when labels.is_fraud = 0 and decisions.decision = 'review' then 'false_review'
        else 'correctly_allowed'
    end as outcome

from decisions
left join payments using (transaction_id)
left join labels using (transaction_id)
