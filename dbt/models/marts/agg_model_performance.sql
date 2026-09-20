-- Model quality over time, per model version.
--
-- Only matured labels are counted, so a version that shipped last week shows
-- nothing until its chargebacks arrive. That gap is real and the dashboard
-- should show it rather than fill it in.

with labelled as (

    select *
    from {{ ref('fct_decisions') }}
    where label_is_known

)

select
    model_version,
    decision_date,
    count(*) as labelled_payments,
    count(*) filter (where is_confirmed_fraud) as fraud_count,

    -- Precision and recall at the block threshold: the decision a customer
    -- actually feels.
    case
        when count(*) filter (where decision = 'block') > 0
        then round(
            count(*) filter (where outcome = 'fraud_blocked') * 1.0
            / count(*) filter (where decision = 'block'),
            6
        )
    end as precision_at_block,

    case
        when count(*) filter (where is_confirmed_fraud) > 0
        then round(
            count(*) filter (where outcome = 'fraud_blocked') * 1.0
            / count(*) filter (where is_confirmed_fraud),
            6
        )
    end as recall_at_block,

    -- Recall counting the review queue too: an analyst does see those.
    case
        when count(*) filter (where is_confirmed_fraud) > 0
        then round(
            count(*) filter (where outcome in ('fraud_blocked', 'fraud_to_review')) * 1.0
            / count(*) filter (where is_confirmed_fraud),
            6
        )
    end as recall_including_review,

    coalesce(sum(amount) filter (where outcome = 'fraud_missed'), 0) as fraud_value_missed

from labelled
group by model_version, decision_date
order by model_version, decision_date
