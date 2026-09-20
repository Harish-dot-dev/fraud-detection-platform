-- How each rule is doing.
--
-- Rules are written by people under time pressure and then never revisited. A
-- rule that fires constantly and is almost never right is not a safety net,
-- it is a tax on the analyst queue - and this is the table that says so.

with fired as (

    select *
    from {{ ref('fct_decisions') }}
    where rule_name is not null

)

select
    rule_name,
    count(*) as times_fired,
    count(*) filter (where decision = 'block') as blocked,
    count(*) filter (where decision = 'review') as sent_to_review,
    count(*) filter (where label_is_known) as labels_known,
    count(*) filter (where is_confirmed_fraud) as caught_fraud,

    -- Of the payments this rule flagged and we now know the truth about, how
    -- many were actually fraud.
    case
        when count(*) filter (where label_is_known) > 0
        then round(
            count(*) filter (where is_confirmed_fraud) * 1.0
            / count(*) filter (where label_is_known),
            6
        )
    end as hit_rate,

    round(avg(amount), 2) as average_amount,
    coalesce(sum(amount) filter (where is_confirmed_fraud), 0) as fraud_value_flagged

from fired
group by rule_name
order by times_fired desc
