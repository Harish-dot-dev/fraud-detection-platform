-- Labels, with the date each one became known.
--
-- label_available_at is the column that keeps this honest: a dashboard that
-- joins on is_fraud alone is reporting performance nobody could have known
-- at the time.

select
    transaction_id,
    is_fraud,
    cast(is_fraud as boolean) as is_confirmed_fraud,
    event_time,
    label_available_at,
    cast(label_available_at as date) as label_available_date,
    label_source,
    delay_days
from {{ source('platform', 'chargebacks') }}
