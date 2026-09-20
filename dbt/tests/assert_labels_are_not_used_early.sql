-- Point-in-time correctness, enforced in the warehouse.
--
-- fct_decisions must not expose a label whose chargeback has not arrived yet.
-- If this ever returns rows, a dashboard is reporting accuracy that nobody
-- could have known at the time - the reporting equivalent of training on the
-- future.

select
    transaction_id,
    label_available_at,
    is_confirmed_fraud
from {{ ref('fct_decisions') }}
where is_confirmed_fraud is not null
  and label_available_at > current_timestamp
