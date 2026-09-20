-- Payments as the warehouse sees them: typed, renamed, no card identifiers.

select
    transaction_id,
    card_token,
    event_time,
    cast(event_time as date) as payment_date,
    amount,
    product_cd,
    card4 as card_brand,
    card6 as card_type,
    p_emaildomain as payer_email_domain,
    r_emaildomain as recipient_email_domain,
    device_type
from {{ source('platform', 'silver') }}
