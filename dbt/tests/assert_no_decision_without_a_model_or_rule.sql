-- Every decision must be attributable.
--
-- The audit requirement in one query: if a payment was blocked or sent to an
-- analyst, the record has to say what did it - a named rule, or a model
-- version and a score. A flagged payment nobody can explain is the failure
-- this whole decision log exists to prevent.

select
    transaction_id,
    decision,
    triggered_by,
    rule_name,
    model_version,
    score
from {{ ref('fct_decisions') }}
where decision in ('block', 'review')
  and rule_name is null
  and (model_version = 'none' or score is null)
