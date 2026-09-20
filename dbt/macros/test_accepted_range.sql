{% test dbt_utils_accepted_range(model, column_name, min_value, max_value) %}
-- A range test, written out rather than pulled from dbt_utils.
--
-- dbt_utils would be a package download at build time, and this project is
-- meant to run on a laptop with no network. One macro is cheaper than a
-- dependency.
--
-- Nulls pass: "not known yet" is a legitimate state for a rate whose labels
-- have not arrived, and not_null is a separate test where it is required.

select *
from {{ model }}
where {{ column_name }} is not null
  and ({{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }})

{% endtest %}
