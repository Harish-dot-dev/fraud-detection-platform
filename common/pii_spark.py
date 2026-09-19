"""Spark-side card tokenisation.

Note the missing ``from __future__ import annotations`` in this module, and
leave it missing: ``pandas_udf`` inspects the decorated function's type hints
*at runtime* to work out its signature. With postponed evaluation the hints are
plain strings and Spark raises ``UNSUPPORTED_SIGNATURE``. Keeping this one
function in its own module means the rest of the codebase can use modern
annotations without stepping on it.
"""

import pandas as pd
from pyspark.sql import Column
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from common.pii import card_token


def tokenise_column(card_key_column: Column, salt: str) -> Column:
    """Vectorised HMAC tokenisation of a card identity proxy column.

    A pandas UDF rather than Spark's native ``sha2``: the token has to be
    byte-identical to the one the scoring API computes through
    ``common.pii.card_token``, and there is exactly one implementation of that.
    Vectorising keeps the cost to seconds over the full dataset rather than the
    minutes a row-at-a-time Python UDF would take.
    """

    @pandas_udf(StringType())
    def _tokenise(keys: pd.Series) -> pd.Series:
        return keys.map(lambda key: card_token(key, salt) if key is not None else None)

    return _tokenise(card_key_column)
