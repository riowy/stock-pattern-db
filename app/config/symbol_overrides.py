"""Canonical <-> provider ticker translation overrides.

Different data providers spell the same security differently (dots vs.
dashes for share classes being the classic example: ``BRK.B`` vs.
``BRK-B``). We keep one *canonical* ticker per security (as close to the
"normal" market convention as possible) and let each provider adapter
translate to/from its own spelling using this seed table.

The seed table below is loaded into the ``symbol_mappings`` DuckDB table on
``stockdb init`` / ``stockdb sync-universe`` (see app/db/schema.py and
app/normalization/symbols.py). Analysis code should only ever use canonical
tickers or, better, ``security_id``.
"""

from __future__ import annotations

# canonical_symbol -> {provider_name: provider_symbol}
SYMBOL_OVERRIDE_SEED: dict[str, dict[str, str]] = {
    "BRK.B": {"yfinance": "BRK-B"},
    "BRK.A": {"yfinance": "BRK-A"},
    "BF.B": {"yfinance": "BF-B"},
}
