"""Numeric comparison helpers for price-scale validation.

Yahoo/yfinance bars are typically float32-ish. A $24 preferred can show
``low = open + 0.0001`` from rounding; that is not a true OHLC violation.

Tolerance (v1, documented and tested):

    tol(a, b) = max(ABS_TOL, REL_TOL * max(|a|, |b|, 1.0))

    ABS_TOL = 0.001   # 0.1 cent; well below the $0.01 US equity tick
    REL_TOL = 1e-5    # ~0.024 on a $24 name, still << a 1-tick error

``approximately_ge(a, b)`` is true iff ``a + tol >= b``.
True violations (e.g. high 10.08 vs open 10.15) remain critical.
Raw prices are never auto-corrected.
"""

from __future__ import annotations

import math

import polars as pl

OHLC_ABS_TOL = 0.001
OHLC_REL_TOL = 1e-5


def ohlc_tolerance(a: float, b: float, abs_tol: float = OHLC_ABS_TOL, rel_tol: float = OHLC_REL_TOL) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return max(abs_tol, rel_tol * scale)


def approximately_ge(a: float, b: float, abs_tol: float = OHLC_ABS_TOL, rel_tol: float = OHLC_REL_TOL) -> bool:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return False
    return a + ohlc_tolerance(a, b, abs_tol, rel_tol) >= b


def approximately_le(a: float, b: float, abs_tol: float = OHLC_ABS_TOL, rel_tol: float = OHLC_REL_TOL) -> bool:
    return approximately_ge(b, a, abs_tol, rel_tol)


def ohlc_tol_expr(a: str, b: str, abs_tol: float = OHLC_ABS_TOL, rel_tol: float = OHLC_REL_TOL) -> pl.Expr:
    scale = pl.max_horizontal(pl.col(a).abs(), pl.col(b).abs(), pl.lit(1.0))
    return pl.max_horizontal(pl.lit(abs_tol), pl.lit(rel_tol) * scale)


def approximately_ge_expr(a: str, b: str) -> pl.Expr:
    return pl.col(a) + ohlc_tol_expr(a, b) >= pl.col(b)


def ohlc_valid_expr(
    high: str = "high",
    low: str = "low",
    open_: str = "open",
    close: str = "close",
) -> pl.Expr:
    """True when the bar is complete and OHLC inequalities hold within tolerance."""
    complete = (
        pl.col(open_).is_not_null()
        & pl.col(high).is_not_null()
        & pl.col(low).is_not_null()
        & pl.col(close).is_not_null()
    )
    return complete & (
        approximately_ge_expr(high, low)
        & approximately_ge_expr(high, open_)
        & approximately_ge_expr(high, close)
        & approximately_ge_expr(open_, low)
        & approximately_ge_expr(close, low)
    )
