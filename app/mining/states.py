"""Human-readable boolean states for mining. Quantile cutoffs freeze on ANALYSIS only."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from app.mining.config import FAMILIES


@dataclass(frozen=True)
class StateSpec:
    name: str
    family: str
    expr: pl.Expr
    source: str = "indicator"
    kind: str = "state"


# Prefer indicator columns. Feature-engine twins are used only if the preferred name is absent.
QUANTILE_SOURCES = (
    ("hist_vol_20", "VOLATILITY"),
    ("volatility_20d", "VOLATILITY"),
    ("dist_high_252", "BREAKOUT"),
    ("distance_high_252d", "BREAKOUT"),
    ("volume_ratio_20", "VOLUME"),
    ("volume_ratio_20d", "VOLUME"),
    ("natr_14", "VOLATILITY"),
    ("atr_pct_14", "VOLATILITY"),
    ("adx_14", "TREND"),
)
QUANTILE_ALIASES = {
    "volatility_20d": "hist_vol_20",
    "distance_high_252d": "dist_high_252",
    "volume_ratio_20d": "volume_ratio_20",
    "atr_pct_14": "natr_14",
}


def _col(name: str) -> pl.Expr | None:
    return pl.col(name) if True else None


def base_state_specs(columns: set[str]) -> list[StateSpec]:
    """Return states whose source columns exist on the frame."""
    candidates = [
        StateSpec("rsi14_le_30", "MOMENTUM", pl.col("rsi_14") <= 30, "rsi"),
        StateSpec("rsi14_ge_70", "MOMENTUM", pl.col("rsi_14") >= 70, "rsi"),
        StateSpec("price_above_sma20", "TREND", pl.col("close_above_sma20") == True, "sma"),  # noqa: E712
        StateSpec("price_above_sma50", "TREND", pl.col("close_above_sma50") == True, "sma"),
        StateSpec("price_above_sma200", "TREND", pl.col("close_above_sma200") == True, "sma"),
        StateSpec("sma20_gt_sma50", "TREND", pl.col("sma20_above_sma50") == True, "sma"),
        StateSpec("sma50_gt_sma200", "TREND", pl.col("sma50_above_sma200") == True, "sma"),
        StateSpec("golden_cross", "TREND", pl.col("golden_cross_50_200") == True, "sma", "event"),
        StateSpec("death_cross", "TREND", pl.col("death_cross_50_200") == True, "sma", "event"),
        StateSpec("macd_positive", "TREND", pl.col("macd") > 0, "macd"),
        StateSpec("macd_cross_up", "TREND", pl.col("macd_cross_signal_up") == True, "macd", "event"),
        StateSpec("adx_gt_25", "TREND", pl.col("adx14_gt_25") == True, "adx"),
        StateSpec("stochastic_oversold", "MOMENTUM", pl.col("stoch_oversold") == True, "stoch"),
        StateSpec("stochastic_overbought", "MOMENTUM", pl.col("stoch_overbought") == True, "stoch"),
        StateSpec("bb_below_lower", "BAND_CHANNEL", pl.col("close_below_bb_lower") == True, "bb"),
        StateSpec("bb_above_upper", "BAND_CHANNEL", pl.col("close_above_bb_upper") == True, "bb"),
        StateSpec("bb_reentry_lower", "BAND_CHANNEL", pl.col("reentry_from_lower") == True, "bb", "event"),
        StateSpec("bollinger_squeeze", "BAND_CHANNEL", pl.col("squeeze_on") == True, "squeeze"),
        StateSpec("squeeze_release", "BAND_CHANNEL", pl.col("squeeze_release") == True, "squeeze", "event"),
        StateSpec("price_above_cloud", "ICHIMOKU", pl.col("price_above_cloud") == True, "ichimoku"),
        StateSpec("price_below_cloud", "ICHIMOKU", pl.col("price_below_cloud") == True, "ichimoku"),
        StateSpec("tk_cross_up", "ICHIMOKU", pl.col("tenkan_cross_kijun_up") == True, "ichimoku", "event"),
        StateSpec("tk_cross_down", "ICHIMOKU", pl.col("tenkan_cross_kijun_down") == True, "ichimoku", "event"),
        StateSpec("supertrend_up", "TREND", pl.col("supertrend_10_3_dir") == 1, "supertrend"),
        StateSpec("supertrend_flip_up", "TREND", pl.col("supertrend_flip_up") == True, "supertrend", "event"),
        StateSpec("donchian_20_breakout", "BREAKOUT", pl.col("donchian_20_breakout_up") == True, "donchian"),
        StateSpec("volume_ratio20_gt_1_5", "VOLUME", pl.col("volume_ratio_20") > 1.5, "volume"),
        StateSpec("volume_ratio20_gt_2", "VOLUME", pl.col("volume_ratio_20") > 2.0, "volume"),
        StateSpec("new_high_252", "BREAKOUT", pl.col("new_high_252") == True, "sr"),
        StateSpec("new_low_252", "BREAKOUT", pl.col("new_low_252") == True, "sr"),
        StateSpec("hammer", "CANDLE", pl.col("hammer") == True, "candle"),
        StateSpec("bullish_engulfing", "CANDLE", pl.col("bullish_engulfing") == True, "candle"),
        StateSpec("bearish_engulfing", "CANDLE", pl.col("bearish_engulfing") == True, "candle"),
    ]
    # Feature-engine aliases only when the indicator equivalent is absent.
    aliases = [
        StateSpec("feat_rsi14_le_30", "MOMENTUM", pl.col("rsi_14") <= 30, "feature"),
        StateSpec("feat_volume_ratio20_gt_1_5", "VOLUME", pl.col("volume_ratio_20d") > 1.5, "feature"),
        StateSpec("feat_close_above_ma200", "TREND", pl.col("ma_200_distance") > 0, "feature"),
        StateSpec("feat_close_above_ma50", "TREND", pl.col("ma_50_distance") > 0, "feature"),
        StateSpec("feat_close_above_ma20", "TREND", pl.col("ma_20_distance") > 0, "feature"),
    ]
    out: list[StateSpec] = []
    seen: set[str] = set()
    for spec in candidates:
        needed = spec.expr.meta.root_names()
        if any(name not in columns for name in needed):
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    have = {s.name for s in out}
    alias_if_missing = {
        "feat_rsi14_le_30": "rsi14_le_30",
        "feat_volume_ratio20_gt_1_5": "volume_ratio20_gt_1_5",
        "feat_close_above_ma200": "price_above_sma200",
        "feat_close_above_ma50": "price_above_sma50",
        "feat_close_above_ma20": "price_above_sma20",
    }
    for spec in aliases:
        if alias_if_missing.get(spec.name) in have:
            continue
        needed = spec.expr.meta.root_names()
        if any(name not in columns for name in needed):
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(spec)
    return out


def freeze_analysis_quantiles(analysis: pl.DataFrame) -> dict[str, dict[str, float]]:
    frozen: dict[str, dict[str, float]] = {}
    for col, _fam in QUANTILE_SOURCES:
        if col not in analysis.columns:
            continue
        s = analysis[col].drop_nulls()
        if s.len() < 100:
            continue
        preferred = QUANTILE_ALIASES.get(col)
        if preferred and preferred in frozen:
            continue
        frozen[col] = {f"q{int(p * 100)}": float(s.quantile(p)) for p in (0.2, 0.4, 0.6, 0.8)}
    return frozen


def quantile_state_specs(columns: set[str], frozen: dict[str, dict[str, float]]) -> list[StateSpec]:
    out: list[StateSpec] = []
    fam = {c: f for c, f in QUANTILE_SOURCES}
    for col, cuts in frozen.items():
        if col not in columns:
            continue
        family = fam.get(col, "MIXED")
        q20, q80 = cuts.get("q20"), cuts.get("q80")
        if q20 is not None:
            out.append(StateSpec(f"analysis_q_{col}_le_q20", family, pl.col(col) <= q20, "analysis_quantile"))
        if q80 is not None:
            out.append(StateSpec(f"analysis_q_{col}_ge_q80", family, pl.col(col) >= q80, "analysis_quantile"))
    return out


def cross_section_quantile_states(df: pl.DataFrame, col: str, family: str) -> pl.DataFrame:
    """Same-day cross-sectional quantile flags. Distinct from analysis-frozen cutoffs."""
    if col not in df.columns:
        return df
    work = df.with_columns(
        pl.col(col).rank(method="average").over("date").alias("_cs_r"),
        pl.col(col).count().over("date").alias("_cs_n"),
    ).with_columns(((pl.col("_cs_r") - 1) / pl.col("_cs_n")).alias("_cs_p"))
    return work.with_columns(
        (pl.col("_cs_p") <= 0.2).alias(f"cs_q_{col}_le_q20"),
        (pl.col("_cs_p") >= 0.8).alias(f"cs_q_{col}_ge_q80"),
    ).drop(["_cs_r", "_cs_n", "_cs_p"])


def apply_states(df: pl.DataFrame, specs: list[StateSpec]) -> pl.DataFrame:
    if not specs:
        return df
    return df.with_columns([s.expr.alias(s.name) for s in specs])


def family_of(name: str, specs: list[StateSpec]) -> str:
    for s in specs:
        if s.name == name:
            return s.family
    if " AND " in name:
        return "MIXED"
    return FAMILIES[-1]
