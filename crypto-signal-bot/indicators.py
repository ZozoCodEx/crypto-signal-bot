"""Transparent pandas implementations of the strategy indicators."""

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with EMA, RSI, ATR, and volume indicators added."""
    required = {"high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = df.copy()
    result["ema50"] = result["close"].ewm(span=50, adjust=False).mean()
    result["ema200"] = result["close"].ewm(span=200, adjust=False).mean()

    delta = result["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["rsi14"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "rsi14"] = 100.0
    result.loc[(average_loss == 0) & (average_gain == 0), "rsi14"] = 50.0

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr14"] = true_range.rolling(14).mean()
    result["atr_avg"] = result["atr14"].rolling(20).mean()

    # ADX14 with Wilder smoothing. Directional movement is kept explicit so
    # the trend-strength calculation remains inspectable and TA-Lib-free.
    upward_move = result["high"].diff()
    downward_move = -result["low"].diff()
    plus_dm = pd.Series(
        np.where(
            (upward_move > downward_move) & (upward_move > 0), upward_move, 0.0
        ),
        index=result.index,
    )
    minus_dm = pd.Series(
        np.where(
            (downward_move > upward_move) & (downward_move > 0), downward_move, 0.0
        ),
        index=result.index,
    )
    wilder_true_range = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    plus_di = 100 * smoothed_plus_dm / wilder_true_range
    minus_di = 100 * smoothed_minus_dm / wilder_true_range
    directional_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / directional_sum
    result["adx14"] = dx.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    # Shift the average by one candle so the current candle is compared with
    # the preceding 20 candles rather than partly with itself.
    result["volume_avg_20"] = result["volume"].rolling(20).mean().shift(1)
    result["volume_ratio"] = result["volume"] / result["volume_avg_20"]
    return result
