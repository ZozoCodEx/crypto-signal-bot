"""Signal-generation rules for the paper-trading bot."""

from typing import Any

from binance_client import get_klines
from config import (
    CANDLE_LIMIT,
    ENTRY_INTERVAL,
    MACRO_INTERVAL,
    MIN_MARKET_SCORE,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    TREND_INTERVAL,
)
from indicators import add_indicators


def analyze_market(
    symbol: str, current: Any, trend: Any, macro: Any
) -> dict[str, Any]:
    """Apply the 4H, 1H, and 15m strategy to prepared candle rows."""
    conditions = {
        "macro": bool(macro["close"] > macro["ema200"]),
        "trend_price": bool(trend["close"] > trend["ema200"]),
        "trend_ema": bool(trend["ema50"] > trend["ema200"]),
        "entry_ema": bool(current["ema50"] > current["ema200"]),
        "rsi": bool(45 <= current["rsi14"] <= 60),
        "volume": bool(current["volume_ratio"] > 1.2),
    }

    rsi = float(current["rsi14"])
    volume_ratio = float(current["volume_ratio"])
    trend_bullish = conditions["macro"] and conditions["trend_price"]
    trend_label = "UPTREND" if trend_bullish else "DOWNTREND"

    trend_score = 10 if trend_bullish else 3
    if 45 <= rsi <= 60:
        momentum_score = 10
    elif 35 <= rsi < 45:
        momentum_score = 7 if rsi >= 40 else 2
    elif 60 < rsi <= 70:
        momentum_score = 4
    else:
        momentum_score = 2

    if volume_ratio > 1.5:
        volume_score = 10
    elif volume_ratio > 1.2:
        volume_score = 8
    elif volume_ratio > 1:
        volume_score = 5
    else:
        volume_score = 2

    market_score = round((trend_score + momentum_score + volume_score) / 3, 1)
    conditions["market_score"] = market_score >= MIN_MARKET_SCORE

    reasons = [
        (
            "✅ 4H macro trend bullish"
            if conditions["macro"]
            else "❌ 4H trend below EMA200"
        ),
        (
            "✅ 1H price trend confirmed"
            if conditions["trend_price"]
            else "❌ 1H price below EMA200"
        ),
        (
            "✅ 1H EMA50 above EMA200"
            if conditions["trend_ema"]
            else "❌ 1H EMA50 below EMA200"
        ),
        (
            "✅ 15m EMA50 above EMA200"
            if conditions["entry_ema"]
            else "❌ 15m EMA50 below EMA200"
        ),
        (
            f"✅ Healthy RSI ({rsi:.1f})"
            if conditions["rsi"]
            else (
                f"❌ RSI above optimal range ({rsi:.1f})"
                if rsi > 60
                else f"❌ RSI below optimal range ({rsi:.1f})"
            )
        ),
        (
            f"✅ Volume above average ({volume_ratio:.2f}x)"
            if conditions["volume"]
            else (
                f"❌ Volume below average ({volume_ratio:.2f}x)"
                if volume_ratio <= 1
                else f"❌ Volume not strong enough ({volume_ratio:.2f}x; needs >1.20x)"
            )
        ),
        (
            f"✅ Market score meets threshold ({market_score:.1f})"
            if conditions["market_score"]
            else f"❌ Market score below {MIN_MARKET_SCORE:.1f} ({market_score:.1f})"
        ),
    ]

    confidence = min(
        95,
        50
        + (10 if conditions["macro"] else 0)
        + (10 if conditions["trend_price"] else 0)
        + (5 if conditions["trend_ema"] else 0)
        + (5 if conditions["entry_ema"] else 0)
        + (10 if conditions["rsi"] else 0)
        + (10 if conditions["volume"] else 0),
    )

    signal = "LONG" if all(conditions.values()) else "WAIT"
    price = float(current["close"])
    atr = float(current["atr14"])
    is_long = signal == "LONG"

    return {
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "ema50": float(current["ema50"]),
        "ema200": float(current["ema200"]),
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "trend": trend_label,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "market_score": market_score,
        "confidence": confidence,
        "atr": atr,
        "entry": price if is_long else None,
        "stop_loss": price - atr * SL_ATR_MULTIPLIER if is_long else None,
        "take_profit": price + atr * TP_ATR_MULTIPLIER if is_long else None,
        "risk_reward": 2 if is_long else None,
        "reason": reasons,
    }


def analyze_symbol(symbol: str) -> dict[str, Any]:
    """Fetch current market data and return a LONG or WAIT signal."""
    signal_df = add_indicators(get_klines(symbol, ENTRY_INTERVAL, CANDLE_LIMIT))
    trend_df = add_indicators(get_klines(symbol, TREND_INTERVAL, CANDLE_LIMIT))
    macro_df = add_indicators(get_klines(symbol, MACRO_INTERVAL, CANDLE_LIMIT))
    return analyze_market(
        symbol, signal_df.iloc[-1], trend_df.iloc[-1], macro_df.iloc[-1]
    )
