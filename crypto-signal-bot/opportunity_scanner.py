"""Cross-market opportunity scoring using public Binance data only."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from binance_client import get_klines
from config import (
    CANDLE_LIMIT,
    ENTRY_INTERVAL,
    MACRO_INTERVAL,
    TOP_SYMBOLS,
    TREND_INTERVAL,
)
from indicators import add_indicators


OPPORTUNITIES_FILE = "opportunities.csv"
OPPORTUNITY_COLUMNS = [
    "timestamp",
    "symbol",
    "price",
    "score",
    "label",
    "trend_4h",
    "trend_1h",
    "ema50",
    "ema200",
    "rsi",
    "volume_ratio",
    "atr",
    "atr_avg",
    "reasons",
]


def _label(score: int) -> str:
    if score >= 90:
        return "🔥 EXCEPTIONAL"
    if score >= 80:
        return "🚀 STRONG"
    if score >= 70:
        return "✅ GOOD"
    if score >= 60:
        return "⚠ WATCH"
    return "IGNORE"


def score_opportunity(
    symbol: str,
    entry: Any,
    trend: Any,
    macro: Any,
    timestamp: Optional[datetime] = None,
) -> dict[str, Any]:
    """Score one prepared 4H/1H/15m market snapshot from 0 to 100."""
    macro_bullish = bool(macro["close"] > macro["ema200"])
    trend_bullish = bool(
        trend["close"] > trend["ema200"]
        and trend["ema50"] > trend["ema200"]
    )
    rsi = float(entry["rsi14"])
    volume_ratio = float(entry["volume_ratio"])
    atr = float(entry["atr14"])
    atr_avg = float(entry["atr_avg"])

    macro_points = 25 if macro_bullish else 0
    trend_points = 20 if trend_bullish else 0

    if 45 <= rsi <= 60:
        momentum_points = 20
        momentum_reason = f"✅ RSI healthy ({rsi:.1f})"
    elif 40 <= rsi < 45:
        momentum_points = 10
        momentum_reason = f"⚠ RSI slightly weak ({rsi:.1f})"
    elif 60 < rsi <= 70:
        momentum_points = 5
        momentum_reason = f"⚠ RSI elevated ({rsi:.1f})"
    else:
        momentum_points = 0
        momentum_reason = f"❌ RSI outside opportunity range ({rsi:.1f})"

    if volume_ratio > 1.5:
        volume_points = 20
        volume_reason = f"✅ Strong volume expansion ({volume_ratio:.2f}x)"
    elif volume_ratio > 1.2:
        volume_points = 15
        volume_reason = f"✅ Volume above average ({volume_ratio:.2f}x)"
    elif volume_ratio > 1:
        volume_points = 10
        volume_reason = f"⚠ Moderate volume ({volume_ratio:.2f}x)"
    else:
        volume_points = 0
        volume_reason = f"❌ Volume below average ({volume_ratio:.2f}x)"

    atr_expanding = bool(pd.notna(atr_avg) and atr > atr_avg)
    atr_points = 15 if atr_expanding else 5
    total = macro_points + trend_points + momentum_points + volume_points + atr_points
    reasons = [
        "✅ Strong 4H trend" if macro_bullish else "❌ Weak 4H trend",
        "✅ Strong 1H trend" if trend_bullish else "❌ Weak 1H trend",
        momentum_reason,
        volume_reason,
        (
            f"✅ ATR expanding ({atr:.4f} > {atr_avg:.4f})"
            if atr_expanding
            else f"⚠ ATR not expanding ({atr:.4f} ≤ {atr_avg:.4f})"
        ),
    ]
    observed_at = timestamp or datetime.now(timezone.utc)

    return {
        "timestamp": observed_at.isoformat(),
        "symbol": symbol,
        "price": float(entry["close"]),
        "score": int(total),
        "label": _label(total),
        "trend_4h": "BULLISH" if macro_bullish else "BEARISH",
        "trend_1h": "BULLISH" if trend_bullish else "BEARISH",
        "ema50": float(entry["ema50"]),
        "ema200": float(entry["ema200"]),
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "atr": atr,
        "atr_avg": atr_avg,
        "reasons": reasons,
    }


def analyze_opportunity(symbol: str) -> dict[str, Any]:
    """Fetch and score the current multi-timeframe snapshot for one symbol."""
    now = pd.Timestamp.now(tz="UTC")

    def latest_completed(interval: str) -> Any:
        data = add_indicators(get_klines(symbol, interval, CANDLE_LIMIT))
        completed = data[data["close_time"] <= now]
        if completed.empty:
            raise ValueError(f"No completed {interval} candles for {symbol}")
        return completed.iloc[-1]

    entry = latest_completed(ENTRY_INTERVAL)
    trend = latest_completed(TREND_INTERVAL)
    macro = latest_completed(MACRO_INTERVAL)
    return score_opportunity(symbol, entry, trend, macro)


def scan_opportunities(
    symbols: Sequence[str] = TOP_SYMBOLS,
) -> pd.DataFrame:
    """Analyze symbols concurrently and return them sorted by score."""
    rows = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_opportunity, symbol): symbol for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                print(f"Could not analyze {symbol}: {error}")

    if not rows:
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["score", "volume_ratio"], ascending=[False, False]
    ).reset_index(drop=True)


def save_opportunities(opportunities: pd.DataFrame) -> None:
    """Save the current ranked market snapshot, replacing the prior snapshot."""
    export = opportunities.copy()
    if not export.empty:
        export["reasons"] = export["reasons"].apply(
            lambda values: " | ".join(values) if isinstance(values, list) else values
        )
    export.reindex(columns=OPPORTUNITY_COLUMNS).to_csv(
        OPPORTUNITIES_FILE, index=False
    )


def top_opportunities(
    opportunities: Optional[pd.DataFrame] = None, limit: int = 5
) -> list[str]:
    """Return the symbols for the highest-ranked opportunities."""
    if opportunities is None:
        path = Path(OPPORTUNITIES_FILE)
        opportunities = (
            pd.read_csv(path)
            if path.exists()
            else scan_opportunities()
        )
    ranked = opportunities.sort_values(
        ["score", "volume_ratio"], ascending=[False, False]
    )
    return ranked.head(limit)["symbol"].tolist()
