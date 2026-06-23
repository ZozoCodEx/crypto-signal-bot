"""Persistent CSV logging for every market-analysis decision."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import SIGNALS_FILE


SIGNAL_COLUMNS = [
    "timestamp",
    "symbol",
    "signal",
    "confidence",
    "price",
    "atr",
    "ema50",
    "ema200",
    "rsi",
    "volume_ratio",
    "trend",
    "trend_score",
    "momentum_score",
    "volume_score",
    "market_score",
    "reasons",
]


def load_signals() -> pd.DataFrame:
    """Load the signal history, creating its CSV file when needed."""
    signals_path = Path(SIGNALS_FILE)
    if not signals_path.exists():
        signals = pd.DataFrame(columns=SIGNAL_COLUMNS)
        signals.to_csv(signals_path, index=False)
        return signals

    try:
        signals = pd.read_csv(signals_path)
    except pd.errors.EmptyDataError:
        signals = pd.DataFrame(columns=SIGNAL_COLUMNS)

    for column in SIGNAL_COLUMNS:
        if column not in signals.columns:
            signals[column] = ""
    return signals[SIGNAL_COLUMNS]


def save_signal(analysis: dict[str, Any]) -> bool:
    """Append one analysis unless that symbol was logged in the same UTC minute."""
    signals = load_signals()
    timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if not signals.empty:
        logged_minutes = pd.to_datetime(
            signals["timestamp"], errors="coerce", utc=True
        ).dt.floor("min")
        duplicate = (signals["symbol"] == analysis["symbol"]) & (
            logged_minutes == pd.Timestamp(timestamp)
        )
        if duplicate.any():
            return False

    row = {
        "timestamp": timestamp.isoformat(),
        "symbol": analysis["symbol"],
        "signal": analysis["signal"],
        "confidence": analysis["confidence"],
        "price": analysis["price"],
        "atr": analysis["atr"],
        "ema50": analysis["ema50"],
        "ema200": analysis["ema200"],
        "rsi": analysis["rsi"],
        "volume_ratio": analysis["volume_ratio"],
        "trend": analysis["trend"],
        "trend_score": analysis["trend_score"],
        "momentum_score": analysis["momentum_score"],
        "volume_score": analysis["volume_score"],
        "market_score": analysis["market_score"],
        "reasons": " | ".join(analysis["reason"]),
    }
    row_frame = pd.DataFrame([row], columns=SIGNAL_COLUMNS)
    signals = row_frame if signals.empty else pd.concat([signals, row_frame], ignore_index=True)
    signals.to_csv(SIGNALS_FILE, index=False)
    return True
