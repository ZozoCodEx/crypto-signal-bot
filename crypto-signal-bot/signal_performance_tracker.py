"""Automated 24h, 48h, and 7d performance tracking for opportunities."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from requests import RequestException

from binance_client import get_klines
from config import ENTRY_INTERVAL


HISTORY_FILE = "signal_history.csv"
SUMMARY_FILE = "signal_performance_summary.csv"
HISTORY_COLUMNS = [
    "signal_id",
    "created_at",
    "symbol",
    "market_regime",
    "signal",
    "score",
    "entry_price",
    "rsi",
    "volume_ratio",
    "atr",
    "status",
    "price_24h",
    "pnl_24h",
    "result_24h",
    "price_48h",
    "pnl_48h",
    "result_48h",
    "price_7d",
    "pnl_7d",
    "result_7d",
]
SUMMARY_COLUMNS = [
    "timestamp",
    "total_signals",
    "evaluated_24h",
    "win_rate_24h",
    "avg_pnl_24h",
    "evaluated_48h",
    "win_rate_48h",
    "avg_pnl_48h",
    "evaluated_7d",
    "win_rate_7d",
    "avg_pnl_7d",
    "best_signal",
    "worst_signal",
]
VALID_SIGNALS = {"BUY", "LONG", "SELL", "SHORT"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_elite(opportunity: Any) -> bool:
    return bool(
        opportunity["score"] >= 90
        and opportunity["volume_ratio"] > 1.5
        and opportunity["atr"] > opportunity["atr_avg"]
        and 48 <= opportunity["rsi"] <= 58
    )


def load_signal_history() -> pd.DataFrame:
    path = Path(HISTORY_FILE)
    if not path.exists():
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
        history.to_csv(path, index=False)
        return history
    try:
        history = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
    for column in HISTORY_COLUMNS:
        if column not in history:
            history[column] = ""
    history = history[HISTORY_COLUMNS]
    for column in [
        "signal_id",
        "created_at",
        "symbol",
        "market_regime",
        "signal",
        "status",
        "result_24h",
        "result_48h",
        "result_7d",
    ]:
        history[column] = history[column].fillna("").astype(str)
    return history


def _latest_regime() -> str:
    path = Path("market_regime.csv")
    if not path.exists():
        return "UNKNOWN"
    frame = pd.read_csv(path)
    return str(frame.iloc[-1]["regime"]) if not frame.empty else "UNKNOWN"


def log_current_opportunities(
    opportunities: Optional[pd.DataFrame] = None,
    market_regime: Optional[str] = None,
) -> pd.DataFrame:
    """Append unseen current opportunities using the required compound key."""
    if opportunities is None:
        path = Path("opportunities.csv")
        opportunities = pd.read_csv(path) if path.exists() else pd.DataFrame()
    history = load_signal_history()
    if opportunities.empty:
        return history

    regime = market_regime or _latest_regime()
    created = _now().replace(minute=0, second=0, microsecond=0)
    existing_hours = pd.to_datetime(
        history["created_at"], errors="coerce", utc=True
    ).dt.floor("h")
    additions = []
    for _, opportunity in opportunities.iterrows():
        signal = "LONG" if _is_elite(opportunity) else "IGNORE"
        entry_price = float(opportunity["price"])
        duplicate = (
            (history["symbol"] == opportunity["symbol"])
            & (existing_hours == pd.Timestamp(created))
            & (history["signal"] == signal)
            & np.isclose(
                pd.to_numeric(history["entry_price"], errors="coerce"),
                entry_price,
                rtol=0,
                atol=1e-12,
                equal_nan=False,
            )
        )
        if duplicate.any():
            continue
        additions.append(
            {
                "signal_id": uuid4().hex,
                "created_at": created.isoformat(),
                "symbol": opportunity["symbol"],
                "market_regime": regime,
                "signal": signal,
                "score": int(opportunity["score"]),
                "entry_price": entry_price,
                "rsi": float(opportunity["rsi"]),
                "volume_ratio": float(opportunity["volume_ratio"]),
                "atr": float(opportunity["atr"]),
                "status": "TRACKING" if signal in VALID_SIGNALS else "IGNORE",
                "price_24h": np.nan,
                "pnl_24h": np.nan,
                "result_24h": "",
                "price_48h": np.nan,
                "pnl_48h": np.nan,
                "result_48h": "",
                "price_7d": np.nan,
                "pnl_7d": np.nan,
                "result_7d": "",
            }
        )
    if additions:
        addition = pd.DataFrame(additions, columns=HISTORY_COLUMNS)
        history = addition if history.empty else pd.concat([history, addition], ignore_index=True)
        history.to_csv(HISTORY_FILE, index=False)
    return history


def _pnl(signal: str, entry: float, current: float) -> float:
    if signal in {"SELL", "SHORT"}:
        return ((entry - current) / entry) * 100
    return ((current - entry) / entry) * 100


def update_signal_performance(history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Fill each due horizon once, using one latest public price per symbol."""
    history = load_signal_history() if history is None else history.copy()
    if history.empty:
        return history
    created = pd.to_datetime(history["created_at"], errors="coerce", utc=True)
    ages = (_now() - created).dt.total_seconds() / 3600
    valid = history["signal"].isin(VALID_SIGNALS)
    horizon_specs = [
        ("24h", 24, "price_24h", "pnl_24h", "result_24h"),
        ("48h", 48, "price_48h", "pnl_48h", "result_48h"),
        ("7d", 24 * 7, "price_7d", "pnl_7d", "result_7d"),
    ]
    due_by_symbol: dict[str, list[tuple[int, tuple[str, int, str, str, str]]]] = {}
    for index in history.index[valid]:
        for spec in horizon_specs:
            _, minimum_age, price_column, _, _ = spec
            if ages.at[index] >= minimum_age and pd.isna(history.at[index, price_column]):
                due_by_symbol.setdefault(str(history.at[index, "symbol"]), []).append(
                    (index, spec)
                )

    for symbol, due_items in due_by_symbol.items():
        try:
            price = float(get_klines(symbol, ENTRY_INTERVAL, 1).iloc[-1]["close"])
        except (RequestException, ValueError, KeyError, IndexError) as error:
            print(f"Could not evaluate signals for {symbol}: {error}")
            continue
        for index, spec in due_items:
            horizon, _, price_column, pnl_column, result_column = spec
            pnl = _pnl(
                str(history.at[index, "signal"]),
                float(history.at[index, "entry_price"]),
                price,
            )
            history.at[index, price_column] = price
            history.at[index, pnl_column] = pnl
            history.at[index, result_column] = "WIN" if pnl > 0 else "LOSS"
            if horizon == "7d":
                history.at[index, "status"] = "COMPLETE"
            else:
                history.at[index, "status"] = f"EVALUATED_{horizon.upper()}"
    history.to_csv(HISTORY_FILE, index=False)
    return history


def _horizon_metrics(history: pd.DataFrame, suffix: str) -> tuple[int, float, float]:
    valid = history[history["signal"].isin(VALID_SIGNALS)]
    pnl = pd.to_numeric(valid[f"pnl_{suffix}"], errors="coerce").dropna()
    results = valid.loc[pnl.index, f"result_{suffix}"]
    evaluated = len(pnl)
    win_rate = (results == "WIN").sum() / evaluated * 100 if evaluated else 0.0
    return evaluated, float(win_rate), float(pnl.mean()) if evaluated else 0.0


def _best_and_worst(history: pd.DataFrame) -> tuple[str, str]:
    valid = history[history["signal"].isin(VALID_SIGNALS)]
    observations = []
    for index, row in valid.iterrows():
        for suffix in ("7d", "48h", "24h"):
            value = pd.to_numeric(pd.Series([row[f"pnl_{suffix}"]]), errors="coerce").iloc[0]
            if pd.notna(value):
                observations.append((str(row["symbol"]), float(value)))
                break
    if not observations:
        return "N/A", "N/A"
    best = max(observations, key=lambda item: item[1])
    worst = min(observations, key=lambda item: item[1])
    return f"{best[0]} {best[1]:+.2f}%", f"{worst[0]} {worst[1]:+.2f}%"


def append_performance_summary(history: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    history = load_signal_history() if history is None else history
    metrics_24h = _horizon_metrics(history, "24h")
    metrics_48h = _horizon_metrics(history, "48h")
    metrics_7d = _horizon_metrics(history, "7d")
    best, worst = _best_and_worst(history)
    row = {
        "timestamp": _now().isoformat(),
        # Total tracking includes every scanner observation. Horizon metrics
        # below remain restricted to VALID_SIGNALS, so IGNORE never affects
        # trade win rate or PnL statistics.
        "total_signals": int(len(history)),
        "evaluated_24h": metrics_24h[0],
        "win_rate_24h": metrics_24h[1],
        "avg_pnl_24h": metrics_24h[2],
        "evaluated_48h": metrics_48h[0],
        "win_rate_48h": metrics_48h[1],
        "avg_pnl_48h": metrics_48h[2],
        "evaluated_7d": metrics_7d[0],
        "win_rate_7d": metrics_7d[1],
        "avg_pnl_7d": metrics_7d[2],
        "best_signal": best,
        "worst_signal": worst,
    }
    path = Path(SUMMARY_FILE)
    try:
        summary = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    except pd.errors.EmptyDataError:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    addition = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    summary = addition if summary.empty else pd.concat([summary, addition], ignore_index=True)
    summary.to_csv(path, index=False)
    return row


def run_signal_performance_tracker(
    opportunities: Optional[pd.DataFrame] = None,
    market_regime: Optional[str] = None,
) -> dict[str, Any]:
    history = log_current_opportunities(opportunities, market_regime)
    history = update_signal_performance(history)
    return append_performance_summary(history)
