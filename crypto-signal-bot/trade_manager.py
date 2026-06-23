"""CSV-backed paper-trade lifecycle management.

This module only records virtual trades. It never authenticates with Binance and
contains no order-placement functionality.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from requests import RequestException

from binance_client import get_klines
from config import (
    MAX_OPEN_TRADES_PER_SYMBOL,
    ENTRY_INTERVAL,
    TRADES_FILE,
    TRADE_EXPIRATION_HOURS,
)


TRADE_COLUMNS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "symbol",
    "signal",
    "entry",
    "atr",
    "stop_loss",
    "take_profit",
    "current_price",
    "status",
    "result",
    "profit_pct",
    "confidence",
    "reason",
]


def load_trades() -> pd.DataFrame:
    """Load saved paper trades, creating an empty CSV when necessary."""
    trades_path = Path(TRADES_FILE)
    if not trades_path.exists():
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        save_trades(trades)
        return trades

    try:
        trades = pd.read_csv(trades_path)
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)

    # Add any columns introduced by future versions while retaining old data.
    schema_changed = False
    for column in TRADE_COLUMNS:
        if column not in trades.columns:
            trades[column] = ""
            schema_changed = True
    trades = trades[TRADE_COLUMNS]
    if schema_changed:
        save_trades(trades)
    return trades


def save_trades(df: pd.DataFrame) -> None:
    """Persist paper trades using the stable CSV column order."""
    df.reindex(columns=TRADE_COLUMNS).to_csv(TRADES_FILE, index=False)


def has_open_trade(symbol: str) -> bool:
    """Return whether the configured open-trade limit is met for *symbol*."""
    trades = load_trades()
    open_count = ((trades["symbol"] == symbol) & (trades["status"] == "OPEN")).sum()
    return int(open_count) >= MAX_OPEN_TRADES_PER_SYMBOL


def open_trade(analysis: dict[str, Any]) -> bool:
    """Open and save a virtual LONG trade when duplicate protection allows it."""
    if analysis.get("signal") != "LONG":
        return False

    symbol = str(analysis["symbol"])
    if has_open_trade(symbol):
        return False

    trades = load_trades()
    trade = {
        "trade_id": uuid4().hex,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "closed_at": "",
        "symbol": symbol,
        "signal": analysis["signal"],
        "entry": float(analysis["entry"]),
        "atr": float(analysis["atr"]),
        "stop_loss": float(analysis["stop_loss"]),
        "take_profit": float(analysis["take_profit"]),
        "current_price": float(analysis["price"]),
        "status": "OPEN",
        "result": "",
        "profit_pct": 0.0,
        "confidence": int(analysis["confidence"]),
        "reason": "; ".join(str(reason) for reason in analysis.get("reason", [])),
    }
    if trades.empty:
        trades = pd.DataFrame([trade], columns=TRADE_COLUMNS)
    else:
        trades = pd.concat([trades, pd.DataFrame([trade])], ignore_index=True)
    save_trades(trades)
    return True


def update_open_trades() -> None:
    """Refresh prices and close virtual trades at TP, SL, or expiration."""
    trades = load_trades()
    open_indices = trades.index[trades["status"] == "OPEN"]
    if open_indices.empty:
        return

    now = datetime.now(timezone.utc)
    for index in open_indices:
        symbol = str(trades.at[index, "symbol"])
        try:
            current_price = float(
                get_klines(symbol, ENTRY_INTERVAL, 1).iloc[-1]["close"]
            )
        except (RequestException, ValueError, KeyError, IndexError) as error:
            print(f"Could not update open paper trade for {symbol}: {error}")
            continue

        entry = float(trades.at[index, "entry"])
        stop_loss = float(trades.at[index, "stop_loss"])
        take_profit = float(trades.at[index, "take_profit"])
        opened_at = pd.to_datetime(trades.at[index, "opened_at"], utc=True)
        age_hours = (now - opened_at.to_pydatetime()).total_seconds() / 3600

        trades.at[index, "current_price"] = current_price
        result = ""
        exit_price = current_price

        if current_price >= take_profit:
            result = "WIN"
            exit_price = take_profit
        elif current_price <= stop_loss:
            result = "LOSS"
            exit_price = stop_loss
        elif age_hours > TRADE_EXPIRATION_HOURS:
            result = "EXPIRED"

        if result:
            trades.at[index, "status"] = "CLOSED"
            trades.at[index, "result"] = result
            trades.at[index, "closed_at"] = now.isoformat()
            trades.at[index, "profit_pct"] = ((exit_price - entry) / entry) * 100

    save_trades(trades)


def print_trade_summary() -> None:
    """Print aggregate paper-trading statistics from the CSV ledger."""
    trades = load_trades()
    total = len(trades)
    open_count = int((trades["status"] == "OPEN").sum())
    closed_count = int((trades["status"] == "CLOSED").sum())
    wins = int((trades["result"] == "WIN").sum())
    losses = int((trades["result"] == "LOSS").sum())
    expired = int((trades["result"] == "EXPIRED").sum())
    win_rate = (wins / closed_count * 100) if closed_count else 0.0
    profit_values = pd.to_numeric(trades["profit_pct"], errors="coerce").fillna(0)

    print("\nPaper Trade Summary\n")
    print(f"Total trades: {total}")
    print(f"Open trades: {open_count}")
    print(f"Closed trades: {closed_count}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Expired: {expired}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total profit percentage: {profit_values.sum():.2f}%")
