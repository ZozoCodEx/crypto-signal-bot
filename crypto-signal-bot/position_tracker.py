"""Persistent lifecycle tracking and analytics for virtual positions only."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-signal-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from requests import RequestException

from binance_client import get_klines
from config import ENTRY_INTERVAL, INITIAL_BALANCE, TRADES_FILE


PORTFOLIO_FILE = "portfolio_summary.csv"
POSITION_COLUMNS = [
    "id",
    "symbol",
    "direction",
    "entry_time",
    "entry_price",
    "current_price",
    "take_profit",
    "stop_loss",
    "status",
    "close_time",
    "close_price",
    "pnl_percent",
    "duration_hours",
    "market_regime",
]
PORTFOLIO_COLUMNS = [
    "timestamp",
    "balance",
    "open_positions",
    "closed_positions",
    "win_rate",
    "profit_factor",
    "max_drawdown",
    "average_trade",
    "return_percent",
]
CLOSED_STATUSES = {"TP_HIT", "SL_HIT", "MANUAL_CLOSE"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _duration_hours(entry_time: Any, end_time: Optional[datetime] = None) -> float:
    entry = pd.to_datetime(entry_time, utc=True, errors="coerce")
    if pd.isna(entry):
        return 0.0
    end = end_time or _now()
    return max(0.0, (end - entry.to_pydatetime()).total_seconds() / 3600)


def _pnl_percent(direction: str, entry: float, current: float) -> float:
    if entry == 0:
        return 0.0
    if direction.upper() == "SHORT":
        return ((entry - current) / entry) * 100
    return ((current - entry) / entry) * 100


def _migrate_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the prior paper-trade schema without discarding old records."""
    rows = []
    for _, old in frame.iterrows():
        result = str(old.get("result", ""))
        old_status = str(old.get("status", ""))
        status = "OPEN" if old_status == "OPEN" else {
            "WIN": "TP_HIT",
            "LOSS": "SL_HIT",
            "EXPIRED": "MANUAL_CLOSE",
        }.get(result, "MANUAL_CLOSE")
        entry = float(old.get("entry", 0) or 0)
        current = float(old.get("current_price", entry) or entry)
        close_price = current if status != "OPEN" else np.nan
        direction = str(old.get("signal", "LONG") or "LONG")
        opened_at = old.get("opened_at", "")
        closed_at = old.get("closed_at", "") if status != "OPEN" else ""
        end = pd.to_datetime(closed_at, utc=True, errors="coerce")
        end_datetime = end.to_pydatetime() if pd.notna(end) else None
        rows.append(
            {
                "id": old.get("trade_id", uuid4().hex),
                "symbol": old.get("symbol", ""),
                "direction": direction,
                "entry_time": opened_at,
                "entry_price": entry,
                "current_price": current,
                "take_profit": old.get("take_profit", np.nan),
                "stop_loss": old.get("stop_loss", np.nan),
                "status": status,
                "close_time": closed_at,
                "close_price": close_price,
                "pnl_percent": _pnl_percent(direction, entry, current),
                "duration_hours": _duration_hours(opened_at, end_datetime),
                "market_regime": "",
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def load_positions() -> pd.DataFrame:
    """Load positions and create or migrate trades.csv to the current schema."""
    path = Path(TRADES_FILE)
    if not path.exists():
        positions = pd.DataFrame(columns=POSITION_COLUMNS)
        save_positions(positions)
        return positions
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    schema_changed = frame.columns.tolist() != POSITION_COLUMNS
    if not schema_changed:
        positions = frame
    elif "trade_id" in frame.columns or "opened_at" in frame.columns:
        positions = _migrate_legacy(frame)
    else:
        positions = frame.copy()
        for column in POSITION_COLUMNS:
            if column not in positions:
                positions[column] = ""
        positions = positions[POSITION_COLUMNS]
    for column in [
        "id",
        "symbol",
        "direction",
        "entry_time",
        "status",
        "close_time",
        "market_regime",
    ]:
        positions[column] = positions[column].fillna("").astype(str)
    if schema_changed:
        save_positions(positions)
    return positions


def save_positions(positions: pd.DataFrame) -> None:
    positions.reindex(columns=POSITION_COLUMNS).to_csv(TRADES_FILE, index=False)


def has_open_position(symbol: str) -> bool:
    positions = load_positions()
    return bool(((positions["symbol"] == symbol) & (positions["status"] == "OPEN")).any())


def open_position(analysis: dict[str, Any], market_regime: str) -> bool:
    """Persist one virtual position unless that symbol is already OPEN."""
    symbol = str(analysis["symbol"])
    if has_open_position(symbol):
        return False
    positions = load_positions()
    entry = float(analysis["entry"])
    row = {
        "id": uuid4().hex,
        "symbol": symbol,
        "direction": str(analysis.get("direction", analysis.get("signal", "LONG"))),
        "entry_time": _now().isoformat(),
        "entry_price": entry,
        "current_price": entry,
        "take_profit": float(analysis["take_profit"]),
        "stop_loss": float(analysis["stop_loss"]),
        "status": "OPEN",
        "close_time": "",
        "close_price": np.nan,
        "pnl_percent": 0.0,
        "duration_hours": 0.0,
        "market_regime": market_regime,
    }
    addition = pd.DataFrame([row], columns=POSITION_COLUMNS)
    positions = addition if positions.empty else pd.concat([positions, addition], ignore_index=True)
    save_positions(positions)
    return True


def update_open_positions() -> pd.DataFrame:
    """Refresh every OPEN position and freeze it when TP or SL is reached."""
    positions = load_positions()
    open_indices = positions.index[positions["status"] == "OPEN"]
    now = _now()
    for index in open_indices:
        symbol = str(positions.at[index, "symbol"])
        try:
            current = float(get_klines(symbol, ENTRY_INTERVAL, 1).iloc[-1]["close"])
        except (RequestException, ValueError, KeyError, IndexError) as error:
            print(f"Could not update position for {symbol}: {error}")
            continue
        direction = str(positions.at[index, "direction"]).upper()
        entry = float(positions.at[index, "entry_price"])
        take_profit = float(positions.at[index, "take_profit"])
        stop_loss = float(positions.at[index, "stop_loss"])
        pnl = _pnl_percent(direction, entry, current)

        positions.at[index, "current_price"] = current
        positions.at[index, "duration_hours"] = _duration_hours(
            positions.at[index, "entry_time"], now
        )
        positions.at[index, "pnl_percent"] = pnl

        tp_hit = current >= take_profit if direction == "LONG" else current <= take_profit
        sl_hit = current <= stop_loss if direction == "LONG" else current >= stop_loss
        if tp_hit or sl_hit:
            positions.at[index, "status"] = "TP_HIT" if tp_hit else "SL_HIT"
            positions.at[index, "close_time"] = now.isoformat()
            positions.at[index, "close_price"] = current
            # pnl_percent is now frozen because only OPEN rows are refreshed.
    save_positions(positions)
    return positions


def get_portfolio_stats(positions: Optional[pd.DataFrame] = None) -> dict[str, float]:
    """Return balance, lifecycle counts, and closed-position performance."""
    positions = load_positions() if positions is None else positions
    if positions.empty:
        open_positions = positions
        closed = positions
    else:
        open_positions = positions[positions["status"] == "OPEN"]
        closed = positions[positions["status"].isin(CLOSED_STATUSES)]
    returns = pd.to_numeric(closed.get("pnl_percent", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
    balances = [float(INITIAL_BALANCE)]
    for value in returns:
        balances.append(balances[-1] * (1 + float(value) / 100))
    equity = np.asarray(balances)
    peaks = np.maximum.accumulate(equity)
    max_drawdown = float(((equity - peaks) / peaks * 100).min())
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    return {
        "Balance": balances[-1],
        "Open trades": len(open_positions),
        "Closed trades": len(closed),
        "Wins": len(wins),
        "Losses": len(losses),
        "Win rate": len(wins) / len(closed) * 100 if len(closed) else 0.0,
        "Return": (balances[-1] / INITIAL_BALANCE - 1) * 100,
        "Profit factor": gross_profit / gross_loss if gross_loss else (
            float("inf") if gross_profit else 0.0
        ),
        "Max DD": max_drawdown,
        "Average trade": float(returns.mean()) if len(returns) else 0.0,
    }


def save_portfolio_summary(positions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    stats = get_portfolio_stats(positions)
    row = {
        "timestamp": _now().isoformat(),
        "balance": stats["Balance"],
        "open_positions": stats["Open trades"],
        "closed_positions": stats["Closed trades"],
        "win_rate": stats["Win rate"],
        "profit_factor": stats["Profit factor"],
        "max_drawdown": stats["Max DD"],
        "average_trade": stats["Average trade"],
        "return_percent": stats["Return"],
    }
    path = Path(PORTFOLIO_FILE)
    try:
        history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    except pd.errors.EmptyDataError:
        history = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    row_frame = pd.DataFrame([row], columns=PORTFOLIO_COLUMNS)
    history = row_frame if history.empty else pd.concat([history, row_frame], ignore_index=True)
    history.reindex(columns=PORTFOLIO_COLUMNS).to_csv(path, index=False)
    return history


def _save_chart(filename: str, title: str, draw) -> None:
    plt.figure(facecolor="white")
    axis = plt.gca()
    axis.set_facecolor("white")
    draw(axis)
    axis.set_title(title)
    axis.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor="white")
    plt.close()


def generate_portfolio_charts(positions: Optional[pd.DataFrame] = None) -> None:
    positions = load_positions() if positions is None else positions
    closed = positions[positions["status"].isin(CLOSED_STATUSES)] if not positions.empty else positions
    opened = positions[positions["status"] == "OPEN"] if not positions.empty else positions
    closed_returns = pd.to_numeric(closed.get("pnl_percent", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
    balances = [float(INITIAL_BALANCE)]
    for value in closed_returns:
        balances.append(balances[-1] * (1 + float(value) / 100))

    def draw_equity(axis):
        axis.plot(range(len(balances)), balances)
        if not len(closed_returns):
            axis.text(
                0.5,
                0.5,
                "No closed positions",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel("Closed trade number")
        axis.set_ylabel("Balance")

    def draw_open(axis):
        if opened.empty:
            axis.text(
                0.5,
                0.5,
                "No open positions",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            pnl = pd.to_numeric(opened["pnl_percent"], errors="coerce").fillna(0)
            axis.bar(opened["symbol"], pnl)
        axis.set_xlabel("Symbol")
        axis.set_ylabel("PnL (%)")

    def draw_distribution(axis):
        if len(closed_returns):
            axis.hist(closed_returns)
        else:
            axis.text(
                0.5,
                0.5,
                "No closed positions",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel("Trade PnL (%)")
        axis.set_ylabel("Frequency")

    _save_chart("portfolio_equity.png", "Paper Portfolio Equity", draw_equity)
    _save_chart("portfolio_open_positions.png", "Open Position PnL", draw_open)
    _save_chart("portfolio_pnl_distribution.png", "Closed Trade PnL Distribution", draw_distribution)


def update_portfolio_analytics(positions: Optional[pd.DataFrame] = None) -> dict[str, float]:
    positions = load_positions() if positions is None else positions
    save_portfolio_summary(positions)
    generate_portfolio_charts(positions)
    return get_portfolio_stats(positions)


def print_portfolio_stats(stats: dict[str, float]) -> None:
    print("\nPosition Tracker Summary\n")
    print(f"Balance: ${stats['Balance']:.2f}")
    print(f"Open trades: {int(stats['Open trades'])}")
    print(f"Closed trades: {int(stats['Closed trades'])}")
    print(f"Wins: {int(stats['Wins'])}")
    print(f"Losses: {int(stats['Losses'])}")
    print(f"Win rate: {stats['Win rate']:.1f}%")
    print(f"Return: {stats['Return']:+.2f}%")
    factor = "∞" if np.isinf(stats["Profit factor"]) else f"{stats['Profit factor']:.2f}"
    print(f"Profit factor: {factor}")
    print(f"Max DD: {stats['Max DD']:.1f}%")
