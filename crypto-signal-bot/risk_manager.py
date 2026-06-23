"""Portfolio-level capital protection for virtual paper positions."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-signal-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS,
    MAX_OPEN_TRADES,
    MAX_PORTFOLIO_EXPOSURE,
    RISK_PER_TRADE,
)
from position_tracker import CLOSED_STATUSES, get_portfolio_stats, load_positions


RISK_REPORT_FILE = "risk_report.csv"
RISK_COLUMNS = [
    "timestamp",
    "balance",
    "daily_pnl",
    "daily_loss_percent",
    "open_trades",
    "exposure",
    "consecutive_losses",
    "risk_allowed",
    "reason",
]


def calculate_position_size(balance: float, entry: float, stop_loss: float) -> float:
    """Return asset units sized so the stop risks exactly RISK_PER_TRADE."""
    stop_distance = abs(float(entry) - float(stop_loss))
    if balance <= 0 or stop_distance <= 0:
        return 0.0
    risk_amount = float(balance) * RISK_PER_TRADE
    return risk_amount / stop_distance


def _closed_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return positions
    return positions[positions["status"].isin(CLOSED_STATUSES)].copy()


def _daily_performance(positions: pd.DataFrame, balance: float) -> tuple[float, float]:
    closed = _closed_positions(positions)
    if closed.empty:
        return 0.0, 0.0
    dates = pd.to_datetime(closed["close_time"], utc=True, errors="coerce").dt.date
    today = datetime.now(timezone.utc).date()
    today_returns = pd.to_numeric(
        closed.loc[dates == today, "pnl_percent"], errors="coerce"
    ).fillna(0)
    daily_return_fraction = float(today_returns.sum()) / 100
    return balance * daily_return_fraction, max(0.0, -daily_return_fraction)


def _consecutive_losses(positions: pd.DataFrame) -> int:
    closed = _closed_positions(positions)
    if closed.empty:
        return 0
    ordered = closed.assign(
        _closed_at=pd.to_datetime(closed["close_time"], utc=True, errors="coerce")
    ).sort_values("_closed_at", ascending=False)
    count = 0
    for pnl in pd.to_numeric(ordered["pnl_percent"], errors="coerce").fillna(0):
        if pnl < 0:
            count += 1
        else:
            break
    return count


def _current_exposure(positions: pd.DataFrame, balance: float) -> float:
    if positions.empty or balance <= 0:
        return 0.0
    opened = positions[positions["status"] == "OPEN"]
    notional = 0.0
    for _, position in opened.iterrows():
        entry = float(position["entry_price"])
        units = calculate_position_size(balance, entry, float(position["stop_loss"]))
        notional += units * entry
    return notional / balance


def get_risk_status(
    symbol: str = "",
    entry: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> dict[str, Any]:
    """Calculate current limits and optionally include a proposed position."""
    positions = load_positions()
    portfolio = get_portfolio_stats(positions)
    balance = float(portfolio["Balance"])
    open_count = int(portfolio["Open trades"])
    daily_pnl, daily_loss = _daily_performance(positions, balance)
    losses = _consecutive_losses(positions)
    exposure = _current_exposure(positions, balance)
    if entry is not None and stop_loss is not None and balance > 0:
        proposed_units = calculate_position_size(balance, entry, stop_loss)
        exposure += proposed_units * float(entry) / balance

    allowed = True
    reason = "Risk checks passed"
    if open_count >= MAX_OPEN_TRADES:
        allowed, reason = False, "Too many open trades"
    elif daily_loss >= MAX_DAILY_LOSS:
        allowed, reason = False, "Daily loss limit"
    elif losses >= MAX_CONSECUTIVE_LOSSES:
        allowed, reason = False, "Consecutive losses limit"
    elif exposure > MAX_PORTFOLIO_EXPOSURE:
        allowed, reason = False, "Portfolio exposure limit"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "balance": balance,
        "daily_pnl": daily_pnl,
        "daily_loss_percent": daily_loss,
        "open_trades": open_count,
        "exposure": exposure,
        "consecutive_losses": losses,
        "risk_allowed": allowed,
        "reason": reason,
    }


def can_open_trade(
    symbol: str,
    entry: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> tuple[bool, str]:
    """Return (allowed, reason) after all portfolio risk checks."""
    status = get_risk_status(symbol, entry, stop_loss)
    return bool(status["risk_allowed"]), str(status["reason"])


def _plot_history(history: pd.DataFrame, column: str, filename: str, title: str, ylabel: str) -> None:
    plt.figure(facecolor="white")
    axis = plt.gca()
    axis.set_facecolor("white")
    values = pd.to_numeric(history[column], errors="coerce").fillna(0)
    axis.plot(range(1, len(history) + 1), values)
    axis.set_xlabel("Bot run")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor="white")
    plt.close()


def update_risk_report() -> dict[str, Any]:
    """Append one risk snapshot and regenerate capital-protection charts."""
    status = get_risk_status()
    row = {column: status[column] for column in RISK_COLUMNS}
    path = Path(RISK_REPORT_FILE)
    try:
        history = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RISK_COLUMNS)
    except pd.errors.EmptyDataError:
        history = pd.DataFrame(columns=RISK_COLUMNS)
    addition = pd.DataFrame([row], columns=RISK_COLUMNS)
    history = addition if history.empty else pd.concat([history, addition], ignore_index=True)
    history.to_csv(path, index=False)

    exposure_percent = history.copy()
    exposure_percent["exposure"] = (
        pd.to_numeric(exposure_percent["exposure"], errors="coerce").fillna(0) * 100
    )
    _plot_history(
        exposure_percent,
        "exposure",
        "risk_exposure.png",
        "Portfolio Exposure",
        "Exposure (%)",
    )
    _plot_history(history, "daily_pnl", "daily_pnl.png", "Daily Paper PnL", "PnL ($)")
    _plot_history(
        history,
        "consecutive_losses",
        "consecutive_losses.png",
        "Consecutive Paper Losses",
        "Losses",
    )
    return status
