"""Plain-text Telegram dashboard for paper-trading analytics."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from config import TRADES_FILE
from position_tracker import get_portfolio_stats
from risk_manager import get_risk_status


_disabled_notified = False


def _credentials() -> Optional[tuple[str, str]]:
    global _disabled_notified
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        if not _disabled_notified:
            print("Telegram disabled.")
            _disabled_notified = True
        return None
    return token, chat_id


def send_message(text: str) -> bool:
    """Send one Bot API message, failing safely without exposing credentials."""
    credentials = _credentials()
    if credentials is None:
        return False
    token, chat_id = credentials
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            print("Telegram send failed.")
            return False
        return True
    except (requests.RequestException, ValueError):
        print("Telegram send failed.")
        return False


def _read_csv(filename: str) -> pd.DataFrame:
    path = Path(filename)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def send_market_regime(regime_data: Optional[dict[str, Any]] = None) -> bool:
    """Send the current BTC regime and indicator dashboard."""
    if regime_data is None:
        frame = _read_csv("market_regime.csv")
        if frame.empty:
            return False
        regime_data = frame.iloc[-1].to_dict()
    regime = str(regime_data["regime"])
    marker = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(regime, "")
    text = (
        "🤖 ELITE Crypto Bot\n\n"
        f"Market regime:\n{regime} {marker}\n\n"
        f"Price: {float(regime_data['price']):.2f}\n"
        f"EMA50: {float(regime_data['ema50']):.2f}\n"
        f"EMA200: {float(regime_data['ema200']):.2f}\n"
        f"RSI: {float(regime_data['rsi']):.2f}\n"
        f"ADX: {float(regime_data['adx']):.2f}\n"
        f"ATR: {float(regime_data['atr']):.2f}"
    )
    return send_message(text)


def _is_elite(row: Any) -> bool:
    return bool(
        row["score"] >= 90
        and row["volume_ratio"] > 1.5
        and row["atr"] > row["atr_avg"]
        and 48 <= row["rsi"] <= 58
    )


def send_top_opportunities(opportunities: Optional[pd.DataFrame] = None) -> bool:
    """Send the five highest-ranked opportunities and their key indicators."""
    if opportunities is None:
        opportunities = _read_csv("opportunities.csv")
    if opportunities.empty:
        return send_message("Top 5 opportunities\n\nNo opportunity data available.")
    ranked = opportunities.sort_values(
        ["score", "volume_ratio"], ascending=[False, False]
    ).head(5)
    sections = ["Top 5 opportunities"]
    for _, row in ranked.iterrows():
        signal = "ELITE" if _is_elite(row) else str(row.get("label", "IGNORE"))
        sections.append(
            f"{row['symbol']}\n"
            f"Score: {int(row['score'])}\n"
            f"RSI: {float(row['rsi']):.1f}\n"
            f"Volume: {float(row['volume_ratio']):.2f}x\n"
            f"ATR: {float(row['atr']):.6g}\n"
            f"Signal: {signal}"
        )
    return send_message("\n\n".join(sections))


def _duration(opened_at: Any) -> str:
    opened = pd.to_datetime(opened_at, utc=True, errors="coerce")
    if pd.isna(opened):
        return "unknown"
    minutes = max(
        0,
        int((datetime.now(timezone.utc) - opened.to_pydatetime()).total_seconds() / 60),
    )
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes}m"


def send_open_trades(trades: Optional[pd.DataFrame] = None) -> bool:
    """Send current virtual positions with unrealized paper PnL."""
    if trades is None:
        trades = _read_csv(TRADES_FILE)
    if trades.empty or "status" not in trades:
        return send_message("Open paper trades\n\nNo open trades.")
    open_trades = trades[trades["status"] == "OPEN"]
    if open_trades.empty:
        return send_message("Open paper trades\n\nNo open trades.")

    sections = ["Open paper trades"]
    for _, trade in open_trades.iterrows():
        entry = float(trade["entry_price"])
        current = float(trade["current_price"])
        pnl = ((current - entry) / entry) * 100
        if str(trade["direction"]).upper() == "SHORT":
            pnl = -pnl
        sections.append(
            f"{trade['symbol']}\n"
            f"Direction: {trade['direction']}\n"
            f"Entry: {entry:.8g}\n"
            f"Current price: {current:.8g}\n"
            f"PnL: {pnl:+.2f}%\n"
            f"TP: {float(trade['take_profit']):.8g}\n"
            f"SL: {float(trade['stop_loss']):.8g}\n"
            f"Duration: {_duration(trade['entry_time'])}"
        )
    return send_message("\n\n".join(sections))


def _portfolio_metrics(trades: pd.DataFrame) -> dict[str, float]:
    stats = get_portfolio_stats(trades)
    return {
        "balance": stats["Balance"],
        "return": stats["Return"],
        "total_trades": stats["Closed trades"],
        "win_rate": stats["Win rate"],
        "profit_factor": stats["Profit factor"],
        "max_drawdown": stats["Max DD"],
        "open_trades": stats["Open trades"],
    }


def send_portfolio_summary(trades: Optional[pd.DataFrame] = None) -> bool:
    """Send aggregate performance for closed and currently open paper trades."""
    if trades is None:
        trades = _read_csv(TRADES_FILE)
    metrics = _portfolio_metrics(trades)
    factor = (
        "∞" if np.isinf(metrics["profit_factor"]) else f"{metrics['profit_factor']:.2f}"
    )
    text = (
        "📊 Portfolio\n\n"
        f"Balance: ${metrics['balance']:.2f}\n"
        f"Return: {metrics['return']:+.2f}%\n"
        f"Total trades: {int(metrics['total_trades'])}\n"
        f"Win rate: {metrics['win_rate']:.1f}%\n"
        f"Profit factor: {factor}\n"
        f"Max drawdown: {metrics['max_drawdown']:.1f}%\n"
        f"Open trades: {int(metrics['open_trades'])}"
    )
    return send_message(text)


def send_risk_manager(risk_data: Optional[dict[str, Any]] = None) -> bool:
    """Send current capital-protection limits and trading permission."""
    risk_data = get_risk_status() if risk_data is None else risk_data
    allowed = bool(risk_data["risk_allowed"])
    text = (
        "🛡 Risk Manager\n\n"
        f"Balance: ${float(risk_data['balance']):.2f}\n"
        f"Risk/trade: 1%\n"
        f"Open trades: {int(risk_data['open_trades'])}/3\n"
        f"Exposure: {float(risk_data['exposure']) * 100:.1f}%\n"
        f"Daily loss: {-float(risk_data['daily_loss_percent']) * 100:.1f}%\n"
        f"Consecutive losses: {int(risk_data['consecutive_losses'])}/3\n"
        f"Trading: {'✅ ENABLED' if allowed else '❌ DISABLED'}\n"
        f"Reason: {risk_data['reason']}"
    )
    return send_message(text)


def send_signal_performance(summary: Optional[dict[str, Any]] = None) -> bool:
    """Send cumulative valid-signal performance across all evaluation horizons."""
    if summary is None:
        frame = _read_csv("signal_performance_summary.csv")
        if frame.empty:
            return send_message("📊 Signal Performance\n\nNot enough data yet.")
        summary = frame.iloc[-1].to_dict()
    evaluated = (
        int(summary["evaluated_24h"])
        + int(summary["evaluated_48h"])
        + int(summary["evaluated_7d"])
    )
    if evaluated == 0:
        return send_message(
            "📊 Signal Performance\n\n"
            f"Total signals tracked: {int(summary['total_signals'])}\n\n"
            "Not enough data yet."
        )
    text = (
        "📊 Signal Performance\n\n"
        f"Total signals tracked: {int(summary['total_signals'])}\n\n"
        "24h:\n"
        f"Evaluated: {int(summary['evaluated_24h'])}\n"
        f"Win rate: {float(summary['win_rate_24h']):.1f}%\n"
        f"Avg PnL: {float(summary['avg_pnl_24h']):+.2f}%\n\n"
        "48h:\n"
        f"Evaluated: {int(summary['evaluated_48h'])}\n"
        f"Win rate: {float(summary['win_rate_48h']):.1f}%\n"
        f"Avg PnL: {float(summary['avg_pnl_48h']):+.2f}%\n\n"
        "7d:\n"
        f"Evaluated: {int(summary['evaluated_7d'])}\n"
        f"Win rate: {float(summary['win_rate_7d']):.1f}%\n"
        f"Avg PnL: {float(summary['avg_pnl_7d']):+.2f}%\n\n"
        f"Best signal:\n{summary['best_signal']}\n\n"
        f"Worst signal:\n{summary['worst_signal']}"
    )
    return send_message(text)
