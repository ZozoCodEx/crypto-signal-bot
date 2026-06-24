"""Plain-text Telegram dashboard and /report command handler."""

import json
import os
import time
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
_commands_initialized = False


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


def _send_direct(token: str, chat_id: str, text: str) -> bool:
    """Send without command initialization, preventing recursive handlers."""
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


def send_message(text: str) -> bool:
    """Send one Bot API message, failing safely without exposing credentials."""
    credentials = _credentials()
    if credentials is None:
        return False
    token, chat_id = credentials
    _initialize_commands(token, chat_id)
    return _send_direct(token, chat_id, text)


def _read_csv(filename: str) -> pd.DataFrame:
    path = Path(filename)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normalized_direction(signal: Any) -> str:
    value = str(signal).upper()
    if value in {"BUY", "LONG"}:
        return "LONG"
    if value in {"SELL", "SHORT"}:
        return "SHORT"
    return "IGNORE"


def _latest_signal_observations(history: pd.DataFrame) -> pd.DataFrame:
    """Return one most-mature completed PnL observation per valid signal."""
    rows = []
    for _, signal in history.iterrows():
        direction = _normalized_direction(signal.get("signal", ""))
        if direction == "IGNORE":
            continue
        for suffix in ("7d", "48h", "24h"):
            value = pd.to_numeric(
                pd.Series([signal.get(f"pnl_{suffix}")]), errors="coerce"
            ).iloc[0]
            if pd.notna(value):
                rows.append(
                    {
                        "symbol": str(signal["symbol"]),
                        "direction": direction,
                        "pnl": float(value),
                    }
                )
                break
    return pd.DataFrame(rows, columns=["symbol", "direction", "pnl"])


def generate_report_message() -> str:
    """Build the /report response from persisted cumulative performance."""
    summary_frame = _read_csv("signal_performance_summary.csv")
    history = _read_csv("signal_history.csv")
    heading = "📊 Signal Performance Report"
    directions = (
        history["signal"].apply(_normalized_direction)
        if "signal" in history
        else pd.Series(dtype=str)
    )
    observation_counts = (
        f"Total observations: {len(history)}\n"
        f"LONG: {int((directions == 'LONG').sum())}\n"
        f"SHORT: {int((directions == 'SHORT').sum())}\n"
        f"IGNORE: {int((directions == 'IGNORE').sum())}"
    )
    if summary_frame.empty:
        return f"{heading}\n\n{observation_counts}\n\nNot enough completed signals yet."
    summary = summary_frame.iloc[-1]
    evaluated = 0
    for column in ("evaluated_24h", "evaluated_48h", "evaluated_7d"):
        value = pd.to_numeric(summary.get(column, 0), errors="coerce")
        evaluated += int(value) if pd.notna(value) else 0
    observations = _latest_signal_observations(history)
    if evaluated == 0 or observations.empty:
        return f"{heading}\n\n{observation_counts}\n\nNot enough completed signals yet."

    valid_history = history.copy()
    valid_history["direction"] = valid_history["signal"].apply(_normalized_direction)
    valid_history = valid_history[valid_history["direction"].isin({"LONG", "SHORT"})]

    direction_stats = {}
    for direction in ("LONG", "SHORT"):
        direction_history = valid_history[valid_history["direction"] == direction]
        completed = observations[observations["direction"] == direction]
        direction_stats[direction] = {
            "count": len(direction_history),
            "win_rate": (
                (completed["pnl"] > 0).mean() * 100 if not completed.empty else 0.0
            ),
            "average": completed["pnl"].mean() if not completed.empty else np.nan,
        }

    coin_performance = observations.groupby("symbol")["pnl"].mean().sort_values()
    best_coin = str(coin_performance.index[-1])
    worst_coin = str(coin_performance.index[0])
    direction_averages = {
        direction: stats["average"]
        for direction, stats in direction_stats.items()
        if pd.notna(stats["average"])
    }
    best_signal = (
        max(direction_averages, key=direction_averages.get)
        if direction_averages
        else "N/A"
    )
    average_return = float(observations["pnl"].mean())

    return (
        f"{heading}\n\n"
        f"{observation_counts}\n\n"
        "24h Results\n"
        f"Win Rate: {float(summary['win_rate_24h']):.1f}%\n"
        f"Average PnL: {float(summary['avg_pnl_24h']):+.2f}%\n\n"
        "48h Results\n"
        f"Win Rate: {float(summary['win_rate_48h']):.1f}%\n"
        f"Average PnL: {float(summary['avg_pnl_48h']):+.2f}%\n\n"
        "7d Results\n"
        f"Win Rate: {float(summary['win_rate_7d']):.1f}%\n"
        f"Average PnL: {float(summary['avg_pnl_7d']):+.2f}%\n\n"
        "LONG signals:\n"
        f"Count: {direction_stats['LONG']['count']}\n"
        f"Win Rate: {direction_stats['LONG']['win_rate']:.1f}%\n\n"
        "SHORT signals:\n"
        f"Count: {direction_stats['SHORT']['count']}\n"
        f"Win Rate: {direction_stats['SHORT']['win_rate']:.1f}%\n\n"
        f"Best coin:\n{best_coin}\n\n"
        f"Worst coin:\n{worst_coin}\n\n"
        f"Best signal:\n{best_signal}\n\n"
        f"Average return:\n{average_return:+.2f}%"
    )


def register_report_command(token: Optional[str] = None) -> bool:
    """Register /report in Telegram's command menu with setMyCommands."""
    if token is None:
        credentials = _credentials()
        if credentials is None:
            return False
        token = credentials[0]
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/setMyCommands",
            data={
                "commands": json.dumps(
                    [
                        {
                            "command": "report",
                            "description": "Signal performance report",
                        }
                    ]
                )
            },
            timeout=15,
        )
        response.raise_for_status()
        return bool(response.json().get("ok", False))
    except (requests.RequestException, ValueError):
        print("Telegram command registration failed.")
        return False


def _fetch_updates(token: str, offset: Optional[int] = None, timeout: int = 0) -> list:
    params: dict[str, Any] = {
        "timeout": timeout,
        "allowed_updates": json.dumps(["message"]),
    }
    if offset is not None:
        params["offset"] = offset
    response = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        params=params,
        timeout=max(15, timeout + 5),
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("result", []) if payload.get("ok", False) else []


def _handle_updates(token: str, chat_id: str, updates: list) -> Optional[int]:
    highest_update_id: Optional[int] = None
    for update in updates:
        update_id = int(update.get("update_id", 0))
        highest_update_id = max(highest_update_id or update_id, update_id)
        message = update.get("message", {})
        incoming_chat = str(message.get("chat", {}).get("id", ""))
        command = str(message.get("text", "")).strip().split(maxsplit=1)[0]
        if incoming_chat == str(chat_id) and command.split("@", 1)[0] == "/report":
            _send_direct(token, chat_id, generate_report_message())
    return highest_update_id


def process_telegram_commands(
    token: Optional[str] = None, chat_id: Optional[str] = None
) -> bool:
    """Process pending /report commands once and acknowledge handled updates."""
    if token is None or chat_id is None:
        credentials = _credentials()
        if credentials is None:
            return False
        token, chat_id = credentials
    try:
        updates = _fetch_updates(token)
        highest = _handle_updates(token, chat_id, updates)
        if highest is not None:
            # A higher offset confirms all returned updates with Telegram.
            _fetch_updates(token, offset=highest + 1)
        return True
    except (requests.RequestException, ValueError):
        print("Telegram command polling failed.")
        return False


def _initialize_commands(token: str, chat_id: str) -> None:
    global _commands_initialized
    if _commands_initialized:
        return
    register_report_command(token)
    process_telegram_commands(token, chat_id)
    _commands_initialized = True


def run_command_listener() -> None:
    """Continuously handle /report while this module is run as a process."""
    credentials = _credentials()
    if credentials is None:
        return
    token, chat_id = credentials
    register_report_command(token)
    offset: Optional[int] = None
    print("Telegram /report listener started.")
    while True:
        try:
            updates = _fetch_updates(token, offset=offset, timeout=30)
            highest = _handle_updates(token, chat_id, updates)
            if highest is not None:
                offset = highest + 1
        except (requests.RequestException, ValueError):
            print("Telegram command polling failed.")
            time.sleep(5)


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


if __name__ == "__main__":
    run_command_listener()
