"""Equity curves and trade diagnostics for unchanged ELITE simulations."""

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-signal-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import INITIAL_BALANCE


EQUITY_COLUMNS = [
    "trade_number",
    "timestamp",
    "balance",
    "profit_pct",
    "cumulative_return",
    "drawdown",
    "symbol",
    "regime",
]
STREAK_COLUMNS = [
    "max_win_streak",
    "max_loss_streak",
    "average_win",
    "average_loss",
    "best_trade",
    "worst_trade",
    "profit_factor",
    "expectancy",
    "sharpe_like_ratio",
]
GENERATED_FILES = [
    "equity_curve_total.csv",
    "equity_curve_bull.csv",
    "equity_curve_bear.csv",
    "equity_curve_sideways.csv",
    "equity_total.png",
    "equity_bull.png",
    "equity_bear.png",
    "equity_sideways.png",
    "drawdown_total.png",
    "trade_streaks.csv",
]
SEPARATOR = "=" * 34


def _equity_frame(trades: list[dict[str, Any]], regime: str) -> pd.DataFrame:
    """Convert chronological realized trades to an equity/drawdown series."""
    balance = float(INITIAL_BALANCE)
    peak = balance
    rows = []
    for number, trade in enumerate(trades, start=1):
        balance += float(trade["_pnl"])
        peak = max(peak, balance)
        rows.append(
            {
                "trade_number": number,
                "timestamp": trade["closed_at"],
                "balance": balance,
                "profit_pct": float(trade["profit_pct"]),
                "cumulative_return": (balance / INITIAL_BALANCE - 1) * 100,
                "drawdown": (balance / peak - 1) * 100,
                "symbol": trade["symbol"],
                "regime": regime,
            }
        )
    return pd.DataFrame(rows, columns=EQUITY_COLUMNS)


def _plot_equity(frame: pd.DataFrame, filename: str, title: str) -> None:
    plt.figure(facecolor="white")
    axis = plt.gca()
    axis.set_facecolor("white")
    if frame.empty:
        axis.plot([0], [INITIAL_BALANCE])
    else:
        axis.plot(frame["trade_number"], frame["balance"])
    axis.set_xlabel("Trade number")
    axis.set_ylabel("Balance")
    axis.set_title(title)
    axis.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor="white")
    plt.close()


def _plot_drawdown(frame: pd.DataFrame) -> None:
    plt.figure(facecolor="white")
    axis = plt.gca()
    axis.set_facecolor("white")
    if frame.empty:
        axis.plot([0], [0])
    else:
        axis.plot(frame["trade_number"], frame["drawdown"])
    axis.set_xlabel("Trade number")
    axis.set_ylabel("Drawdown (%)")
    axis.set_title("ELITE Strategy Drawdown — Total")
    axis.grid(True)
    plt.tight_layout()
    plt.savefig("drawdown_total.png", dpi=150, facecolor="white")
    plt.close()


def _longest_streak(results: list[str], target: str) -> int:
    longest = 0
    current = 0
    for result in results:
        if result == target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _trade_analysis(trades: list[dict[str, Any]]) -> dict[str, float]:
    returns = np.asarray([float(trade["profit_pct"]) for trade in trades])
    results = [str(trade["result"]) for trade in trades]
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy = float(returns.mean()) if len(returns) else 0.0
    deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe_like = expectancy / deviation * np.sqrt(len(returns)) if deviation else 0.0
    return {
        "max_win_streak": _longest_streak(results, "WIN"),
        "max_loss_streak": _longest_streak(results, "LOSS"),
        "average_win": average_win,
        "average_loss": average_loss,
        "best_trade": float(returns.max()) if len(returns) else 0.0,
        "worst_trade": float(returns.min()) if len(returns) else 0.0,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "sharpe_like_ratio": float(sharpe_like),
    }


def _print_regime(name: str, frame: pd.DataFrame) -> None:
    total_return = frame["cumulative_return"].iloc[-1] if not frame.empty else 0.0
    max_drawdown = frame["drawdown"].min() if not frame.empty else 0.0
    print(f"\n{name}:")
    print(f"Trades: {len(frame)}")
    print(f"Return: {total_return:+.1f}%")
    print(f"Max Drawdown: {max_drawdown:.1f}%")


def generate_visual_report(
    total_state: dict[str, Any], regime_states: dict[str, dict[str, Any]]
) -> None:
    """Create all CSV reports and PNG charts from chronological simulations."""
    frames = {
        "TOTAL": _equity_frame(total_state["trades"], "TOTAL"),
        "BULL": _equity_frame(regime_states["BULL"]["trades"], "BULL"),
        "BEAR": _equity_frame(regime_states["BEAR"]["trades"], "BEAR"),
        "SIDEWAYS": _equity_frame(
            regime_states["SIDEWAYS"]["trades"], "SIDEWAYS"
        ),
    }
    csv_files = {
        "TOTAL": "equity_curve_total.csv",
        "BULL": "equity_curve_bull.csv",
        "BEAR": "equity_curve_bear.csv",
        "SIDEWAYS": "equity_curve_sideways.csv",
    }
    png_files = {
        "TOTAL": "equity_total.png",
        "BULL": "equity_bull.png",
        "BEAR": "equity_bear.png",
        "SIDEWAYS": "equity_sideways.png",
    }
    for name, frame in frames.items():
        frame.to_csv(csv_files[name], index=False)
        _plot_equity(frame, png_files[name], f"ELITE Strategy Equity — {name.title()}")
    _plot_drawdown(frames["TOTAL"])

    analysis = _trade_analysis(total_state["trades"])
    pd.DataFrame([analysis], columns=STREAK_COLUMNS).to_csv(
        "trade_streaks.csv", index=False
    )
    total = frames["TOTAL"]
    total_return = total["cumulative_return"].iloc[-1] if not total.empty else 0.0
    max_drawdown = total["drawdown"].min() if not total.empty else 0.0

    print(f"\n{SEPARATOR}\nSTRATEGY VISUAL REPORT\n{SEPARATOR}\n")
    print(f"Total Return: {total_return:+.1f}%")
    print(f"Max Drawdown: {max_drawdown:.1f}%")
    print(f"Longest Win Streak: {int(analysis['max_win_streak'])}")
    print(f"Longest Loss Streak: {int(analysis['max_loss_streak'])}")
    print(f"Average Win: {analysis['average_win']:+.2f}%")
    print(f"Average Loss: {analysis['average_loss']:.2f}%")
    print(f"Expectancy: {analysis['expectancy']:+.2f}%")
    _print_regime("BULL", frames["BULL"])
    _print_regime("BEAR", frames["BEAR"])
    _print_regime("SIDEWAYS", frames["SIDEWAYS"])
    print(f"\n{SEPARATOR}")
    print("\nVisual report generated successfully.")
    print("\nGenerated files:")
    for filename in GENERATED_FILES:
        if Path(filename).exists():
            print(f"- {filename}")
