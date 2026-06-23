"""Historical comparison of five exceptional-quality opportunity modes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from binance_client import get_historical_klines
from config import (
    BACKTEST_CANDLES,
    BACKTEST_INTERVAL,
    BACKTEST_SYMBOLS,
    BACKTEST_RISK_PERCENT,
    INITIAL_BALANCE,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
)
from indicators import add_indicators
from opportunity_scanner import score_opportunity


RESULTS_FILE = "elite_backtest_results.csv"
SYMBOL_REPORT_FILE = "symbol_report.csv"
FILTERED_RESULTS_FILE = "filtered_backtest_results.csv"
TRAIN_SYMBOL_REPORT_FILE = "symbol_report_train.csv"
WALKFORWARD_TRAIN_FILE = "walkforward_train.csv"
WALKFORWARD_TEST_FILE = "walkforward_test.csv"
WALKFORWARD_SUMMARY_FILE = "walkforward_summary.csv"
BULL_RESULTS_FILE = "bull_market_results.csv"
BEAR_RESULTS_FILE = "bear_market_results.csv"
SIDEWAYS_RESULTS_FILE = "sideways_market_results.csv"
REGIME_SUMMARY_FILE = "market_regime_summary.csv"
RESULT_COLUMNS = [
    "mode",
    "symbol",
    "entry",
    "exit",
    "result",
    "profit_pct",
    "score",
    "volume_ratio",
    "atr",
    "rsi",
]
SYMBOL_REPORT_COLUMNS = [
    "symbol",
    "trades",
    "wins",
    "losses",
    "win_rate",
    "total_profit_pct",
    "average_trade_pct",
    "profit_factor",
    "max_drawdown",
    "best_trade_pct",
    "worst_trade_pct",
]
FILTERED_COLUMNS = [
    "group_name",
    "symbols",
    "trades",
    "win_rate",
    "return_pct",
    "max_drawdown",
    "profit_factor",
    "average_trade_pct",
    "best_symbol",
    "worst_symbol",
]
WALKFORWARD_SUMMARY_COLUMNS = [
    "period",
    "symbols",
    "trades",
    "win_rate",
    "return_pct",
    "profit_factor",
    "max_drawdown",
    "average_trade_pct",
    "best_symbol",
    "worst_symbol",
]
REGIME_SUMMARY_COLUMNS = [
    "regime",
    "trades",
    "win_rate",
    "return_pct",
    "profit_factor",
    "max_drawdown",
    "average_trade_pct",
    "best_coin",
    "worst_coin",
    "status",
]
SEPARATOR = "=" * 94
MACRO_WARMUP_CANDLES = 3200


def _mode_1(item: dict[str, Any]) -> bool:
    return item["score"] >= 85


def _mode_2(item: dict[str, Any]) -> bool:
    return item["score"] >= 90


def _mode_3(item: dict[str, Any]) -> bool:
    return item["score"] >= 95


def _mode_4(item: dict[str, Any]) -> bool:
    return item["score"] >= 90 and item["volume_ratio"] > 1.5


def _mode_5(item: dict[str, Any]) -> bool:
    return bool(
        item["score"] >= 90
        and item["volume_ratio"] > 1.5
        and item["atr"] > item["atr_avg"]
        and 48 <= item["rsi"] <= 58
    )


MODE_RULES: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
    ("MODE_1", "Score ≥ 85", _mode_1),
    ("MODE_2", "Score ≥ 90", _mode_2),
    ("MODE_3", "Score ≥ 95", _mode_3),
    ("MODE_4", "Score ≥ 90 + volume > 1.5", _mode_4),
    ("MODE_5_ELITE", "ELITE", _mode_5),
]


def _build_timeframe(candles: pd.DataFrame, frequency: str) -> pd.DataFrame:
    aggregated = candles.set_index("open_time").resample(
        frequency, label="left", closed="left"
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "close_time": "max",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    return add_indicators(aggregated.reset_index())


def _download_symbol(symbol: str, count: int) -> tuple[str, pd.DataFrame]:
    candles = get_historical_klines(symbol, BACKTEST_INTERVAL, count)
    now = pd.Timestamp.now(tz="UTC")
    complete = candles[candles["close_time"] <= now].reset_index(drop=True)
    return symbol, complete


def _download_universe() -> dict[str, pd.DataFrame]:
    count = BACKTEST_CANDLES + MACRO_WARMUP_CANDLES
    histories: dict[str, pd.DataFrame] = {}
    print(
        f"Downloading {BACKTEST_CANDLES} test + {MACRO_WARMUP_CANDLES} "
        f"warmup candles for {len(BACKTEST_SYMBOLS)} symbols...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_download_symbol, symbol, count): symbol
            for symbol in BACKTEST_SYMBOLS
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                name, history = future.result()
                histories[name] = history
                print(f"  {name}: {len(history)} candles", flush=True)
            except Exception as error:
                print(f"  {symbol}: download failed ({error})", flush=True)
    return histories


def _prepare_history(candles: pd.DataFrame) -> dict[str, Any]:
    entry = add_indicators(candles)
    trend = _build_timeframe(candles, "1h")
    macro = _build_timeframe(candles, "4h")
    entry_times = entry["close_time"].astype("int64").to_numpy()
    return {
        "entry": entry,
        "entry_lookup": {int(value): index for index, value in enumerate(entry_times)},
        "trend": trend,
        "trend_times": trend["close_time"].astype("int64").to_numpy(),
        "macro": macro,
        "macro_times": macro["close_time"].astype("int64").to_numpy(),
    }


def _snapshot(
    symbol: str, prepared: dict[str, Any], timestamp_ns: int
) -> Optional[dict[str, Any]]:
    entry_position = prepared["entry_lookup"].get(timestamp_ns)
    if entry_position is None:
        return None
    trend_position = int(
        prepared["trend_times"].searchsorted(timestamp_ns, side="right") - 1
    )
    macro_position = int(
        prepared["macro_times"].searchsorted(timestamp_ns, side="right") - 1
    )
    if trend_position < 199 or macro_position < 199:
        return None

    entry = prepared["entry"].iloc[entry_position]
    if pd.isna(entry["atr14"]) or pd.isna(entry["atr_avg"]):
        return None
    scored = score_opportunity(
        symbol,
        entry,
        prepared["trend"].iloc[trend_position],
        prepared["macro"].iloc[macro_position],
        timestamp=entry["close_time"].to_pydatetime(),
    )
    scored["high"] = float(entry["high"])
    scored["low"] = float(entry["low"])
    return scored


def _new_state(
    name: str,
    description: str,
    rule: Callable,
    allowed_symbols: Optional[set[str]] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "rule": rule,
        "allowed_symbols": allowed_symbols,
        "balance": float(INITIAL_BALANCE),
        "position": None,
        "trades": [],
        "equity": [float(INITIAL_BALANCE)],
    }


def _close_position(state: dict[str, Any], snapshots: dict[str, dict]) -> None:
    position = state["position"]
    if position is None:
        return
    snapshot = snapshots.get(position["symbol"])
    if snapshot is None:
        return

    exit_price: Optional[float] = None
    result = ""
    if snapshot["high"] >= position["take_profit"]:
        exit_price = position["take_profit"]
        result = "WIN"
    elif snapshot["low"] <= position["stop_loss"]:
        exit_price = position["stop_loss"]
        result = "LOSS"
    if exit_price is None:
        return

    pnl = position["quantity"] * (exit_price - position["entry"])
    state["balance"] += pnl
    state["equity"].append(state["balance"])
    state["trades"].append(
        {
            "mode": state["name"],
            "symbol": position["symbol"],
            "entry": position["entry"],
            "exit": exit_price,
            "result": result,
            "profit_pct": ((exit_price - position["entry"]) / position["entry"])
            * 100,
            "score": position["score"],
            "volume_ratio": position["volume_ratio"],
            "atr": position["atr"],
            "rsi": position["rsi"],
            "opened_at": position["opened_at"],
            "closed_at": snapshot["timestamp"],
            "_pnl": pnl,
        }
    )
    state["position"] = None


def _open_best_exceptional(
    state: dict[str, Any], ranked: list[dict[str, Any]]
) -> None:
    if state["position"] is not None:
        return
    allowed = state["allowed_symbols"]
    opportunity = next(
        (
            item
            for item in ranked
            if state["rule"](item)
            and (allowed is None or item["symbol"] in allowed)
        ),
        None,
    )
    if opportunity is None:
        return

    entry = opportunity["price"]
    atr = opportunity["atr"]
    stop_loss = entry - atr * SL_ATR_MULTIPLIER
    take_profit = entry + atr * TP_ATR_MULTIPLIER
    risk_amount = state["balance"] * (BACKTEST_RISK_PERCENT / 100)
    risk_quantity = risk_amount / (entry - stop_loss)
    cash_quantity = state["balance"] / entry
    # Cap notional at the account balance: spot-only, no implicit leverage.
    quantity = min(risk_quantity, cash_quantity)
    state["position"] = {
        "symbol": opportunity["symbol"],
        "opened_at": opportunity["timestamp"],
        "entry": entry,
        "atr": atr,
        "score": opportunity["score"],
        "volume_ratio": opportunity["volume_ratio"],
        "rsi": opportunity["rsi"],
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "quantity": quantity,
    }


def _statistics(state: dict[str, Any]) -> dict[str, Any]:
    trades = state["trades"]
    wins = [trade for trade in trades if trade["result"] == "WIN"]
    gross_profit = sum(max(0.0, trade["_pnl"]) for trade in trades)
    gross_loss = abs(sum(min(0.0, trade["_pnl"]) for trade in trades))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    equity = np.asarray(state["equity"], dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks * 100

    by_symbol: dict[str, float] = {}
    for trade in trades:
        by_symbol[trade["symbol"]] = by_symbol.get(trade["symbol"], 0.0) + trade["_pnl"]
    ordered_symbols = sorted(by_symbol.items(), key=lambda item: item[1], reverse=True)

    return {
        "mode": state["name"],
        "description": state["description"],
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "return": ((state["balance"] - INITIAL_BALANCE) / INITIAL_BALANCE) * 100,
        "max_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "profit_factor": profit_factor,
        "average_trade": (
            float(np.mean([trade["profit_pct"] for trade in trades]))
            if trades
            else 0.0
        ),
        "best_symbol": ordered_symbols[0][0] if ordered_symbols else "N/A",
        "worst_symbol": ordered_symbols[-1][0] if ordered_symbols else "N/A",
        "average_score": (
            float(np.mean([trade["score"] for trade in trades])) if trades else 0.0
        ),
        "average_volume": (
            float(np.mean([trade["volume_ratio"] for trade in trades]))
            if trades
            else 0.0
        ),
        "average_rsi": (
            float(np.mean([trade["rsi"] for trade in trades])) if trades else 0.0
        ),
        "best_coins": [symbol for symbol, _ in ordered_symbols[:3]],
    }


def _factor_text(value: float) -> str:
    return "∞" if np.isinf(value) else f"{value:.2f}"


def _run_timeline(
    prepared: dict[str, dict[str, Any]],
    timeline: list[int],
    states: list[dict[str, Any]],
) -> None:
    """Replay one market timeline for any set of independent mode states."""
    for timestamp_ns in timeline:
        snapshots = {}
        for symbol, market in prepared.items():
            snapshot = _snapshot(symbol, market, int(timestamp_ns))
            if snapshot is not None:
                snapshots[symbol] = snapshot
        ranked = sorted(
            snapshots.values(),
            key=lambda item: (item["score"], item["volume_ratio"]),
            reverse=True,
        )
        for state in states:
            _close_position(state, snapshots)
            _open_best_exceptional(state, ranked)


def _create_symbol_report(elite_state: dict[str, Any]) -> pd.DataFrame:
    """Attribute realized ELITE performance to every configured symbol."""
    rows = []
    for symbol in BACKTEST_SYMBOLS:
        trades = [
            trade for trade in elite_state["trades"] if trade["symbol"] == symbol
        ]
        wins = sum(trade["result"] == "WIN" for trade in trades)
        losses = sum(trade["result"] == "LOSS" for trade in trades)
        gross_profit = sum(max(0.0, trade["_pnl"]) for trade in trades)
        gross_loss = abs(sum(min(0.0, trade["_pnl"]) for trade in trades))
        profit_factor = gross_profit / gross_loss if gross_loss else (
            float("inf") if gross_profit else 0.0
        )
        equity = [float(INITIAL_BALANCE)]
        for trade in trades:
            equity.append(equity[-1] + trade["_pnl"])
        equity_values = np.asarray(equity, dtype=float)
        peaks = np.maximum.accumulate(equity_values)
        max_drawdown = float(((equity_values - peaks) / peaks * 100).min())
        trade_returns = [trade["profit_pct"] for trade in trades]
        rows.append(
            {
                "symbol": symbol,
                "trades": len(trades),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(trades) * 100 if trades else 0.0,
                "total_profit_pct": sum(
                    trade["_pnl"] for trade in trades
                )
                / INITIAL_BALANCE
                * 100,
                "average_trade_pct": (
                    float(np.mean(trade_returns)) if trade_returns else 0.0
                ),
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "best_trade_pct": max(trade_returns) if trade_returns else 0.0,
                "worst_trade_pct": min(trade_returns) if trade_returns else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=SYMBOL_REPORT_COLUMNS).sort_values(
        ["total_profit_pct", "profit_factor", "max_drawdown"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _print_symbol_report(report: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}\nSYMBOL PERFORMANCE — ELITE MODE\n{SEPARATOR}")
    print(
        f"{'Symbol':<11}{'Trades':>8}{'Win Rate':>11}{'Return':>10}"
        f"{'PF':>9}{'Max DD':>10}{'Best':>10}{'Worst':>10}"
    )
    for _, row in report.iterrows():
        print(
            f"{row['symbol']:<11}{int(row['trades']):>8}{row['win_rate']:>10.1f}%"
            f"{row['total_profit_pct']:>9.1f}%{_factor_text(row['profit_factor']):>9}"
            f"{row['max_drawdown']:>9.1f}%{row['best_trade_pct']:>9.2f}%"
            f"{row['worst_trade_pct']:>9.2f}%"
        )
    top_five = report.head(5)["symbol"].tolist()
    bottom_five = report.sort_values(
        "total_profit_pct", ascending=True
    ).head(5)["symbol"].tolist()
    print(f"\nTop 5 symbols: {', '.join(top_five)}")
    print(f"Bottom 5 symbols: {', '.join(bottom_five)}")
    print(SEPARATOR)


def _filtered_groups(report: pd.DataFrame) -> dict[str, list[str]]:
    top = report.sort_values("total_profit_pct", ascending=False)
    bottom_five = set(top.tail(5)["symbol"])
    return {
        "GROUP_A_TOP_PERFORMERS": top.head(5)["symbol"].tolist(),
        "GROUP_B_REMOVE_WORST": [
            symbol for symbol in BACKTEST_SYMBOLS if symbol not in bottom_five
        ],
        "GROUP_C_MANUAL_HIGH_QUALITY": [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "INJUSDT",
            "LINKUSDT",
            "ATOMUSDT",
            "SUIUSDT",
            "NEARUSDT",
        ],
        "GROUP_D_ONLY_BEST_3": top.head(3)["symbol"].tolist(),
    }


def _run_filtered_groups(
    prepared: dict[str, dict[str, Any]],
    timeline: list[int],
    report: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    groups = _filtered_groups(report)
    states = [
        _new_state(name, "ELITE filtered group", _mode_5, set(symbols))
        for name, symbols in groups.items()
    ]
    _run_timeline(prepared, timeline, states)

    rows = []
    for state in states:
        stats = _statistics(state)
        symbols = groups[state["name"]]
        rows.append(
            {
                "group_name": state["name"],
                "symbols": ",".join(symbols),
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "return_pct": stats["return"],
                "max_drawdown": stats["max_drawdown"],
                "profit_factor": stats["profit_factor"],
                "average_trade_pct": stats["average_trade"],
                "best_symbol": stats["best_symbol"],
                "worst_symbol": stats["worst_symbol"],
            }
        )
    filtered = pd.DataFrame(rows, columns=FILTERED_COLUMNS).sort_values(
        ["return_pct", "max_drawdown", "profit_factor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return filtered, rows


def _print_filtered_groups(filtered: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}\nFILTERED GROUP BACKTEST\n{SEPARATOR}")
    print(
        f"{'Group':<30}{'Symbols':>8}{'Trades':>8}{'Win Rate':>11}"
        f"{'Return':>10}{'Max DD':>10}{'PF':>9}{'Best':>10}{'Worst':>10}"
    )
    for _, row in filtered.iterrows():
        symbol_count = len(str(row["symbols"]).split(","))
        print(
            f"{row['group_name']:<30}{symbol_count:>8}{int(row['trades']):>8}"
            f"{row['win_rate']:>10.1f}%{row['return_pct']:>9.1f}%"
            f"{row['max_drawdown']:>9.1f}%{_factor_text(row['profit_factor']):>9}"
            f"{row['best_symbol']:>10}{row['worst_symbol']:>10}"
        )
    winner = filtered.iloc[0]
    print("\n🏆 BEST GROUP\n")
    print(winner["group_name"])
    print(f"Symbols: {winner['symbols']}")
    print(
        f"Return {winner['return_pct']:+.1f}%, max drawdown "
        f"{winner['max_drawdown']:.1f}%, profit factor "
        f"{_factor_text(winner['profit_factor'])}."
    )
    print(f"\n{SEPARATOR}")


def _export_state_trades(state: dict[str, Any], filename: str) -> pd.DataFrame:
    """Export one walk-forward state's closed trades using the elite schema."""
    rows = []
    for trade in state["trades"]:
        rows.append({column: trade.get(column) for column in RESULT_COLUMNS})
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_csv(filename, index=False)
    return frame


def _summary_row(
    period: str, symbols: list[str], stats: dict[str, Any]
) -> dict[str, Any]:
    return {
        "period": period,
        "symbols": ",".join(symbols),
        "trades": stats["trades"],
        "win_rate": stats["win_rate"],
        "return_pct": stats["return"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown": stats["max_drawdown"],
        "average_trade_pct": stats["average_trade"],
        "best_symbol": stats["best_symbol"],
        "worst_symbol": stats["worst_symbol"],
    }


def _run_walkforward(
    prepared: dict[str, dict[str, Any]], reference_times: pd.Series
) -> tuple[pd.DataFrame, bool]:
    """Select symbols on the first 70% and validate frozen symbols on 30%."""
    window = reference_times.tail(BACKTEST_CANDLES).tolist()
    split_position = int(len(window) * 0.70)
    train_timeline = window[250:split_position]
    test_timeline = window[split_position:]

    train_state = _new_state(
        "WALKFORWARD_TRAIN", "ELITE train period", _mode_5
    )
    _run_timeline(prepared, train_timeline, [train_state])
    train_report = _create_symbol_report(train_state)
    train_report.to_csv(TRAIN_SYMBOL_REPORT_FILE, index=False)
    frozen_symbols = train_report.head(5)["symbol"].tolist()

    test_state = _new_state(
        "WALKFORWARD_TEST",
        "ELITE unseen test period",
        _mode_5,
        set(frozen_symbols),
    )
    _run_timeline(prepared, test_timeline, [test_state])

    _export_state_trades(train_state, WALKFORWARD_TRAIN_FILE)
    _export_state_trades(test_state, WALKFORWARD_TEST_FILE)
    train_stats = _statistics(train_state)
    test_stats = _statistics(test_state)
    summary = pd.DataFrame(
        [
            _summary_row("TRAIN", list(BACKTEST_SYMBOLS), train_stats),
            _summary_row("TEST", frozen_symbols, test_stats),
        ],
        columns=WALKFORWARD_SUMMARY_COLUMNS,
    )
    summary.to_csv(WALKFORWARD_SUMMARY_FILE, index=False)

    return_difference = test_stats["return"] - train_stats["return"]
    drawdown_difference = test_stats["max_drawdown"] - train_stats["max_drawdown"]
    factor_difference = test_stats["profit_factor"] - train_stats["profit_factor"]
    survives = bool(
        test_stats["return"] > 0
        and test_stats["profit_factor"] > 1.2
        and abs(test_stats["max_drawdown"]) < 10
    )

    print(f"\n{SEPARATOR}\nWALK FORWARD VALIDATION\n{SEPARATOR}")
    print("\nTRAIN\n")
    print(f"Universe: {', '.join(BACKTEST_SYMBOLS)}")
    print(f"Frozen Top 5: {', '.join(frozen_symbols)}")
    print(f"Trades: {train_stats['trades']}")
    print(f"Win Rate: {train_stats['win_rate']:.1f}%")
    print(f"Return: {train_stats['return']:+.1f}%")
    print(f"Profit Factor: {_factor_text(train_stats['profit_factor'])}")
    print(f"Max DD: {train_stats['max_drawdown']:.1f}%")
    print(f"Average Trade: {train_stats['average_trade']:.2f}%")
    print(f"Best Symbol: {train_stats['best_symbol']}")
    print(f"Worst Symbol: {train_stats['worst_symbol']}")

    print("\nTEST\n")
    print(f"Symbols: {', '.join(frozen_symbols)}")
    print(f"Trades: {test_stats['trades']}")
    print(f"Win Rate: {test_stats['win_rate']:.1f}%")
    print(f"Return: {test_stats['return']:+.1f}%")
    print(f"Profit Factor: {_factor_text(test_stats['profit_factor'])}")
    print(f"Max DD: {test_stats['max_drawdown']:.1f}%")
    print(f"Average Trade: {test_stats['average_trade']:.2f}%")
    print(f"Best Symbol: {test_stats['best_symbol']}")
    print(f"Worst Symbol: {test_stats['worst_symbol']}")

    print("\nDIFFERENCE (TEST − TRAIN)\n")
    print(f"Return: {return_difference:+.1f} percentage points")
    print(f"Drawdown: {drawdown_difference:+.1f} percentage points")
    print(f"Profit Factor: {factor_difference:+.2f}")
    print("\nVERDICT\n")
    print(
        "✅ Strategy generalizes well."
        if survives
        else "❌ Strategy is overfitted."
    )
    print(
        "The strategy survives unseen market data."
        if survives
        else "The strategy does not survive unseen market data under the required criteria."
    )
    print(f"\n{SEPARATOR}")
    return summary, survives


def _print_report(ranked_stats: list[dict[str, Any]]) -> None:
    print(f"\n{SEPARATOR}\nFULL MODE COMPARISON TABLE\n{SEPARATOR}")
    print(
        f"{'Mode':<15}{'Trades':>8}{'Win Rate':>11}{'Return':>10}"
        f"{'Max DD':>10}{'PF':>9}{'Avg Trade':>12}{'Best':>10}{'Worst':>10}"
    )
    for stats in ranked_stats:
        print(
            f"{stats['mode']:<15}{stats['trades']:>8}{stats['win_rate']:>10.1f}%"
            f"{stats['return']:>9.1f}%{stats['max_drawdown']:>9.1f}%"
            f"{_factor_text(stats['profit_factor']):>9}"
            f"{stats['average_trade']:>11.2f}%"
            f"{stats['best_symbol']:>10}{stats['worst_symbol']:>10}"
        )
    print(SEPARATOR)

    winner = ranked_stats[0]
    print("\n🏆 BEST MODE\n")
    print(f"{winner['mode']} — {winner['description']}")
    print(
        f"\nWhy it won: highest ranked by return ({winner['return']:+.1f}%), "
        f"then drawdown ({winner['max_drawdown']:.1f}%) and profit factor "
        f"({_factor_text(winner['profit_factor'])})."
    )
    print(f"Trade count: {winner['trades']}")
    print(f"Average score: {winner['average_score']:.1f}")
    print(f"Average volume ratio: {winner['average_volume']:.2f}x")
    print(f"Average RSI: {winner['average_rsi']:.1f}")
    print(
        "Best coins: "
        + (", ".join(winner["best_coins"]) if winner["best_coins"] else "N/A")
    )
    print(f"\n{SEPARATOR}")


def _classify_regime(btc_market: dict[str, Any], timestamp_ns: int) -> str:
    """Classify causally from completed BTC 4H EMA200 and its past slope."""
    position = int(
        btc_market["macro_times"].searchsorted(timestamp_ns, side="right") - 1
    )
    slope_lookback = 20
    if position < slope_lookback:
        return "SIDEWAYS"
    current = btc_market["macro"].iloc[position]
    prior = btc_market["macro"].iloc[position - slope_lookback]
    above_ema = bool(current["close"] > current["ema200"])
    below_ema = bool(current["close"] < current["ema200"])
    positive_slope = bool(current["ema200"] > prior["ema200"])
    negative_slope = bool(current["ema200"] < prior["ema200"])
    if above_ema and positive_slope:
        return "BULL"
    if below_ema and negative_slope:
        return "BEAR"
    return "SIDEWAYS"


def _regime_status(stats: dict[str, Any]) -> str:
    passed = bool(
        stats["return"] > 0
        and stats["profit_factor"] > 1
        and abs(stats["max_drawdown"]) < 15
    )
    return "✅ PASSED" if passed else "❌ FAILED"


def _run_regime_tests(
    prepared: dict[str, dict[str, Any]], timeline: list[int]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run unchanged ELITE entries independently in each causal BTC regime."""
    if "BTCUSDT" not in prepared:
        raise RuntimeError("BTCUSDT data is required for market regime classification")

    regimes = ("BULL", "BEAR", "SIDEWAYS")
    states = {
        regime: _new_state(regime, f"ELITE {regime} entries", _mode_5)
        for regime in regimes
    }
    candle_counts = {regime: 0 for regime in regimes}

    for timestamp_ns in timeline:
        snapshots = {}
        for symbol, market in prepared.items():
            snapshot = _snapshot(symbol, market, int(timestamp_ns))
            if snapshot is not None:
                snapshots[symbol] = snapshot
        ranked = sorted(
            snapshots.values(),
            key=lambda item: (item["score"], item["volume_ratio"]),
            reverse=True,
        )
        current_regime = _classify_regime(prepared["BTCUSDT"], int(timestamp_ns))
        candle_counts[current_regime] += 1

        # Open trades receive every candle after entry. The regime only gates
        # fresh entries, so transitions cannot delay or cherry-pick exits.
        for state in states.values():
            _close_position(state, snapshots)
        _open_best_exceptional(states[current_regime], ranked)

    files = {
        "BULL": BULL_RESULTS_FILE,
        "BEAR": BEAR_RESULTS_FILE,
        "SIDEWAYS": SIDEWAYS_RESULTS_FILE,
    }
    stats_by_regime = {}
    rows = []
    for regime in regimes:
        state = states[regime]
        _export_state_trades(state, files[regime])
        stats = _statistics(state)
        stats_by_regime[regime] = stats
        rows.append(
            {
                "regime": regime,
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "return_pct": stats["return"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown": stats["max_drawdown"],
                "average_trade_pct": stats["average_trade"],
                "best_coin": stats["best_symbol"],
                "worst_coin": stats["worst_symbol"],
                "status": _regime_status(stats),
            }
        )
    summary = pd.DataFrame(rows, columns=REGIME_SUMMARY_COLUMNS)
    summary.to_csv(REGIME_SUMMARY_FILE, index=False)

    ranked_regimes = sorted(
        regimes,
        key=lambda regime: (
            stats_by_regime[regime]["return"],
            stats_by_regime[regime]["max_drawdown"],
            stats_by_regime[regime]["profit_factor"],
        ),
        reverse=True,
    )
    strongest, weakest = ranked_regimes[0], ranked_regimes[-1]
    passed = {
        regime
        for regime in regimes
        if _regime_status(stats_by_regime[regime]).startswith("✅")
    }
    if passed == set(regimes):
        recommendation = "Trade always across the tested regimes."
    elif passed == {"BULL"}:
        recommendation = "Trade only in Bull markets."
    elif "BULL" in passed and "SIDEWAYS" in passed and "BEAR" not in passed:
        recommendation = "Trade Bull + Sideways and avoid Bear markets."
    elif "BEAR" in passed and "SIDEWAYS" in passed and "BULL" not in passed:
        recommendation = "Trade Bear + Sideways and avoid Bull markets."
    elif "BEAR" not in passed and passed:
        recommendation = "Avoid Bear markets; trade only the regimes that passed."
    else:
        recommendation = "Do not trade continuously; robustness criteria were not met."

    print(f"\n{SEPARATOR}\nMARKET REGIME TEST\n{SEPARATOR}")
    for regime in regimes:
        stats = stats_by_regime[regime]
        print(f"\n{regime}\n")
        print(f"Classified candles: {candle_counts[regime]}")
        print(f"Trades: {stats['trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Return: {stats['return']:+.1f}%")
        print(f"Profit Factor: {_factor_text(stats['profit_factor'])}")
        print(f"Max DD: {stats['max_drawdown']:.1f}%")
        print(f"Average Trade: {stats['average_trade']:.2f}%")
        print(f"Best Coin: {stats['best_symbol']}")
        print(f"Worst Coin: {stats['worst_symbol']}")
        print(f"Scorecard: {_regime_status(stats)}")

    print("\nFINAL VERDICT\n")
    print(f"Strongest regime: {strongest}")
    print(f"Weakest regime: {weakest}")
    print(f"Recommendation: {recommendation}")
    print(f"\n{SEPARATOR}")
    return summary, stats_by_regime, states


def run_backtest() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    histories = _download_universe()
    if not histories:
        raise RuntimeError("No historical market data could be downloaded")
    prepared = {
        symbol: _prepare_history(history) for symbol, history in histories.items()
    }

    reference_symbol = "BTCUSDT" if "BTCUSDT" in prepared else next(iter(prepared))
    reference_times = prepared[reference_symbol]["entry"]["close_time"].astype("int64")
    timeline = reference_times.tail(BACKTEST_CANDLES).iloc[250:].tolist()
    states = [_new_state(*definition) for definition in MODE_RULES]

    for timestamp_ns in timeline:
        snapshots = {}
        for symbol, market in prepared.items():
            snapshot = _snapshot(symbol, market, int(timestamp_ns))
            if snapshot is not None:
                snapshots[symbol] = snapshot
        ranked = sorted(
            snapshots.values(),
            key=lambda item: (item["score"], item["volume_ratio"]),
            reverse=True,
        )
        for state in states:
            _close_position(state, snapshots)
            _open_best_exceptional(state, ranked)

    statistics = [_statistics(state) for state in states]
    ranked_stats = sorted(
        statistics,
        key=lambda item: (
            item["return"],
            item["max_drawdown"],
            item["profit_factor"],
        ),
        reverse=True,
    )
    mode_order = {stats["mode"]: rank for rank, stats in enumerate(ranked_stats)}
    rows = [trade for state in states for trade in state["trades"]]
    results = pd.DataFrame(rows)
    if results.empty:
        export = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        results["_mode_rank"] = results["mode"].map(mode_order)
        export = results.sort_values(
            ["_mode_rank", "profit_pct"], ascending=[True, False]
        ).reindex(columns=RESULT_COLUMNS)
    export.to_csv(RESULTS_FILE, index=False)
    _print_report(ranked_stats)

    elite_state = next(state for state in states if state["name"] == "MODE_5_ELITE")
    symbol_report = _create_symbol_report(elite_state)
    symbol_report.to_csv(SYMBOL_REPORT_FILE, index=False)
    _print_symbol_report(symbol_report)

    filtered, _ = _run_filtered_groups(prepared, timeline, symbol_report)
    filtered.to_csv(FILTERED_RESULTS_FILE, index=False)
    _print_filtered_groups(filtered)

    _run_walkforward(prepared, reference_times)
    _, _, regime_states = _run_regime_tests(prepared, timeline)

    from visual_report import generate_visual_report

    generate_visual_report(elite_state, regime_states)
    return export, ranked_stats


if __name__ == "__main__":
    run_backtest()
