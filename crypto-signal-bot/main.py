"""Full live analytics and paper-trading cycle (no exchange orders)."""

from config import SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER
from market_regime_detector import detect_market_regime, print_market_regime
from opportunity_scanner import (
    save_opportunities,
    scan_opportunities,
    top_opportunities,
)
from position_tracker import (
    has_open_position,
    load_positions,
    open_position,
    print_portfolio_stats,
    update_open_positions,
    update_portfolio_analytics,
)
from risk_manager import can_open_trade, update_risk_report
from signal_logger import save_signal
from telegram_bot import (
    send_market_regime,
    send_open_trades,
    send_portfolio_summary,
    send_risk_manager,
    send_top_opportunities,
)


SEPARATOR = "=" * 50
ITEM_SEPARATOR = "-" * 50
BEST_FILTERED_GROUP = {
    "INJUSDT",
    "ATOMUSDT",
    "LINKUSDT",
    "ADAUSDT",
    "APTUSDT",
}


def _is_elite(opportunity) -> bool:
    """Apply the frozen ELITE entry rules without optimization."""
    return bool(
        opportunity["score"] >= 90
        and opportunity["volume_ratio"] > 1.5
        and opportunity["atr"] > opportunity["atr_avg"]
        and 48 <= opportunity["rsi"] <= 58
    )


def _signal_payload(opportunity, can_open: bool) -> dict:
    """Translate an opportunity row into the persistent signal schema."""
    rsi = float(opportunity["rsi"])
    volume = float(opportunity["volume_ratio"])
    trend_bullish = (
        opportunity["trend_4h"] == "BULLISH"
        and opportunity["trend_1h"] == "BULLISH"
    )
    momentum_score = 10 if 45 <= rsi <= 60 else (7 if 40 <= rsi < 45 else 4 if 60 < rsi <= 70 else 2)
    volume_score = 10 if volume > 1.5 else (8 if volume > 1.2 else 5 if volume > 1 else 2)
    trend_score = 10 if trend_bullish else 3
    return {
        "symbol": opportunity["symbol"],
        "signal": "LONG" if can_open else "WAIT",
        "confidence": int(opportunity["score"]),
        "price": float(opportunity["price"]),
        "atr": float(opportunity["atr"]),
        "ema50": float(opportunity["ema50"]),
        "ema200": float(opportunity["ema200"]),
        "rsi": rsi,
        "volume_ratio": volume,
        "trend": "UPTREND" if trend_bullish else "DOWNTREND",
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "market_score": round(float(opportunity["score"]) / 10, 1),
        "reason": list(opportunity["reasons"]),
    }


def _trade_payload(opportunity) -> dict:
    """Build an ATR-based virtual trade accepted by trade_manager."""
    entry = float(opportunity["price"])
    atr = float(opportunity["atr"])
    return {
        "symbol": opportunity["symbol"],
        "signal": "LONG",
        "price": entry,
        "entry": entry,
        "atr": atr,
        "stop_loss": entry - atr * SL_ATR_MULTIPLIER,
        "take_profit": entry + atr * TP_ATR_MULTIPLIER,
        "confidence": int(opportunity["score"]),
        "reason": list(opportunity["reasons"]),
    }


def print_opportunities(opportunities, limit: int = 10) -> None:
    """Print a readable ranked opportunity list."""
    print(f"\n{SEPARATOR}\nTODAY'S OPPORTUNITIES\n{SEPARATOR}")
    if opportunities.empty:
        print("\nNo symbols could be analyzed.")
        return

    for rank, (_, row) in enumerate(opportunities.head(limit).iterrows(), start=1):
        print(f"\n{rank}.\n")
        print(row["symbol"])
        print(f"\n{row['label']}")
        print(f"\nScore:\n{int(row['score'])}/100")
        print(f"\nPrice:\n{row['price']:.8g}")
        print("\nReasons:\n")
        for reason in row["reasons"]:
            print(reason)
        print(f"\n{ITEM_SEPARATOR}")

    picks = top_opportunities(opportunities)
    print("\nTOP PICKS\n")
    print(", ".join(picks) if picks else "None")
    print(f"\n{SEPARATOR}")


def main() -> None:
    regime_result = detect_market_regime()
    print_market_regime(regime_result)
    regime = regime_result["regime"]

    # Existing virtual positions are always maintained, even when new entries
    # are disabled by the current regime.
    update_open_positions()

    opportunities = scan_opportunities()
    save_opportunities(opportunities)
    trading_enabled = regime in {"BEAR", "SIDEWAYS"}

    for _, opportunity in opportunities.iterrows():
        eligible = bool(
            trading_enabled
            and opportunity["symbol"] in BEST_FILTERED_GROUP
            and _is_elite(opportunity)
        )
        save_signal(_signal_payload(opportunity, eligible))

    print_opportunities(opportunities)

    if regime == "BULL":
        print("\nTrading disabled in BULL regime")
    elif regime == "SIDEWAYS":
        print("\nTrade carefully")
    else:
        print("\nELITE scanner enabled")

    if trading_enabled:
        for _, opportunity in opportunities.iterrows():
            symbol = str(opportunity["symbol"])
            if symbol not in BEST_FILTERED_GROUP or not _is_elite(opportunity):
                continue
            payload = _trade_payload(opportunity)
            risk_allowed, risk_reason = can_open_trade(
                symbol, payload["entry"], payload["stop_loss"]
            )
            if not risk_allowed:
                print(f"Risk Manager skipped {symbol}: {risk_reason}")
            elif has_open_position(symbol):
                print(f"Open paper trade already exists for {symbol}")
            elif open_position(payload, regime):
                print(f"Paper trade opened for {symbol}")

    current_trades = load_positions()
    portfolio_stats = update_portfolio_analytics(current_trades)
    print_portfolio_stats(portfolio_stats)
    risk_status = update_risk_report()
    print(
        f"Risk Manager: {'ENABLED' if risk_status['risk_allowed'] else 'DISABLED'} "
        f"— {risk_status['reason']}"
    )

    # Notifications are deliberately last and non-blocking. Missing secrets or
    # Telegram errors never interrupt market analytics or paper-trade state.
    send_market_regime(regime_result)
    send_top_opportunities(opportunities)
    send_open_trades(current_trades)
    send_portfolio_summary(current_trades)
    send_risk_manager(risk_status)


if __name__ == "__main__":
    main()
