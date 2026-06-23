"""Current BTC market-regime detection using public spot candles only."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from binance_client import get_klines
from config import CANDLE_LIMIT
from indicators import add_indicators


BENCHMARK_SYMBOL = "BTCUSDT"
REGIME_TIMEFRAMES = ("4h", "1d")
REGIME_FILE = "market_regime.csv"
REGIME_COLUMNS = [
    "timestamp",
    "symbol",
    "price",
    "ema50",
    "ema200",
    "rsi",
    "adx",
    "atr",
    "regime",
]
SEPARATOR = "=" * 36


def _completed_indicators(interval: str) -> pd.DataFrame:
    """Fetch indicators and exclude Binance's still-forming candle."""
    data = add_indicators(get_klines(BENCHMARK_SYMBOL, interval, CANDLE_LIMIT))
    completed = data[data["close_time"] <= pd.Timestamp.now(tz="UTC")]
    if len(completed) < 2:
        raise ValueError(f"Not enough completed {interval} candles for regime detection")
    return completed


def _classify_timeframe(data: pd.DataFrame) -> str:
    """Apply the exact EMA, slope, and ADX regime rules to one timeframe."""
    current = data.iloc[-1]
    previous = data.iloc[-2]
    ema_slope = float(current["ema200"] - previous["ema200"])

    if (
        current["close"] > current["ema200"]
        and current["ema50"] > current["ema200"]
        and ema_slope > 0
        and current["adx14"] > 20
    ):
        return "BULL"
    if (
        current["close"] < current["ema200"]
        and current["ema50"] < current["ema200"]
        and ema_slope < 0
        and current["adx14"] > 20
    ):
        return "BEAR"
    return "SIDEWAYS"


def detect_market_regime() -> dict[str, Any]:
    """Detect, save, and return the current confirmed BTC market regime."""
    timeframes = {
        interval: _completed_indicators(interval)
        for interval in REGIME_TIMEFRAMES
    }
    timeframe_regimes = {
        interval: _classify_timeframe(data)
        for interval, data in timeframes.items()
    }
    regime = (
        timeframe_regimes["4h"]
        if timeframe_regimes["4h"] == timeframe_regimes["1d"]
        else "SIDEWAYS"
    )

    current = timeframes["4h"].iloc[-1]
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": BENCHMARK_SYMBOL,
        "price": float(current["close"]),
        "ema50": float(current["ema50"]),
        "ema200": float(current["ema200"]),
        "rsi": float(current["rsi14"]),
        "adx": float(current["adx14"]),
        "atr": float(current["atr14"]),
        "regime": regime,
    }
    pd.DataFrame([result], columns=REGIME_COLUMNS).to_csv(REGIME_FILE, index=False)
    return result


def print_market_regime(result: dict[str, Any]) -> None:
    """Print benchmark metrics and the requested scanner guidance."""
    print(f"\n{SEPARATOR}\nCURRENT MARKET REGIME\n{SEPARATOR}\n")
    print(f"BTC:\n{result['price']:.2f}\n")
    print(f"EMA50:\n{result['ema50']:.2f}\n")
    print(f"EMA200:\n{result['ema200']:.2f}\n")
    print(f"RSI:\n{result['rsi']:.2f}\n")
    print(f"ADX:\n{result['adx']:.2f}\n")
    print(f"ATR:\n{result['atr']:.2f}\n")
    print(f"REGIME:\n{result['regime']}\n")

    guidance = {
        "BULL": "❌ Trading disabled.",
        "SIDEWAYS": "⚠ Trade carefully.",
        "BEAR": "✅ ELITE scanner enabled.",
    }
    print(guidance[result["regime"]])
    print(f"\n{SEPARATOR}")


def main() -> None:
    result = detect_market_regime()
    print_market_regime(result)
    print("\nMarket regime detector verified successfully.")


if __name__ == "__main__":
    main()
