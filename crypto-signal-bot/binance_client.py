"""Small client for Binance's public market-data API."""

from typing import Any, Optional

import pandas as pd
import requests


KLINES_URL = "https://api.binance.com/api/v3/klines"
OUTPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
]


def _request_klines(
    symbol: str, interval: str, limit: int, end_time: Optional[int] = None
) -> list[list[Any]]:
    """Request one Binance page (the public endpoint allows at most 1,000)."""
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(
        KLINES_URL,
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _to_frame(payload: list[list[Any]], symbol: str, interval: str) -> pd.DataFrame:
    """Normalize raw Binance candles into the application's DataFrame format."""

    if not payload:
        raise ValueError(f"Binance returned no candles for {symbol} ({interval})")

    # Binance returns 12 values per candle. Only the requested market fields are
    # retained; positions 7-11 contain quote/trade/taker metadata.
    frame = pd.DataFrame(
        [row[:7] for row in payload],
        columns=OUTPUT_COLUMNS,
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame


def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch one page of public Binance candlesticks."""
    if not 1 <= limit <= 1000:
        raise ValueError("A single Binance kline request supports 1 to 1000 candles")
    return _to_frame(_request_klines(symbol, interval, limit), symbol, interval)


def get_historical_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch up to *limit* recent candles by paging backward through public data."""
    if limit < 1:
        raise ValueError("Historical candle limit must be positive")

    rows: list[list[Any]] = []
    end_time: Optional[int] = None
    while len(rows) < limit:
        page_limit = min(1000, limit - len(rows))
        page = _request_klines(symbol, interval, page_limit, end_time)
        if not page:
            break
        rows = page + rows
        end_time = int(page[0][0]) - 1
        if len(page) < page_limit:
            break

    if not rows:
        raise ValueError(f"Binance returned no historical candles for {symbol}")

    frame = _to_frame(rows, symbol, interval)
    return (
        frame.drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .tail(limit)
        .reset_index(drop=True)
    )
