# Crypto Opportunity Scanner

A public-market analytics application that ranks 20 liquid Binance spot symbols
and answers: **What are the best opportunities in the market right now?** It does
not force a trade when market quality is poor.

It uses **no real money**, requires **no Binance API keys**, and contains **no
order-execution code**. Signals are educational analytics, not financial advice.

## Requirements

- Python 3.9 or newer
- Internet access to Binance's public API

## Installation

```bash
cd crypto-signal-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Running

```bash
python main.py
```

Every run analyzes all configured symbols across 4H, 1H, and 15m timeframes,
sorts them by Opportunity Score, writes the current snapshot to
`opportunities.csv`, prints the top ten, and identifies five top picks. It does
not open paper trades automatically.

## Opportunity scoring

The score ranges from 0 to 100: 25 points for bullish 4H macro trend, 20 for
confirmed 1H trend, up to 20 for 15m RSI momentum, up to 20 for relative volume,
and either 15 or 5 points for ATR expansion. ATR expansion compares ATR14 with
its 20-candle rolling average. Labels range from `🔥 EXCEPTIONAL` (90+) through
`🚀 STRONG`, `✅ GOOD`, and `⚠ WATCH` to `IGNORE` below 60.

`opportunities.csv` is a sorted snapshot of the latest scan and includes prices,
scores, labels, indicator values, timeframe trends, and readable reasons.

## Paper trades and `trades.csv`

Paper trading simulates entries and exits for evaluating the strategy. No funds
are transferred and no exchange order is created. When the program first runs it
automatically creates `trades.csv`, which acts as the local trade ledger. Each row
contains the virtual entry, stop loss, take profit, latest price, status, result,
confidence, and profit percentage.

Only one open paper trade per symbol is allowed by default. Closed trades remain
in the CSV so results can be reviewed across future runs.

## Signal history and `signals.csv`

Every successful market analysis is stored in `signals.csv`. Each run normally
adds one row per symbol with its decision, indicator values, component scores,
market score, and human-readable reasons. If the script is run more than once in
the same minute, it will not add a duplicate row for the same symbol.

The two CSV files have separate jobs:

- `trades.csv` stores virtual paper trades and their lifecycle.
- `signals.csv` stores all market decisions, including `WAIT` signals.

## Historical backtesting

Run the independent historical simulation with:

```bash
python backtest.py
```

The elite backtester downloads historical data for the 20-symbol universe and
ranks all available symbols at every candle. It compares five increasingly strict
modes: scores of 85, 90, and 95; score 90 with strong volume; and ELITE mode with
strong volume, expanding ATR, and RSI from 48 to 58. A mode stays in cash unless
at least one opportunity passes its rule.

Each mode holds at most one virtual spot position. Position notional is capped at
the virtual balance, preventing implicit leverage. Trades are exported to
`elite_backtest_results.csv`, while the console prints the full mode comparison
and explains the best-ranked mode.

After the ELITE run, `symbol_report.csv` attributes trades, wins, losses, return
contribution, profit factor, drawdown, and best/worst trades to every configured
symbol. This shows which markets helped or hurt the strategy.

The same historical timeline is then replayed for four ELITE symbol filters: the
top five performers, the full universe without its worst five, a manual
high-quality list, and the best three performers. Their comparable statistics and
symbol membership are stored in `filtered_backtest_results.csv`.

## Walk-forward validation

The backtest also performs a strict 70/30 chronological split. ELITE runs on the
first 70% only and writes `symbol_report_train.csv`; its top five symbols are then
frozen. No symbols or parameters are re-selected when those five are tested on
the unseen final 30%.

Closed train and test trades are stored in `walkforward_train.csv` and
`walkforward_test.csv`. `walkforward_summary.csv` records both periods' symbols,
trade counts, win rates, returns, profit factors, drawdowns, average trades, and
best/worst symbols. Earlier candles may warm causal indicators, but future test
candles never influence training selection.

## Market regime testing

The unchanged ELITE strategy is also evaluated in BULL, BEAR, and SIDEWAYS
markets. Regimes are classified causally from BTC's latest completed 4-hour
candle: price above EMA200 with a positive 20-bar EMA slope is BULL; price below
EMA200 with a negative slope is BEAR; all other states are SIDEWAYS. No future
returns are used to label a candle.

New entries are assigned to the current regime, while existing trades continue
to receive every candle until exit. This avoids delayed exits at regime
boundaries. Trades are written to `bull_market_results.csv`,
`bear_market_results.csv`, and `sideways_market_results.csv`; comparable metrics
and pass/fail scorecards are saved in `market_regime_summary.csv`.

## Visual strategy report

The historical run also creates total and per-regime equity-curve CSVs and PNG
charts, plus a total drawdown chart. `trade_streaks.csv` summarizes consecutive
wins/losses, average and extreme trades, profit factor, expectancy, and a
Sharpe-like trade-return ratio. The plots use simple matplotlib defaults on a
white background and do not alter strategy execution.

## Current market regime detector

Run the standalone detector with:

```bash
python market_regime_detector.py
```

It analyzes the latest completed BTCUSDT 4-hour and daily candles using EMA50,
EMA200, RSI14, ATR14, and ADX14. Both timeframes must independently satisfy the
same BULL or BEAR conditions to confirm that regime; otherwise the result is
SIDEWAYS. The current 4-hour benchmark metrics and confirmed regime are written
to `market_regime.csv`. This is detection and scanner guidance only—no trades or
orders are created.

Additional pre-window candles are downloaded only to seed the derived 4-hour
EMA200. They are not counted as part of the configured 5,000-candle comparison
window, preventing an under-warmed macro indicator without shortening the test.

The current strategy uses 4-hour macro trend, 1-hour confirmation, and 15-minute
entries. Its stop loss is one ATR14 below entry and its take profit is two ATR14
above entry. ATR14 is the rolling 14-candle mean of True Range and is implemented
directly with pandas—no TA-Lib or opaque indicator package is used.

Backtesting is historical simulation only. It does not read or modify
`trades.csv`, does not write to `signals.csv`, uses no real money or API keys, and
cannot place Binance orders. Because 15-minute candles do not reveal intrabar
price ordering, a candle that touches both exit levels is counted using the
specified take-profit-first rule. Historical outcomes do not guarantee future
performance.

For the current strategy, one continuous `LONG` condition counts as one setup in
the simulator. It must return to `WAIT` before a new historical entry is allowed;
this prevents repeated entries on an unchanged signal.

## Safety boundary

This application:

- Uses only Binance's unauthenticated public market-data endpoint.
- Does not use or request Binance API keys.
- Does not contain order-execution code.
- Never connects to real money or a trading account.

Signals and simulated results are educational analytics, not financial advice.

## GitHub Actions Automation

The repository workflow `.github/workflows/run-bot.yml` runs every 15 minutes and can also
be started manually from the repository's **Actions** tab with
`workflow_dispatch`. It installs Python 3.11 dependencies, runs `python main.py`,
and commits the live paper, regime, risk, opportunity, and signal-performance
CSV files as `github-actions[bot]`. If none of those files changed, the workflow
exits successfully without creating an empty commit.

Every automated run detects the BTC regime, updates existing virtual positions,
scans all opportunities, records signals, and permits new ELITE paper trades only
for the frozen best-performing symbol group when the regime is BEAR or SIDEWAYS.
BULL disables new entries but still refreshes analytics. This remains paper
trading only: it uses no real money, API keys, futures, leverage, or order
execution.

## Telegram Dashboard

The optional Telegram dashboard sends plain-text updates after each complete run,
including market regime, opportunities, positions, portfolio, risk, and signal
performance. If either credential is missing, the bot prints
`Telegram disabled.` and continues normally.

1. Open [@BotFather](https://t.me/BotFather), run `/newbot`, choose a name and a
   username ending in `bot`, and securely copy the token. Telegram's
   [official BotFather guide](https://core.telegram.org/bots/features#botfather)
   documents this process.
2. Open your new bot and send it a message. Visit
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`, then find
   `message.chat.id` in the JSON response. This is `TELEGRAM_CHAT_ID`. Keep the
   token private; anyone holding it can control the bot.
3. In GitHub, open **Settings → Secrets and variables → Actions**, create
   repository secrets named `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, and use
   the exact values above. The workflow passes them only to the `main.py` step.

The implementation uses Telegram's official HTTP
[Bot API](https://core.telegram.org/bots/api) `sendMessage` endpoint. Telegram
notifications report paper analytics only and cannot execute orders or access an
exchange account.

## Position Tracker

`position_tracker.py` owns the live paper-position lifecycle. Every row in
`trades.csv` moves from `OPEN` to `TP_HIT`, `SL_HIT`, or `MANUAL_CLOSE`; each run
refreshes current price, unrealized PnL, and duration for open LONG or SHORT
positions. On closure, time, price, duration, and PnL are frozen. Duplicate open
positions for the same symbol remain blocked.

`portfolio_summary.csv` records balance, open/closed counts, win rate, profit
factor, drawdown, average trade, and return after every run. The tracker also
generates `portfolio_equity.png`, `portfolio_open_positions.png`, and
`portfolio_pnl_distribution.png`. Telegram's `📊 Portfolio` message uses these
same portfolio calculations, so console, CSV, charts, and notifications agree.

All positions are virtual analytics. The tracker has no API keys, account
connection, leverage, futures, or order-execution capability.

## Risk Manager

`risk_manager.py` gates every new paper position after ELITE qualification and
before persistence. It risks 1% of current paper balance using
`position_size = (balance × 0.01) / abs(entry − stop_loss)`, allows at most three
open positions, caps total notional exposure at 30%, and disables entries after
a 3% daily loss or three consecutive closed losses.

Every run appends the current decision and reason to `risk_report.csv` and
regenerates `risk_exposure.png`, `daily_pnl.png`, and
`consecutive_losses.png`. Telegram receives the same balance, exposure, daily
loss, streak, and enabled/disabled decision. These controls protect only virtual
paper capital and do not modify the ELITE strategy or connect to an exchange.

## Signal Performance Tracking

`signal_performance_tracker.py` automatically snapshots every current
opportunity into `signal_history.csv`. Exact ELITE opportunities are recorded as
`LONG`; all other rows are retained as `IGNORE` observations and excluded from
trade-signal statistics. The compound duplicate key is symbol, UTC creation
hour, signal, and entry price.

Every scheduled run freezes the first available public price observation after
24 hours, 48 hours, and 7 days, then calculates direction-aware PnL and WIN/LOSS
results. `signal_performance_summary.csv` is an append-only cumulative report of
evaluation counts, win rates, average PnL, and best/worst signals. Telegram sends
the latest summary, or `Not enough data yet.` until at least one valid signal has
reached an evaluation horizon.

After 30 days, review the latest cumulative row alongside individual entries in
`signal_history.csv` to compare short- and medium-horizon behavior. This remains
public-data observation only: no API keys, futures, leverage, or real orders.
