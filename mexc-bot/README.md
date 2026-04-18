# MEXC Futures Bollinger Bot (Scaffold)

This repository is a clarity-first, safety-first scaffold for a production-grade BTC/USDT perpetual futures bot on MEXC.

The strategy is Bollinger Band mean reversion on 15-minute candles with strict risk controls, realistic backtesting assumptions, and a phased rollout path:

1. Backtest only
2. Paper trade
3. Tiny live
4. Gradual scaling

Live order placement is intentionally blocked in this session.

## Safety Principles

- No hardcoded strategy magic numbers in code: parameters live in [config/config.yaml](config/config.yaml)
- No credentials in source control: template only in [config/.env.example](config/.env.example)
- No API key/signature logging
- Hard risk limits are built into configuration and engine logic
- Live broker is not implemented yet on purpose

## Current Scope (Implemented)

- Async MEXC REST client with public/private request support and request signing
- Fail-closed startup API permission checks for paper/live runners
- Historical kline downloader and Parquet caching
- Pure indicator functions with unit tests:
	- SMA, EMA, RSI, ATR, Bollinger Bands
- Bollinger mean reversion signal logic
- Event-driven backtest engine with:
	- Maker/taker fees
	- Tick-based slippage with low-volume penalty
	- Probabilistic maker fills
	- Daily drawdown and loss-streak circuit breaker handling
- Backtest metrics and markdown/html reporting
- SQLite audit persistence for every backtest decision, order, and fill

## Project Structure

See the scaffold under [src](src) and [scripts](scripts) for separated responsibilities:

- [src/exchange](src/exchange): MEXC API client, models, rate limiting
- [src/data](src/data): historical/live data handling and storage
- [src/strategy](src/strategy): indicators, filters, strategy logic
- [src/risk](src/risk): sizing and risk limits
- [src/execution](src/execution): paper/live broker interfaces
- [src/backtest](src/backtest): engine, metrics, reports
- [tests](tests): unit tests for deterministic logic

## Install

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy [config/.env.example](config/.env.example) to `.env`.
2. Fill API key/secret only when you need private endpoints.
3. Keep withdrawal permissions disabled on the exchange account.

## Usage

### 1. Download Historical Data

```bash
python scripts/download_data.py --months 12
```

Expected output file:

- `data/historical/BTC_USDT_15m.parquet`

### 2. Run Backtest

```bash
python scripts/run_backtest.py --rows 5000 --report-prefix sample_run
```

Expected report outputs:

- `data/reports/sample_run.md`
- `data/reports/sample_run.html`

Expected audit output:

- `data/bot_audit.db`

## Backtest Modeling Notes

- Entry mode: post-only limit logic with probabilistic maker fills
- Fees: configured maker/taker rates from config
- Slippage:
	- Base 1 tick per fill
	- +1 tick under low volume
- Position sizing: fixed fractional risk using stop distance and leverage cap
- Exits:
	- Take profit at middle band
	- ATR stop loss
	- Time stop after configured candle count

## Phased Rollout Plan

### Phase 1: Backtest (current)

- Validate positive expectancy with realistic assumptions
- Evaluate monthly consistency and stress-test resilience

### Phase 2: Paper Trading

- Consume live market data
- Route orders to paper broker only
- Compare behavior to backtest outcomes

### Phase 3: Tiny Live

- Enable live broker with strict notional caps
- Focus on operational reliability, not PnL

### Phase 4: Controlled Scaling

- Increase size only after sustained profitable live consistency

## Notes

- This scaffold prioritizes auditability and debuggability over compactness.
- Live broker implementation remains intentionally disabled.
