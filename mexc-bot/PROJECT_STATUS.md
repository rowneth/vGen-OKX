# MEXC Bot Project Status and Next Steps

Date: 2026-04-19
Repository: /Users/rowneth/vGen/mexc-bot

## 1) What has been built so far

This project has been scaffolded and advanced through a complete Phase 1 backtesting baseline plus Phase 2 paper-trading infrastructure scaffolding.

### A. Project structure and baseline setup
- Full directory structure created for exchange, data, strategy, risk, execution, backtest, monitoring, tests, and scripts.
- Python dependency list created and populated.
- Configuration files created with externalized strategy and risk parameters.
- Environment templates created for credentials and runtime flags.

### B. Core strategy and analytics
- Pure indicator implementations completed:
  - SMA
  - EMA
  - RSI
  - ATR
  - Bollinger Bands
- Bollinger mean-reversion strategy implemented with:
  - Long and short mirrored logic
  - Confirmation candle behavior
  - Trend and volatility filters
  - ATR-based stop and middle-band take-profit handling

### C. Exchange integration and safety
- Async REST client implemented for MEXC futures.
- Signing logic for authenticated requests implemented.
- Historical kline retrieval implemented.
- Startup API permission verification pipeline implemented with fail-closed behavior:
  - Requires futures-trade permission when configured
  - Requires withdrawal-disabled status when configured
  - Refuses startup when permission status cannot be verified

### D. Data pipeline
- Historical downloader implemented for BTC_USDT 15m candles.
- MEXC payload normalization implemented for multiple response shapes.
- Parquet storage and loading implemented.
- 12-month historical dataset already downloaded locally and usable.

### E. Backtest engine and reporting
- Event-driven backtest engine implemented with:
  - Maker and taker fee modeling
  - Slippage in ticks (+ extra low-volume tick)
  - Probabilistic maker fills
  - Spread filter proxy logic
  - Daily drawdown and consecutive-loss pause logic
- Metrics implemented:
  - Sharpe
  - Max drawdown
  - Win rate
  - Profit factor
  - Monthly returns
  - Trades/day distribution
  - Win-rate by month
  - Stress test with win-rate reduction
- Report generation implemented:
  - Markdown report
  - HTML report with visualizations
- SQLite audit persistence implemented for:
  - Decisions
  - Orders
  - Fills

### F. Phase 2 paper infrastructure
- Live feed WebSocket wrapper implemented with reconnect and heartbeat timeout.
- Order manager implemented for post-only paper order lifecycle and stale-order cancellation.
- Monitoring implemented:
  - Structured logging with sensitive-value redaction
  - Alerts module (console, telegram, email sinks)
  - Lightweight dashboard server with health and state endpoints
- Paper runner wired to startup checks, feed loop, order manager, alerts, and dashboard.

### G. Test suite
- Unit tests implemented and passing for:
  - Indicators
  - Strategy signal behavior
  - Risk sizing and limits
  - Paper broker behavior
  - Security permission parsing and fail-closed checks

## 2) Why each major decision was made

### A. Fail-closed security at startup
Reason:
- Real-money systems should not assume permissions are safe.
- If permission status is ambiguous, refusing to start is safer than guessing.

### B. Externalized config values
Reason:
- Keeps strategy/risk behavior auditable and tunable.
- Avoids hidden magic constants in execution paths.

### C. Event-driven backtester with realistic frictions
Reason:
- Gross PnL without fees/slippage/fill realism is misleading.
- Better to fail strategy assumptions early than overfit on idealized fills.

### D. SQLite audit trail
Reason:
- Required for forensic debugging, reproducibility, and trust.
- Enables post-mortem analysis of every decision, order, and fill.

### E. Separation of paper and live execution
Reason:
- Prevents accidental transition from simulation to real trading.
- Maintains phased rollout discipline and risk containment.

## 3) Current status by rollout phase

### Phase 1 (Backtesting)
Status: Completed baseline implementation and validated execution flow.

Ready now:
- Run backtest from downloaded data
- Generate markdown/html reports
- Persist audit logs to SQLite

### Phase 2 (Paper trading)
Status: Infrastructure implemented, but strict startup permission checks may block runtime if exchange responses do not expose explicit permission flags.

What this means:
- The system is behaving safely.
- A practical paper-only override path may be needed to continue live-feed behavior testing when permission metadata is unavailable.

### Phase 3 (Tiny live)
Status: Intentionally not implemented.

Current blocker by design:
- Live broker remains disabled and refuses order placement.

## 4) Known limitations right now

1. Permission verification for paper may halt when exchange payloads do not include explicit permission flags.
2. No real live order placement code exists (intentional safety gate).
3. Some spread and fill assumptions in backtest are simplified proxies.
4. Full strategy-to-paper signal-to-order orchestration still needs tightening for long-running operational sessions.

## 5) What to do next (recommended order)

## Priority 1: Add paper-only permission override mode (safe by default)
Goal:
- Keep strict checks for live.
- Allow controlled paper execution when permission metadata is not provided by the exchange endpoint.

Deliverables:
- Config switch for paper-only override
- Explicit warning logs when override is active
- Tests proving live mode remains fail-closed

## Priority 2: Harden paper session orchestration
Goal:
- Make paper loop robust for multi-day runtime.

Deliverables:
- Heartbeat-loss reaction path and halt behavior
- Better event normalization for feed payload variations
- Graceful reconnect state recovery
- Order manager reconciliation checks

## Priority 3: Improve backtest realism and diagnostics
Goal:
- Narrow gap between backtest and paper behavior.

Deliverables:
- More explicit fill queue/depth approximation
- Spread model from higher-fidelity market snapshots
- Regime-specific performance slices

## Priority 4: Prepare tiny-live implementation (still disabled by default)
Goal:
- Build live broker safely behind hard gates.

Deliverables:
- Post-only limit entry flow
- Emergency-only market exits
- Notional and leverage hard caps
- Explicit multi-step runtime confirmation gate

## 6) How to run what is already working

From project root:
/Users/rowneth/vGen/mexc-bot

Backtest:
- /Users/rowneth/vGen/.venv/bin/python scripts/run_backtest.py --report-prefix run_001

Historical download refresh:
- /Users/rowneth/vGen/.venv/bin/python scripts/download_data.py --months 12

Paper runner (currently strict permission checks):
- /Users/rowneth/vGen/.venv/bin/python scripts/run_paper.py --max-messages 100

## 7) Current outputs location

- Historical parquet:
  - data/historical/BTC_USDT_15m.parquet
- Reports:
  - data/reports/
- Audit database:
  - data/bot_audit.db
- Paper runtime logs:
  - data/logs/paper.log

## 8) Safety reminders

1. Never store real keys in any example file.
2. Keep real keys only in local environment file that is git-ignored.
3. Keep withdrawal permission disabled on exchange keys.
4. Do not enable live trading until paper behavior is stable for multiple weeks.

## 9) Suggested checkpoint goal before moving forward

Run 1-2 weeks of paper sessions with:
- No unhandled runtime errors
- Stable reconnect and heartbeat handling
- Decision-to-fill audit consistency
- Similar behavior profile between backtest and paper

Once that is stable, proceed to tiny-live implementation with strict hard caps.
