# Executor comparison — paper farmer (sim) vs demo-v2 live executor

Two ledgers pulled from droplet `167.172.88.229` on 2026-06-04:
- **Paper farmer (sim):** `volume_farmer_okx_okx-paper_state.json` → local `data/okx-paper_real_state.json`.
  179 trades, 2026-06-01 22:30 → 06-04 10:15 (2.49d). Old executor: no time-stop, all SLs taker.
- **demo-v2 (live executor):** `volume_farmer_okx_okx-demo-v2_state.json` → local `data/demo_v2_state.json`.
  55 trades, 2026-06-03 20:20 → 06-04 10:05 (0.57d). New: time-stop + maker-exit + entry-repeg.

See also [trade_analysis_demo_v2.md](trade_analysis_demo_v2.md) (demo-v2 deep dive) and
[trade_analysis.md](trade_analysis.md) (original 133-trade paper analysis).

## Headline: same market window, opposite outcome

Both bots traded the **identical** window 2026-06-03 20:20 → 06-04 10:05 UTC (apples-to-apples).

| metric | PAPER farmer (sim) | DEMO-V2 live executor |
| --- | --- | --- |
| trades | 40 | 55 |
| **sum net P&L** | **−$19.29** | **+$13.85** |
| win rate (net>0) | **75.0%** | 67.3% |
| avg win / avg loss | +$2.57 / **−$9.63** | +$3.66 / **−$6.75** |
| **RR** | **0.27** | **0.54** |
| profit factor | 0.80 | 1.11 |
| expectancy / trade | −$0.48 | +$0.25 |
| worst single | −$11.56 | −$13.99 |
| exits maker / taker | 30 / **10** | 51 / **4** |
| taker fee-overage | $4.35 | $1.41 |
| reasons | 30 tp / 9 sl / 1 sl_ambig | 33 tp / 18 time_stop / 4 sl |

**The paper sim wins 3 of 4 trades and still loses money; demo-v2 wins less often and makes
money.** RR beats win rate. ~$33 swing on the same candles in 14h.

### What each change bought

1. **Time-stop halved the average loss** (−$9.63 → −$6.75) and **doubled RR** (0.27 → 0.54).
   The 18 time-stops cut would-be deep losers at ~−$3.5 instead of letting them run to −$9…−$14.
2. **Maker-exit killed fee leakage** — 10 taker SL exits / $4.35 overage → 4 / $1.41.
3. **Cost:** win rate dropped 75% → 67% (time-stop force-closes some eventual winners). Net hugely
   positive. → motivates testing `max_hold` 2→3 to recover some red time-stops into TPs.
4. **More trades in the same window** (55 vs 40) → more volume/rebate (entry-repeg re-enters faster).

## Full-period (context, not apples-to-apples)

| metric | PAPER (2.49d, 179) | DEMO-V2 (0.57d, 55) |
| --- | --- | --- |
| sum net P&L | −$89.15 | +$13.85 |
| equity | $500 → $399.75 | $500 → $505.99 |
| total_pnl (state) | −$96.98 | +$13.85 |
| win rate | 74.3% | 67.3% |
| RR | 0.26 | 0.54 |
| profit factor | 0.74 | 1.11 |
| max drawdown | **−$109.51** | −$43.57 |
| exits maker/taker | 133 / 46 | 51 / 4 |
| taker overage | **$21.36** | $1.41 |
| total fees | $129.53 | $30.44 |
| rebate accrued | $53.31 | $12.26 |
| volume | $270,441 | $72,569 |

Paper ran 2.5 days straight into a −$97 / max-DD −$110 hole with $21 of pure taker fee leakage.

## Still open: the stop-loss

Even in the winning demo-v2 run, the 4 SLs that fire cost **−$54.89** (~−$14 each, −0.7% to
−1.6% adverse at 125× lev, all short, all taker). The time-stop catches slow bleeders but not fast
intrabar spikes into a too-wide SL band. **Lever #1: tighten SL to ~−0.5%** → projected ~+$42 (≈3×).

## Local files
- `data/okx-paper_real_state.json` — paper ledger (179)
- `data/demo_v2_state.json` — demo-v2 ledger (55)
- `data/okx-paper_state.json` — currently a copy of demo-v2 (the slot `export_trade_log.py` reads)
