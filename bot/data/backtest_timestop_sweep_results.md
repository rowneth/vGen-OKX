# Backtest — time-stop / SL / TP sweep (mirrors live demo-v2 executor)

Tool: [scripts/backtest_okx_timestop_sweep.py](../scripts/backtest_okx_timestop_sweep.py)
Data: `data/historical/BTC_USDT_5m.parquet` — 102,945 bars, 2025-05-01 → 2026-04-27 (360d).
Model: ATR-rel TP=tp_mult×ATR, SL=sl_mult×ATR; maker TP + maker time-stop + taker SL (+slippage);
OKX maker 2bps / taker 5bps / 40% rebate; **fixed $500 sizing (no compounding)** so per-trade
economics are comparable and a negative edge doesn't spiral to ruin and corrupt the averages.
SL-slippage = 5 bps (realistic; zero-slip checked as a floor).

## Headline: nothing is profitable, and "tighter SL + max_hold=3" is the worst row

| config (TP/SL/hold) | trades | WR% | RR | gross bps/t | TRUEnet (incl 40% rebate) | BE-rebate |
| --- | --- | --- | --- | --- | --- | --- |
| **CURRENT** 0.5 / 1.5 / 2 | 21,076 | 55.0% | 0.31 | **−1.32** | **−$12,274** | 131% |
| max_hold=3 only 0.5/1.5/3 | 18,729 | 62.1% | 0.25 | −1.24 | −$10,940 | 128% |
| **PROPOSED** 0.5 / 1.0 / 3 | 19,940 | 59.0% | 0.25 | **−2.04** | **−$14,814** | 142% |
| RR-rebalance 1.5 / 1.5 / 3 | 15,365 | 41.7% | 0.78 | −1.12 | −$8,837 | 125% |
| no time-stop (old paper) 0.5/1.5/999 | 15,246 | 73.7% | 0.15 | −1.27 | −$10,693 | 129% |

Zero-slippage floor (best case) still loses everywhere (−$7.7k to −$12.2k).

## Why — the entry has no edge

- **Gross price P&L (pre-fee) is −1.1 to −2.0 bps/trade regardless of SL/max_hold/TP.** `micro_momentum`
  (enter long on up-candle close) on 5m BTC is a coin flip that drifts slightly against you. Exit
  tuning only rearranges how the fees are paid — it cannot create edge.
- **Break-even rebate = 125–142%.** Even with fees *fully* refunded the strategy loses (gross is
  −$4,152), so no rebate tier (max 100%) can rescue it. The strategy is rebate-driven but 40% (or
  any %) can't cover a negative gross edge.
- Tightening SL (1.5→0.8) trips far more SLs (2,027 → 7,981), each taker + slippage → monotonically
  worse. Raising max_hold barely moves it. Raising TP toward SL (Block B) lifts RR 0.25→0.78 but
  per-trade gross stays ~−1.2 bps — it just loses slower by trading less.

## Reconciling with the live +$13.85 (demo-v2, 55 trades, 14h)

Live ran +$0.47/trade (incl rebate); backtest expects ≈ −$0.58/trade. The ~$1/trade gap over 55
trades is ≈1–2 standard errors — **the live profit is within noise of a zero-edge strategy on a
mildly favorable window.** 55 trades cannot distinguish edge from luck; 21,000 trades say no edge.

## Caveats

- Backtest period (2025-05→2026-04) ≠ live window (2026-06). Momentum works in trends, loses in chop.
- Entry modeled as maker fill at bar close; real `entry_repeg`/taker-fallback differs. Real SLs
  gapped worse than 5 bps slippage → real result likely worse, not better.
- Says nothing about whether a *better entry* could profit — only that this one doesn't, and no exit
  knob fixes it.

## When to stop the hold (data-proven max_hold)

Tool: [scripts/analyze_hold_horizon.py](../scripts/analyze_hold_horizon.py) — single forward-walk per
entry (15,247 entries), evaluates every max_hold + a per-bar hazard. TP 0.5×ATR / SL 1.5×ATR.

| max_hold | min | net bps/t | WR% | RR | tp/ts/sl |
| --- | --- | --- | --- | --- | --- |
| **1 bar** | 5 | **−3.04** (best) | 49% | 0.56 | 40/56/4 |
| 2 (current) | 10 | −3.65 | 58% | 0.40 | 53/36/10 |
| 3 (proposed) | 15 | −3.94 | 64% | 0.32 | 61/24/15 |
| 6 | 30 | −4.35 | 70% | 0.24 | 69/9/22 |
| none | — | −4.67 (worst) | 74% | 0.20 | 74/0/26 |

**Monotonic — every extra bar held loses more.** Optimum = shortest hold (1 bar). WR rises with hold
(49%→74%) but net falls, because avg loss widens (−13→−41 bps) faster than wins are added (1:3 TP:SL).

Hazard: **40% of winners hit TP on bar 1**; positions still open after that average −6→−12 bps drift
(winners left, losers marinate). Marginal value of holding is negative from bar 2 on. A monotonic
"cut ASAP" curve is itself the signature of a no-edge entry — an entry with real alpha would have an
interior optimum. Even the best hold (1 bar) still loses −3.04 bps/trade.

## Implication

Don't ship tighter-SL/max_hold=3 — it's worse, and the hold-horizon proves the best you can do by
tuning the stop is shave ~0.6 bps (still a loss). The lever is the **entry** (add a directional
filter: HTF trend / EMA regime / funding / orderflow), or treat this purely as volume generation and
only run it if a points/rebate program is worth ~$0.8/trade of bleed.
