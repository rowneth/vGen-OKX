# vGen Bot — Current Strategy, Logic & Math (for external AI review)

> Self-contained brief of what the bot is **actually** doing right now, taken
> directly from the live code (`src/execution/volume_farmer.py`) and the
> active config (`config/config_volume_farmer_optimal.yaml`).
> No marketing claims — just what is implemented.

---

## 1. Goal & Profit Thesis

This is **not** a directional/alpha bot. It is a **volume-farming bot** that
trades BTC/USDT perpetual futures on **MEXC** to qualify for the exchange's
**70% maker/taker fee rebate** program.

The strategy is *expected* to bleed PnL on the trades themselves; the profit
comes from the rebate paid back at month end. Specifically:

- Each trade has a tiny TP and a much larger SL.
- High win rate (~85–90%) by design of the bracket geometry.
- When a loss hits, it is ~6–10× the size of a win → expected value per trade
  is mildly negative.
- That bleed is more than recovered by the rebate on the volume traded.

Active capital target (current config):

| Param | Value |
|---|---|
| Starting capital | **$30 USD** |
| Volume target | **$100,000 USD** |
| Rebate assumption | **70%** of fees paid |

(Earlier config used $300 → $1M; this is the scaled-down test.)

---

## 2. Architecture (high level)

```
                  ┌───────────────────────────┐
 MEXC 5m candles  │   live_feed (WebSocket)   │
 ────────────────►│   + reconnect/heartbeat   │
                  └───────────┬───────────────┘
                              │ closed bar
                              ▼
                  ┌───────────────────────────┐
                  │  VolumeFarmerSession      │
                  │  (bar-driven state mach.) │
                  │                           │
                  │  on_new_candle(history):  │
                  │   1. halt checks          │
                  │   2. if in pos → check    │
                  │      TP/SL/trend-break/   │
                  │      time-stop intrabar   │
                  │   3. else → entry gates → │
                  │      open new position    │
                  └───────────┬───────────────┘
                              │ FarmerEvent (entry/exit/halt/milestone)
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
        Telegram alerts   JSON state     SQLite audit
                          (resume)       (decisions/orders/fills)
```

Key isolation:
- Volume farmer is **fully decoupled** from the older Bollinger mean-reversion
  stack so the experiment can't corrupt that path.
- Live order placement (`execution/live_broker.py`) currently **paper-only**
  in this session; an executor (`execution/live_volume_executor.py`) exists
  for real orders behind hard gates.
- All signals act **on bar close** (no intrabar entries).

---

## 3. The active entry strategy: **Micro Momentum + Forced Alternation**

Ordered set of gates run on every closed 5-minute bar:

### Gate 1 — Position check
If a position is already open, skip entry; only run exit logic.

### Gate 2 — Cooldown
After **10 consecutive losses**, pause for **12 bars (1 hour)**.
```python
if consec_losses >= 10:
    cooldown_bars_left = 12
    consec_losses = 0   # reset
```

### Gate 3 — Bar range filter
```
bar_range_bps = |close - open| / open * 10_000
```
- Skip if `bar_range_bps < 4` → bar too dead
- Skip if `bar_range_bps > 40` → bar too explosive (likely news spike)

(4 bps = 0.04%; 40 bps = 0.40%.)

### Gate 4 — Direction signal (Micro Momentum)
```python
bias = "long" if close > open else "short"
```
i.e. follow the body of the just-closed candle.

### Gate 5 — Forced alternation
```python
if alternate_direction and last_side == bias:
    bias = flip(bias)
```
Forces strict L/S/L/S/... regardless of signal. Net directional exposure
across the session is ~0.

If all gates pass → open position at the close price of the just-closed bar.

---

## 4. Position sizing — dynamic leverage (ebirth.net formula)

For each new trade:

```
margin   = equity × margin_fraction_per_trade   (0.05 → 5% of equity)
risk_$   = equity × risk_per_trade_pct          (0.025 → 2.5% of equity)
sl_frac  = sl_bps / 10_000                      (0.005 for 50 bps)

leverage = risk_$ / (margin × sl_frac)
leverage = clamp(leverage, min_leverage=5, max_leverage=125)

notional = margin × leverage
```

**Worked example with current $30 equity:**
```
margin   = 30 × 0.05 = $1.50
risk_$   = 30 × 0.025 = $0.75
sl_frac  = 50 / 10_000 = 0.005
leverage = 0.75 / (1.50 × 0.005) = 100x
notional = 1.50 × 100 = $150
```

So at $30 equity, every trade puts ~$150 notional on the wire.

The leverage **scales with equity**: as equity drops, leverage drops, keeping
the dollar risk per trade pinned to 2.5% of current equity.

---

## 5. Bracket: TP / SL (current config)

| Param | bps | % | $ on $150 notional |
|---|---|---|---|
| `tp_bps` | **8** | 0.08% | +$0.12 gross |
| `sl_bps` | **50** | 0.50% | −$0.75 gross |

```python
# long
tp = entry × (1 + 0.0008)
sl = entry × (1 - 0.0050)
# short
tp = entry × (1 - 0.0008)
sl = entry × (1 + 0.0050)
```

**Why this asymmetry produces high WR:** on any 5-minute bar BTC is far more
likely to wiggle ±8 bps than to range ±50 bps. The 85–90% WR is a
*geometric property of the bracket*, not predictive skill.

---

## 6. Exit logic (in priority order, every closed bar)

Checked intrabar against the **just-closed candle's high/low**:

1. **Both TP and SL hit in same bar** → assume worst case, exit at SL
   (`reason: "sl_ambiguous"`).
2. **TP hit** → exit at TP price (`reason: "tp"`).
3. **SL hit** → exit at SL price (`reason: "sl"`).
4. **Trend-break early exit** (cuts losers before full SL):
   - Enabled.
   - Only after `bars_held >= 3` (15 minutes minimum).
   - Triggers if adverse move ≥ **20 bps** at the close of the current bar.
   - Exits at `close` (`reason: "trend_break"`).
   - This improved End+Rebate from $20.51 → $23.60 in backtest (+15%).
5. **Time stop**: exit at close after `max_hold_bars = 999`
   (effectively disabled — let TP/SL resolve).

---

## 7. Fee model & rebate (confirmed from live MEXC trades)

| Fee | Rate | When |
|---|---|---|
| **Maker** | 0.01% (0.0001) | Open (POST_ONLY limit) |
| **Taker** | 0.05% (0.0005) | All exits — TP, SL, trend-break, time-stop |
| **Rebate** | 70% of total fees | Paid at month end |

**Important:** On MEXC futures, the attached TP/SL trigger orders execute as
**market** orders → always taker. So a "winning" trade still pays:
```
round_trip_fee_bps = maker_open + taker_close
                   = 1 bps + 5 bps
                   = 6 bps
```

This is why `tp_bps=8` is the practical floor — TP must clear 6 bps to be
gross-positive.

---

## 8. Per-trade economics (current $30, 8/50 bracket)

Win path:
```
notional       = $150
gross_pnl_win  = +0.0008 × 150  = +$0.120
open_fee       = 0.0001 × 150   = -$0.015
close_fee      = 0.0005 × 150   = -$0.075
net_pnl_win    = +$0.120 - $0.090 = +$0.030
```

Loss path (full SL):
```
gross_pnl_loss = -0.0050 × 150  = -$0.750
fees           = same -$0.090
net_pnl_loss   = -$0.840
```

Loss path (trend-break at -20 bps):
```
gross_pnl_loss = -0.0020 × 150  = -$0.300
fees           = -$0.090
net_pnl_tb     = -$0.390
```

**Expected value per trade** (assuming 88% WR and ~half the losses caught
by trend-break):
```
EV = 0.88 × (+0.030)
   + 0.06 × (-0.390)         # trend-break losses
   + 0.06 × (-0.840)         # full SL losses
   = +0.0264 - 0.0234 - 0.0504
   ≈ -$0.047 per trade
```

**Volume per trade**: $150 × 2 (open + close) = **$300 per round trip.**

To hit $100,000 volume target: **~334 round trips**.

**Total expected bleed**: 334 × $0.047 ≈ **−$15.7**
**Total fees gross**: 334 × ($150 × 0.0006) ≈ **$30.06**
**Rebate (70%)**: ≈ **+$21.04**
**Net result**: −$15.7 + $21.04 ≈ **+$5.3** on $30 capital + any tier bonuses.

(Numbers are sensitive to actual WR; real backtests showed $+23.60 net at the
prior $300 / $1M scale.)

---

## 9. Risk halts (hard stops in code)

| Halt | Threshold | Behaviour |
|---|---|---|
| `daily_loss_limit_pct` | 50% of starting equity per day | Hard halt for the day |
| `max_drawdown_pct` | 95% from peak equity | **Permanent session halt** |
| `consecutive_losses_limit` | 10 in a row | 12-bar (1h) cooldown |
| `stop_on_volume_target` | volume ≥ $100k | Clean halt + Telegram |

These are set deliberately **lenient** because the rebate model expects
capital bleed — the halts are catastrophe protection only.

---

## 10. Indicators implemented (used by alternative entry modes only)

The codebase ships three entry modes; only `micro_momentum` is active.

```
SMA, EMA, RSI, ATR, Bollinger Bands, WaveTrend (LazyBear-style)
```

### Alternative entry mode A — `rsi_wt` (rejected, kept in code)
Long when `RSI[-2] < 30 and RSI[-1] >= 30` (crossback up).
Short on the mirror.
Optional WaveTrend confirmation (cross + zone) within last N bars.
Backtest: ~38% WR, 3–5 trades/day → far too slow for volume target.

### Alternative entry mode B — `bollinger_fade` (rejected, kept in code)
```
Long  : close <= lower_BB AND RSI < 35 AND BB_width in [20, 200] bps
Short : close >= upper_BB AND RSI > 65 AND BB_width in [20, 200] bps
[Optional EMA(200) trend filter on top]
```
Backtest: ~38% WR, hurt badly during BTC trends.

### Active mode — `micro_momentum` (chosen)
Section 3 above. ~85–90% WR, 15–20 trades/day, hits $1M in ~22 days at
$300 capital. Simplest possible signal.

---

## 11. State persistence

- File: `data/volume_farmer_optimal_state.json`
- Saves: equity, peak_equity, volume, fees, wins/losses, round_trips,
  consec_losses, cooldown, halt flag, milestones hit, last 500 ledger rows.
- Loaded on startup → resumes mid-month if process dies.

---

## 12. Full active config (verbatim from `config_volume_farmer_optimal.yaml`)

```yaml
exchange:
  symbol: BTC_USDT
  timeframe: 5m

fees:
  maker: 0.0001        # 0.01%
  taker: 0.0005        # 0.05%
  rebate_pct: 0.70

farmer:
  capital_usd: 30.0
  leverage: 0          # 0 = dynamic
  margin_fraction_per_trade: 0.05
  sizing:
    dynamic_leverage: true
    risk_per_trade_pct: 0.025
    max_leverage: 125
    min_leverage: 5
  tp_bps: 8.0
  sl_bps: 50.0
  max_hold_bars: 999
  entry:
    mode: micro_momentum
    min_bar_range_bps: 4.0
    max_bar_range_bps: 40.0
  alternate_direction: true
  trend_break:
    enabled: true
    min_bars_held: 3
    adverse_bps: 20.0

risk:
  daily_loss_limit_pct: 0.50
  max_drawdown_pct: 0.95
  consecutive_losses_limit: 10
  consecutive_losses_cooldown_bars: 12
  stop_on_volume_target: true

target:
  volume_usd: 100000.0
  min_fee_cover_pct: 0.40
```

---

## 13. Known weaknesses (honest list — these are what we want help on)

1. **Negative trade EV** — the strategy *requires* the rebate to be net
   positive. If MEXC ever changes the rebate %, the bot becomes a slow
   drainer.
2. **No edge in entry signal** — `close > open → long` is essentially noise;
   the WR comes purely from bracket geometry, not predictive skill.
3. **Forced alternation is arbitrary** — it cancels even the small momentum
   signal on every other trade.
4. **Trend-break threshold (20 bps) was tuned on one symbol/period**; could
   be overfit.
5. **Range filter (4 / 40 bps) is static** — doesn't adapt to volatility
   regime (could use ATR-relative bands).
6. **All exits are taker** on MEXC futures (attached triggers can't be
   limits). 5 bps on every exit is a structural drag; we have not explored
   manually-placed limit-TPs as a workaround.
7. **No multi-symbol** — single-symbol concentration risk; ETH/SOL might
   have better bps-per-bar ratios at lower fees on other venues.
8. **WR sensitivity** — a 5-point WR drop (88 → 83) flips trade EV from
   −$0.05 to ~−$0.10/trade, doubling the bleed and possibly outrunning the
   rebate at smaller capital.
9. **No regime detection** — strategy runs identically in trending vs.
   ranging vs. high-vol regimes; backtests show very different per-regime WR.
10. **Time-stop is effectively off** (`max_hold_bars=999`) — a stuck
    sideways position can sit indefinitely tying up margin.

---

## 14. What to ask the external AI

Suggested framing:

> "This bot's profit depends on a 70% fee rebate covering negative-EV
> bracket trades. Win rate (~88%) comes from TP/SL asymmetry, not
> prediction. Given the math in section 8, can you suggest:
> (a) entry filters that would lift WR above 90% without killing volume
>     throughput,
> (b) a smarter dynamic TP/SL that adapts to BTC realised volatility
>     (e.g. ATR-scaled brackets) while preserving the bracket-WR effect,
> (c) ways to reduce the structural taker-fee drag on exits,
> (d) regime detection that disables trading during conditions where the
>     bracket WR collapses?"

---

*Generated from live source: `src/execution/volume_farmer.py` +
`config/config_volume_farmer_optimal.yaml`. All numbers and code paths
verified against the repo at the time of writing.*
