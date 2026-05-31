# Volume Farmer Bot — Guide for Hirusha

> Written for someone from a software background who is new to crypto trading.  

---

## What Is This?

This is a **Python bot** that automatically opens and closes futures (leveraged) trades on the **MEXC crypto exchange** against the BTC/USDT pair.

It is **not** trying to predict whether Bitcoin goes up or down. The goal is to **generate as much trading volume as possible** — specifically $1,000,000 USD of volume in 30 days — because MEXC pays back a percentage of fees to high-volume traders (called a **rebate**).

Think of it like a cashback card: every time you swipe (trade), MEXC gives back 70% of what you paid in fees at end of month.

---

## Why Did We Choose This Strategy?

We tested 3 different entry strategies. Here is what we found (on real historical BTC data):

| Strategy | Win Rate | Notes |
|---|---|---|
| **RSI + WaveTrend** | ~38% | Technically sophisticated but too few trades, slow volume |
| **Bollinger Band Fade** | ~38% | Good in ranging markets, bad during trends |
| **Micro Momentum** ✅ | ~**90%** | Simple, high frequency, best volume throughput |

The Micro Momentum strategy has a **90% win rate** — not because it predicts price well, but because of how the trade bracket is set up (explained below). It also fires the most trades, which is exactly what we need for volume farming.

---

## How the Bot Works — Step by Step

### 1. Every 5 Minutes, a New Candle Closes

Bitcoin's price data comes in 5-minute "candles". Each candle has:
- `open` — price at start of 5 minutes
- `close` — price at end of 5 minutes  
- `high` — highest price during those 5 minutes
- `low` — lowest price during those 5 minutes

The bot wakes up every time a 5-minute candle closes.

### 2. Should We Enter a Trade?

Before entering, the bot checks:

**Gate 1 — Is a trade already open?**  
If yes → skip entry, just check if the open trade should be closed.

**Gate 2 — Are we in cooldown?**  
After 10 consecutive losses, the bot pauses for 12 bars (1 hour) to cool down.

**Gate 3 — Bar Range Filter**  
```
range = |close - open| / open × 10,000   (result is in "bps", basis points)
```
- If range < 4 bps → bar is too flat (nothing happening) → **skip**
- If range > 40 bps → bar is too explosive (likely a news spike) → **skip**
- Otherwise → **proceed**

A "bps" (basis point) = 0.01%. So 4 bps = 0.04% move. On BTC at $74,000, that is a $29.60 candle body. Very small — just filtering out completely dead bars.

**Gate 4 — Direction Signal (Micro Momentum)**

```python
if close > open:
    go LONG   # green candle → bet price keeps going up briefly
else:
    go SHORT  # red candle → bet price keeps going down briefly
```

**Gate 5 — Alternation**  
The bot alternates direction every trade regardless of the signal:
- Last trade was LONG → this one is forced SHORT
- Last trade was SHORT → this one is forced LONG

This keeps the bot balanced (net zero directional exposure over time). It does not care about overall BTC trend.

### 3. Opening the Trade

Once all gates pass, the bot opens the trade:

```
Equity:    $300
Margin:    $300 × 5% = $15    (only $15 at risk per trade)
Leverage:  100x               (calculated dynamically)
Notional:  $15 × 100 = $1,500 (actual position size)
Open fee:  $1,500 × 0.01% = $0.15  (maker order — cheap)
```

**Leverage** is not fixed — it is calculated using this formula from ebirth.net:
```
leverage = (equity × 2.5% risk) / (margin × SL%)
         = ($300 × 0.025) / ($15 × 0.005)
         = $7.50 / $0.075
         = 100x
```

As equity drops, leverage drops too. As equity grows, it grows. This keeps risk proportional.

### 4. Exit Conditions

Two targets are set when the trade opens:

| Name | Value | Meaning |
|---|---|---|
| **TP** (Take Profit) | +5 bps (+0.05%) | Close with a small win |
| **SL** (Stop Loss) | −50 bps (−0.50%) | Close with a larger loss |

Every subsequent 5-minute bar, the bot checks:
- Did the candle's **high** reach TP? → Close at TP (win)
- Did the candle's **low** reach SL? → Close at SL (loss)
- Did both happen in same bar? → Close at SL (assume worst case)
- Has the trade been held for 999 bars (83 hours) with neither hit? → Close at market (safety valve)

The trade can stay open for **multiple bars**. If BTC chops sideways for 30 minutes, the bot holds and waits. It does not force-close after 5 minutes.

---

## Why 90% Win Rate?

This is the key insight. Look at the asymmetry:

```
TP = +5 bps   (only needs a $37 move on BTC to win)
SL = −50 bps  (needs a $370 move to lose — 10× bigger)
```

On any given 5-minute bar, BTC is far more likely to tick up $37 than to crash $370. The 90% win rate is a **mathematical property of the bracket**, not a prediction skill.

The trade-off: when a loss happens, it is **10× bigger** than a win. The math:

```
Expected value per trade =
  0.90 × (+$0.45 win)  +  0.09 × (−$8.40 loss)  =  −$0.36 per trade
```

**Every trade bleeds $0.36 on average.** Over 334 trades = ~$120 total bleed.

This is fine. The rebate covers it.

---

## The Fee & Rebate Model

Every trade has two fees:

| Fee type | When | Rate |
|---|---|---|
| **Maker** (limit order) | Open + TP exits | 0.01% of notional |
| **Taker** (market order) | SL exits | 0.05% of notional |

MEXC pays back **70% of all fees** at end of month.

With 334 round trips:
```
Total fees paid:  ~$200
Rebate (70%):     ~$140
Net fees:         ~$60
```

The $140 rebate more than covers the ~$120 trade bleed. **That is the profit model** — the exchange pays you back more than the strategy loses.

---

## Money Flow Over 30 Days

```
Start:         $300.00
Trade bleed:  −$120.00  (334 trades × −$0.36 avg)
──────────────────────
End equity:   ~$180.00

End-of-month rebate: +$140.00
──────────────────────────────
Total:        ~$320.00  (+6.7% on $300)
```

Plus any tier bonuses, KOL referral split, or exchange incentive that the $1M volume qualifies for.

---

## Why $300 Capital Specifically?

| Capital | Notional/trade | Volume per trip | Trips needed | Time needed |
|---|---|---|---|---|
| $100 | $500 | $1,000 | 1,000 trips | >30 days ❌ |
| $200 | $1,000 | $2,000 | 500 trips | ~28 days ⚠️ |
| **$300** | **$1,500** | **$3,000** | **334 trips** | **~22 days ✅** |
| $500 | $2,500 | $5,000 | 200 trips | ~14 days ✅ |

$300 is the minimum that reliably hits $1M before the 30-day window closes, based on backtesting.

---

## Risk Controls (Safety Rails)

The bot has hard stops to protect capital:

| Limit | Threshold | What happens |
|---|---|---|
| Daily loss | −50% of equity in one day | Hard stop for that day |
| Max drawdown | −95% from starting equity | **Permanent halt** (session over) |
| Consecutive losses | 10 in a row | 1-hour cooldown pause |
| Volume target reached | $1,000,000 hit | **Clean stop** — sends Telegram alert |

In plain English: if something goes very wrong (e.g. BTC crashes 50% suddenly), the bot stops before it wipes the account. The 95% drawdown limit is intentionally lenient because the rebate model accepts capital bleed — but it still prevents a full wipeout.

---

## Telegram Alerts

The bot sends real-time messages to a Telegram chat. Example messages:

```
🟢 LONG BTC_USDT
  Entry:     $74,077
  Notional:  $1,500 (100x)
  TP:        $74,114  (+5 bps)
  SL:        $73,707  (−50 bps)
  Equity:    $299.85
  Volume:    $1,500 / $1,000,000

✅ LONG closed — TP hit
  Exit:      $74,114
  Net PnL:   +$0.45
  Equity:    $300.30
  Trips:     1 | Win rate: 100%

🎯 Milestone: 10% of volume target reached!
  Volume:    $100,000
  Equity:    $295.20
```

---

## Running the Bot

### Prerequisites

```bash
# Python 3.9+ required
cd /Users/rowneth/vGen/mexc-bot
source /Users/rowneth/vGen/.venv/bin/activate
pip install -r requirements.txt
```

### Check Config (Active Config)

File: `config/config_volume_farmer_optimal.yaml`

Key settings you might want to change:
```yaml
farmer:
  capital_usd: 300.0       # starting capital in USD
  tp_bps: 5.0              # take profit (don't change — this is the optimised value)
  sl_bps: 50.0             # stop loss   (don't change)
  entry:
    mode: micro_momentum   # the winning strategy
```

### Launch the Bot

```bash
bash scripts/_launch_optimal_farmer.sh
```

This runs in the background. It prints the PID (process ID). Example: `PID=61259`.

### Check If It's Running

```bash
# See logs in real-time
tail -f data/logs/vol_farm_300.out

# Confirm process is alive
ps -p <PID> -o pid,etime,stat,%cpu
```

### Stop the Bot

```bash
pkill -f run_volume_farmer
```

---

## File Map — Where Things Live

```
mexc-bot/
│
├── config/
│   └── config_volume_farmer_optimal.yaml  ← ACTIVE config ($300, 5bps TP)
│
├── scripts/
│   ├── _launch_optimal_farmer.sh          ← run this to start the bot
│   └── run_volume_farmer.py               ← the runner script (don't edit)
│
├── src/
│   └── execution/
│       └── volume_farmer.py               ← ALL the bot logic lives here
│
├── data/
│   ├── logs/
│   │   └── vol_farm_300.out               ← live log output
│   └── volume_farmer_optimal_state.json   ← session state (auto-saved)
│
└── tests/
    └── test_*.py                          ← run with: pytest tests/
```

---

## Strategies We Tested and Rejected

### RSI + WaveTrend (based on Pine Script indicators)
- **What it does:** Buys when RSI crosses back up from oversold (<30), sells when it crosses down from overbought (>70). Optionally requires WaveTrend (a momentum oscillator) to confirm.
- **Why rejected:** Only fires ~3–5 trades per day. At that rate we would need 200+ days to hit $1M. Also only 38% win rate, so expected value is negative without the volume to justify it.

### Bollinger Band Fade
- **What it does:** Buys when price touches the lower Bollinger Band (statistically "too low") and RSI confirms oversold. Shorts the upper band.
- **Why rejected:** Works well in sideways markets, but BTC trends strongly. During a trend, price walks along the band and the strategy takes repeated SL hits. 38% win rate. Volume throughput similar to RSI/WT — too slow.

### Micro Momentum ✅ (chosen)
- **What it does:** Simply follows the direction of each candle body (green = long, red = short), alternating every trade.
- **Why chosen:** 90% win rate due to the TP/SL asymmetry. Fires 15–20 trades per day. Hits $1M in ~22 days. Simple = fewer things to break.

---

## Common Questions

**Q: Is this risky?**  
The maximum you can lose is $300 (the starting capital). The 95% drawdown halt kicks in at $15 remaining, so realistically the worst case is losing ~$285. The probability of hitting that is extremely low given the 90% win rate.

**Q: Does it trade automatically 24/7?**  
Yes. It runs as a background process on the machine. As long as the computer is on and connected to the internet, it trades every 5 minutes.

**Q: What if the computer restarts?**  
Re-run `bash scripts/_launch_optimal_farmer.sh`. The bot reads `data/volume_farmer_optimal_state.json` to resume from where it left off (equity, volume progress, etc.).

**Q: What if I change the capital to more than $300?**  
Edit `capital_usd` in the config. Everything else (leverage, position size, fees) scales automatically. Higher capital = larger positions = faster volume = hits $1M sooner.

**Q: Does this work on other coins?**  
The code is written for BTC_USDT futures on MEXC. It could work on other liquid pairs (ETH, SOL) but the TP/SL values and range filters would need re-tuning via backtesting.

**Q: What is "paper trading" vs "live trading"?**  
Paper trading = simulated trades with real prices but fake money (no real orders sent to exchange). This bot is currently running in **paper mode** — it calculates as if trading but no actual orders are placed. Live mode would require API keys and a funded MEXC account.

---

## Running the Tests

```bash
cd /Users/rowneth/vGen/mexc-bot
source /Users/rowneth/vGen/.venv/bin/activate
pytest tests/ -v
```

All 44 tests should pass. Tests cover: indicators, risk limits, paper broker order logic, security (API key handling), and the full strategy pipeline.

---

*Last updated: April 2026 | Bot: VOL-FARM-300 | Target: $1,000,000 / 30 days*
