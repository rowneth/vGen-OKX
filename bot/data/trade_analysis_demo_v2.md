# Trade Log Analysis — demo-v2 live executor (55 trades)

Source: droplet `volume_farmer_okx_okx-demo-v2_state.json` (live OKX **demo** executor, new
time-stop + maker-exit + entry-repeg logic). State saved 2026-06-04T10:15Z.
Period: **2026-06-03 20:20 → 2026-06-04 10:05 UTC** (13h 45m).
CSV: [all_trades.csv](all_trades.csv)

This is a different bot from the 133-trade paper run in [trade_analysis.md](trade_analysis.md):
that one was the *simulated farmer* (RR 0.25, −$88, all-taker SLs). This is the **live demo
executor** with the maker-exit + time-stop rewrite — and it behaves much better.

## Headline

| metric | value |
| --- | --- |
| trades closed | 55 (33 TP / 18 time_stop / 4 SL) |
| win rate (net > 0) | 67.3% (37/55) |
| avg win net | **+$3.66** |
| avg loss net | **−$6.75** |
| RR (avg win / avg loss) | **0.54** (paper run was 0.25) |
| expectancy / trade | **+$0.25** |
| profit factor | **1.11** (thin but positive) |
| sum net P&L (ledger) | **+$13.85** |
| equity | $500.00 → $505.99 (peak $546.75) |
| max drawdown | −$43.57 (−8.7% of start) |
| longest win / loss streak | 9 / 4 |
| total volume | $146,186 |
| fees gross | $30.65 · rebate accrued **$12.26** · net fee after rebate $18.18 |
| close fills | **51 maker / 4 taker** · taker-overage only **$1.41** |

The maker-exit + time-stop rewrite flipped a −$88 / RR-0.25 bleeder into a thin-positive
+$13.85 / RR-0.54 system. The remaining problem is the **stop-loss**, not the exits or fees.

## Where the P&L came from (by reason)

| reason | n | net $ | avg $ | avg gross $ | avg hold | avg move (favor) | win% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tp | 33 | **+132.41** | +4.01 | +4.29 | 6.7m | +0.32% | 100% |
| time_stop | 18 | **−63.66** | −3.54 | −3.29 | 10.0m | −0.28% | 22% |
| sl | 4 | **−54.89** | **−13.72** | −13.14 | 8.8m | **−1.20%** | 0% |

`33 TP (+132) − 14 red time-stops (−67) − 4 SL (−55) + 4 green time-stops (+3) = +13.85`

## The structural issues (in priority order)

### 1. The 4 stop-losses do 41% of the damage in 7% of the trades — **fix the SL first**
```
20:55 short  move -0.732%  net -$13.64  hold 5m   taker
03:20 short  move -1.304%  net -$13.99  hold 10m  taker
08:40 short  move -1.209%  net -$13.83  hold 10m  taker
09:15 short  move -1.570%  net -$13.44  hold 10m  taker
```
At 125× leverage, these −0.7% to −1.6% adverse moves cost ~−$14 each. The time-stop catches
the *slow* adverse moves at ~−0.39% (red time-stops only lose ~−$4.76), but these 4 spiked
**intrabar** ("INTRABAR sl detected" in the logs) and hit the SL before the 2-bar window — and
the SL band is wide enough that each loss is ~3× a normal loser. **All four are shorts; longs
took zero SLs.**

**Fix:** tighten the SL band to ~−0.5%. Those 4 trades would cost ~−$6.7 each (~−$27) instead of
−$54.89 → net would rise from +$13.85 to **~+$42 (≈3×)**. Highest-value knob by far.

### 2. Red time-stops are the largest single loss bucket (−$66.62)
14 red time-stops, avg −$4.76, avg adverse move −0.39%. These are positions force-closed at the
2-bar (`max_hold=2`, 10 min) limit before hitting TP. The time-stop is doing its job (it's why
adverse moves stop at −0.39% instead of becoming −1.2% SLs), but the window may be too tight in
chop. Worth A/B testing **`max_hold=3 bars`** — some of these need one more bar to reach TP, but
loosening risks turning time-stops back into SLs. Test, don't assume.

### 3. Hour 04 UTC is a dead zone
Hour 04 = −$21.76 (4 trades, 100% time_stop). Hours 02–04 all red. Same Asia→London handoff
window the paper analysis flagged. Consider blocking entries 02–04 UTC or halving notional.

### 4. Day-2 chop regime
2026-06-03 (17 trades): **+$29.20**. 2026-06-04 (38 trades): **−$15.35**, driven by 15 of 38
going time_stop — a regime where TP isn't getting hit inside 2 bars. Relates to #2.

### 5. Short side carried all the SL damage
Long: +$13.97 (24 trades, 0 SL, 10 time_stop). Short: −$0.12 (31 trades, 4 SL, 8 time_stop).
Opposite of the paper run (where longs were worse). Small sample (4 SLs), but if it persists,
look at the short-side SL placement specifically.

## By hour UTC (entry)

| hr | n | net $ | sl% | ts% |
| --- | --- | --- | --- | --- |
| 00 | 2 | −2.82 | 0 | 100 |
| 01 | 5 | **+20.98** | 0 | 0 |
| 02 | 3 | −7.09 | 0 | 67 |
| 03 | 4 | −10.39 | 25 | 50 |
| 04 | 4 | **−21.76** | 0 | 100 |
| 05 | 5 | +7.10 | 0 | 40 |
| 06 | 4 | +5.40 | 0 | 25 |
| 07 | 3 | +12.05 | 0 | 0 |
| 08 | 3 | −8.94 | 33 | 33 |
| 09 | 5 | −9.89 | 20 | 20 |
| 20 | 4 | −2.03 | 25 | 0 |
| 21 | 4 | +8.01 | 0 | 25 |
| 22 | 4 | +11.72 | 0 | 25 |
| 23 | 5 | +11.50 | 0 | 20 |

## Don't touch — these are working

- **Maker exits:** 51/55 maker, taker-overage just $1.41 (paper run leaked ~$17). The maker-exit
  path fixed the fee bleed.
- **Rebate engine:** $146k volume → $12.26 rebate accrued in 13.75h — nearly equal to the $13.85
  trading profit. For a volume farmer the combined edge (trading + rebate) is the real return.
  Equity reads $505.99 not $513.85 because $6.87 of rebate was already transferred to the wallet
  — that's a transfer, not a loss.

## Suggested next test pass

1. SL band → ~−0.5% (cap the −$14 SLs). Biggest lever.
2. `max_hold` 2 → 3 bars, measure red-time-stop rate and SL conversion.
3. Block entries 02–04 UTC (or halve notional).
4. Re-pull this ledger after another ~24h and re-run to confirm on a bigger sample (55 trades is
   thin; the 4-SL conclusion especially needs more data).
