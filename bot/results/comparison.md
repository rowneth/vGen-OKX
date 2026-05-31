# Backtest Comparison Report

```
========================================================================
  BACKTEST COMPARISON REPORT
========================================================================
  Date range 5m: ~360 days  (1,037 trades)
  Date range 1m: ~25 days  (982 trades)

  ── Timeframe: 5m ──────────────────────────────────────────
  Config                               Trades    WR%  WR-gap  end+reb    DD%  avgL  MTFskip
  ──────────────────────────────────────────────────────────────────────
  Baseline (TP=8/SL=50, alternate=on)   1,037  78.9%   -7.3% $  23.72  95.1% 28.6        - HALTED
  SL30 only (TP=8/SL=30, alternate=on)     721  74.3%   -4.6% $  20.76  95.1% 25.5        - HALTED
  MTF only  (TP=8/SL=50, MTF=on)          938  77.0%   -9.2% $  19.33  95.1% 28.2      464 HALTED
  SL30+MTF  (TP=8/SL=30, MTF=on)          713  73.3%   -5.6% $  17.80  95.1% 24.5      415 HALTED

  ── Timeframe: 1m ──────────────────────────────────────────
  Config                               Trades    WR%  WR-gap  end+reb    DD%  avgL  MTFskip
  ──────────────────────────────────────────────────────────────────────
  Baseline (TP=8/SL=50, alternate=on)     982  76.8%   -9.4% $  23.19  94.9% 26.1        -
  SL30 only (TP=8/SL=30, alternate=on)     772  73.8%   -5.1% $  21.79  95.0% 23.1        - HALTED
  MTF only  (TP=8/SL=50, MTF=on)            0   0.0%  -86.2% $  30.00   0.0%  0.0    4,786
  SL30+MTF  (TP=8/SL=30, MTF=on)            0   0.0%  -79.0% $  30.00   0.0%  0.0    3,589

  ── DECISION RULE EVALUATION (5m 12-month data) ─────────────
  Rules set in advance — results applied without modification.

  CONFIG: SL30 only
    PASS conditions (all required):
      [✗] wr >= 80%
      [✗] end+reb >= base+10%
      [✓] dd <= base+5pp
      [✓] trades >= 50% base
      [✗] avg_loss <= 25bps
    FAIL conditions (none must trigger):
      [✗] wr >= 78% (NOT fail)
      [✗] end+reb >= baseline
      [✓] dd <= base+10pp
      [✓] trades >= 30% base
      [✓] 1m/5m WR within 5pts
    vs baseline: end+rebate -2.95 (-12.5%)
    ➜ RECOMMENDATION: ✗ REJECT — fail condition triggered

  CONFIG: MTF only
    PASS conditions (all required):
      [✗] wr >= 80%
      [✗] end+reb >= base+10%
      [✓] dd <= base+5pp
      [✓] trades >= 50% base
      [✗] avg_loss <= 25bps
    FAIL conditions (none must trigger):
      [✗] wr >= 78% (NOT fail)
      [✗] end+reb >= baseline
      [✓] dd <= base+10pp
      [✓] trades >= 30% base
      [✗] 1m/5m WR within 5pts
    vs baseline: end+rebate -4.39 (-18.5%)
    ➜ RECOMMENDATION: ✗ REJECT — fail condition triggered

  CONFIG: SL30+MTF (proposed live)
    PASS conditions (all required):
      [✗] wr >= 80%
      [✗] end+reb >= base+10%
      [✓] dd <= base+5pp
      [✓] trades >= 50% base
      [✓] avg_loss <= 25bps
    FAIL conditions (none must trigger):
      [✗] wr >= 78% (NOT fail)
      [✗] end+reb >= baseline
      [✓] dd <= base+10pp
      [✓] trades >= 30% base
      [✗] 1m/5m WR within 5pts
    vs baseline: end+rebate -5.92 (-25.0%)
    ➜ RECOMMENDATION: ✗ REJECT — fail condition triggered

========================================================================
  NOTE: 1m results use ~25-day window (MEXC API cap).
  5m results use ~12 months. Deploy decisions based on 5m only.
========================================================================
```
