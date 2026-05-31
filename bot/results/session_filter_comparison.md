# Session Filter Comparison Results

Date: 2026-04-29
Data: BTC/USDT 5m, 360 days (2025-05-01 → 2026-04-27)
Capital: $30 (matches production)

## Config Comparison

| Config                       |  Trades |   WR% |  AdjBps |  End+Reb |  AdjNet/mo |    Vol/30d | Block |
|------------------------------|---------|-------|---------|----------|------------|------------|-------|
| Baseline                     |   1,037 |  78.9 |  -0.387 | $  23.72 | $     0.18 | $    8,815 |  0.0% |
| Filter-2h [1,6]              |   1,055 |  79.2 |  -0.210 | $  24.47 | $     0.27 | $    9,114 | -1.7% |
| Filter-4h [1,6,12,22]        |   1,048 |  79.7 |  -0.121 | $  24.87 | $     0.31 | $    9,275 | -1.1% |

## Decision Rule Evaluation

### Filter-2h [1, 6]
- [✓] End+Rebate ≥ baseline × 1.02
- [✓] WR within 2pp of baseline
- [✓] Trades ≥ 85% of baseline
- [✓] Net bps (adj) > baseline net bps (adj)

Verdict: **PASS**

### Filter-4h [1, 6, 12, 22]
- [✓] End+Rebate ≥ baseline × 1.03
- [✓] WR within 2pp of baseline
- [✓] Trades ≥ 80% of baseline
- [✓] Net bps (adj) > baseline net bps (adj)
- [✓] Net bps (adj) > Filter-2h net bps (adj)

Verdict: **PASS**

## Recommendation

**Filter-4h**

Reason: Both pass; picked higher adj net bps

Expected improvement over baseline: +4.87%

## Next Steps

1. Chosen config: `config/config_volume_farmer_filter4h.yaml`
2. Deploy to live trading at $30 capital
3. Run 200 trades minimum before evaluating
4. If live WR diverges by >25% from backtest WR, pause and investigate