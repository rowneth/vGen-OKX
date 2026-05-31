#!/usr/bin/env bash
# Run all 8 backtests and generate comparison report.
# Must be run from /Users/rowneth/vGen/bot
set -e
cd "$(dirname "$0")/.."
echo "=== Running 8 backtests ==="

DATA5M="data/historical/BTC_USDT_5m.parquet"
DATA1M="data/historical/BTC_USDT_1m.parquet"
RESULTS="results"

echo ""
echo "--- baseline 5m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_optimal.yaml \
  --data "$DATA5M" \
  --output "$RESULTS/baseline_5m.json"

echo ""
echo "--- baseline 1m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_optimal.yaml \
  --data "$DATA1M" \
  --output "$RESULTS/baseline_1m.json"

echo ""
echo "--- sl30 5m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_sl30.yaml \
  --data "$DATA5M" \
  --output "$RESULTS/sl30_5m.json"

echo ""
echo "--- sl30 1m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_sl30.yaml \
  --data "$DATA1M" \
  --output "$RESULTS/sl30_1m.json"

echo ""
echo "--- mtf 5m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_mtf.yaml \
  --data "$DATA5M" \
  --output "$RESULTS/mtf_5m.json"

echo ""
echo "--- mtf 1m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_mtf.yaml \
  --data "$DATA1M" \
  --output "$RESULTS/mtf_1m.json"

echo ""
echo "--- sl30_mtf 5m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_sl30_mtf.yaml \
  --data "$DATA5M" \
  --output "$RESULTS/sl30_mtf_5m.json"

echo ""
echo "--- sl30_mtf 1m ---"
python3 scripts/backtest_volume_farmer.py \
  --config config/config_volume_farmer_sl30_mtf.yaml \
  --data "$DATA1M" \
  --output "$RESULTS/sl30_mtf_1m.json"

echo ""
echo "=== All 8 backtests complete. Generating comparison report... ==="
python3 scripts/compare_backtests.py \
  "$RESULTS/baseline_5m.json" \
  "$RESULTS/baseline_1m.json" \
  "$RESULTS/sl30_5m.json" \
  "$RESULTS/sl30_1m.json" \
  "$RESULTS/mtf_5m.json" \
  "$RESULTS/mtf_1m.json" \
  "$RESULTS/sl30_mtf_5m.json" \
  "$RESULTS/sl30_mtf_1m.json" \
  --output "$RESULTS/comparison.md"

echo "Done."
