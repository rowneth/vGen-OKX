#!/bin/bash
# Launch the $300 optimal volume farmer in the background.
cd /Users/rowneth/vGen/mexc-bot
nohup /Users/rowneth/vGen/.venv/bin/python scripts/run_volume_farmer.py \
  --config config/config_volume_farmer_optimal.yaml \
  --state-file data/volume_farmer_optimal_state.json \
  --label VOL-FARM-300 \
  --duration-days 30 \
  >> data/logs/vol_farm_300.out 2>&1 &
echo "PID=$!"
