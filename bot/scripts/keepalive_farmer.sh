#!/usr/bin/env bash
# Keep-alive supervisor for the volume farmer — local equivalent of the
# droplet's systemd Restart=always. The bot already saves state and sends a
# crash card on unhandled exceptions; this wrapper guarantees it comes back.
#
# Usage:
#   ./scripts/keepalive_farmer.sh <label> [extra runner args...]
# Example (demo rehearsal):
#   nohup ./scripts/keepalive_farmer.sh okx-v3-demo --demo > /dev/null 2>&1 &
# Example (REAL money, after LIVE_OKX_ACK=I_UNDERSTAND in .env):
#   nohup ./scripts/keepalive_farmer.sh okx-v3-live --live > /dev/null 2>&1 &
#
# Restart policy: 15s pause normally; exponential-ish backoff (max 5 min)
# when the bot dies within 60s of starting, so a hard config error cannot
# turn into a tight crash loop that spams Telegram and the OKX API.
set -u
cd "$(dirname "$0")/.."

LABEL="${1:?usage: keepalive_farmer.sh <label> [args...]}"
shift
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="$LOG_DIR/keepalive_${LABEL}.log"
BOT_LOG="$LOG_DIR/${LABEL}.out"
BACKOFF=15

echo "$(date '+%F %T') keepalive: supervising label=$LABEL args=$*" >> "$SUPERVISOR_LOG"
while true; do
    START=$(date +%s)
    .venv/bin/python scripts/run_volume_farmer_okx.py \
        --config config/config_volume_farmer_okx_v3.yaml \
        --label "$LABEL" \
        --duration-days 365 \
        "$@" >> "$BOT_LOG" 2>&1
    CODE=$?
    RAN=$(( $(date +%s) - START ))
    echo "$(date '+%F %T') keepalive: bot exited code=$CODE after ${RAN}s" >> "$SUPERVISOR_LOG"
    # Clean operator stop (SIGINT/SIGTERM propagated -> exit 0 or 130): stay down.
    if [ "$CODE" -eq 0 ] || [ "$CODE" -eq 130 ] || [ "$CODE" -eq 143 ]; then
        echo "$(date '+%F %T') keepalive: clean exit — not restarting" >> "$SUPERVISOR_LOG"
        exit 0
    fi
    if [ "$RAN" -lt 60 ]; then
        BACKOFF=$(( BACKOFF * 2 )); [ "$BACKOFF" -gt 300 ] && BACKOFF=300
    else
        BACKOFF=15
    fi
    echo "$(date '+%F %T') keepalive: restarting in ${BACKOFF}s" >> "$SUPERVISOR_LOG"
    sleep "$BACKOFF"
done
