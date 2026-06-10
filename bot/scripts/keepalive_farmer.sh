#!/usr/bin/env bash
# Keep-alive supervisor for the volume farmer — local equivalent of the
# droplet's systemd Restart=on-failure. The bot already saves state and sends
# a crash card on unhandled exceptions; this wrapper guarantees it comes back,
# forwards operator signals, refuses to run twice, and alerts via Telegram if
# the bot crash-loops at startup (when the bot itself can't send anything).
#
# Usage:
#   ./scripts/keepalive_farmer.sh <label> [extra runner args...]
# Demo:  nohup ./scripts/keepalive_farmer.sh okx-v3-demo --demo > /dev/null 2>&1 &
# LIVE:  nohup ./scripts/keepalive_farmer.sh okx-v3-live --live > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1

LABEL="${1:?usage: keepalive_farmer.sh <label> [args...]}"
shift
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="$LOG_DIR/keepalive_${LABEL}.log"
BOT_LOG="$LOG_DIR/${LABEL}.out"
LOCK_DIR="data/keepalive_${LABEL}.lock"
BACKOFF=15
FAST_EXITS=0

log() { echo "$(date '+%F %T') keepalive: $*" >> "$SUPERVISOR_LOG"; }

# Single-instance lock (mkdir is atomic and works on macOS, unlike flock).
# Two supervisors = two bots on one account = doubled exposure + 409s.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    OLD_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo '')"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "keepalive already running (pid $OLD_PID) — refusing to start twice" >&2
        log "refused second instance (existing pid $OLD_PID)"
        exit 1
    fi
    log "stale lock found (pid ${OLD_PID:-unknown} dead) — taking over"
fi
echo $$ > "$LOCK_DIR/pid"
BOT_PID=""
cleanup() {
    # Forward the stop to the bot (SIGINT → graceful state save + stop card),
    # wait for it, then release the lock. Killing the supervisor must never
    # orphan a live trading process.
    log "supervisor stopping — forwarding SIGINT to bot pid ${BOT_PID:-none}"
    [ -n "$BOT_PID" ] && kill -INT "$BOT_PID" 2>/dev/null
    [ -n "$BOT_PID" ] && wait "$BOT_PID" 2>/dev/null
    rm -rf "$LOCK_DIR"
    exit 0
}
trap cleanup TERM INT

tg_alert() {
    # Supervisor-level Telegram alert, independent of the bot (used when the
    # bot dies too early to say anything itself). Reads creds from .env.
    local MSG="$1"
    local TOK CHAT
    TOK="$(grep -E '^TELEGRAM_BOT_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2-)"
    CHAT="$(grep -E '^TELEGRAM_CHAT_ID=' .env 2>/dev/null | head -1 | cut -d= -f2-)"
    [ -n "$TOK" ] && [ -n "$CHAT" ] && curl -sS -m 10 \
        "https://api.telegram.org/bot${TOK}/sendMessage" \
        -d chat_id="${CHAT}" -d text="${MSG}" >/dev/null 2>&1
}

log "supervising label=$LABEL args=$*"
while true; do
    # Cap the raw stdout capture (the runner keeps its own full log file via
    # FileHandler; this is just crash-context, so rotate aggressively).
    if [ -f "$BOT_LOG" ] && [ "$(wc -c < "$BOT_LOG")" -gt 20000000 ]; then
        tail -c 1000000 "$BOT_LOG" > "$BOT_LOG.tmp" && mv "$BOT_LOG.tmp" "$BOT_LOG"
        log "rotated $BOT_LOG"
    fi
    START=$(date +%s)
    .venv/bin/python scripts/run_volume_farmer_okx.py \
        --config config/config_volume_farmer_okx_v3.yaml \
        --label "$LABEL" \
        --duration-days 365 \
        "$@" >> "$BOT_LOG" 2>&1 &
    BOT_PID=$!
    wait "$BOT_PID"
    CODE=$?
    BOT_PID=""
    RAN=$(( $(date +%s) - START ))
    log "bot exited code=$CODE after ${RAN}s"
    # Clean operator stop / graceful shutdown: stay down.
    if [ "$CODE" -eq 0 ] || [ "$CODE" -eq 130 ] || [ "$CODE" -eq 143 ]; then
        log "clean exit — not restarting"
        rm -rf "$LOCK_DIR"
        exit 0
    fi
    if [ "$RAN" -lt 60 ]; then
        FAST_EXITS=$(( FAST_EXITS + 1 ))
        BACKOFF=$(( BACKOFF * 2 )); [ "$BACKOFF" -gt 300 ] && BACKOFF=300
        if [ "$FAST_EXITS" -eq 3 ]; then
            # The bot is dying before it can alert anyone — alert from here.
            tg_alert "🚨 keepalive[$LABEL]: bot crash-looping at startup (3 fast exits, last code=$CODE). It will keep retrying every ${BACKOFF}s but needs attention NOW — check $SUPERVISOR_LOG and $BOT_LOG."
            log "crash-loop alert sent to Telegram"
        fi
    else
        FAST_EXITS=0
        BACKOFF=15
    fi
    log "restarting in ${BACKOFF}s"
    sleep "$BACKOFF"
done
