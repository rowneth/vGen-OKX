#!/usr/bin/env bash
# One-shot droplet deployment for the vGen OKX v3 campaign bot.
# Run AS ROOT on the droplet. Idempotent — safe to re-run.
#
#   bash scripts/deploy_droplet.sh demo     # rehearsal (OKX simulated)
#   bash scripts/deploy_droplet.sh live     # REAL MONEY (requires the ack below)
#
# It: pulls okx-reconcile, ensures the venv, validates .env, stops any OLD
# unit, installs the v3 systemd unit for the chosen mode, starts it, and tails
# the first lines so you can confirm the startup card fired.
set -euo pipefail

MODE="${1:-demo}"
REPO="/root/vGen-OKX/bot"
BRANCH="okx-reconcile"
UNIT="vgen-farmer-okx.service"
UNIT_DST="/etc/systemd/system/${UNIT}"

[[ "$MODE" == "demo" || "$MODE" == "live" ]] || { echo "mode must be demo|live"; exit 1; }
cd "$REPO" || { echo "ERROR: $REPO not found — clone the repo first:"; \
                echo "  git clone https://github.com/rowneth/vGen-OKX.git /root/vGen-OKX"; exit 1; }

echo "== 1/7 pull $BRANCH =="
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git --no-pager log --oneline -1

echo "== 2/7 venv + deps =="
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt 2>/dev/null || \
  .venv/bin/pip install -q aiohttp pandas numpy pyyaml matplotlib
.venv/bin/python -c "import ast; ast.parse(open('scripts/run_volume_farmer_okx.py').read())" \
  && echo "syntax OK"

echo "== 3/7 validate .env =="
[[ -f .env ]] || { echo "ERROR: $REPO/.env missing — scp it from your laptop first:"; \
                   echo "  scp '/Users/rowneth/vGen v2.0 OKX/bot/.env' root@167.172.88.229:$REPO/.env"; exit 1; }
need=(OKX_API_KEY OKX_API_SECRET OKX_API_PASSPHRASE TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID)
miss=0
for k in "${need[@]}"; do grep -q "^${k}=" .env || { echo "  MISSING $k"; miss=1; }; done
[[ "$miss" == 0 ]] && echo "core keys present"
if [[ "$MODE" == "live" ]]; then
  if ! grep -q "^LIVE_OKX_ACK=I_UNDERSTAND" .env; then
    echo "ERROR: live mode needs the explicit ack. Add it and re-run:"
    echo "  echo 'LIVE_OKX_ACK=I_UNDERSTAND' >> $REPO/.env"
    exit 1
  fi
  echo "live ack present"
fi

echo "== 4/7 stop ALL old farmer units (any name) =="
for u in $(systemctl list-units --type=service --all --no-legend 2>/dev/null \
           | awk '{print $1}' | grep -iE 'vgen|farmer' || true); do
  echo "  stopping $u"; systemctl stop "$u" 2>/dev/null || true
  systemctl disable "$u" 2>/dev/null || true
done
# also kill any stray foreground bot
pkill -INT -f "run_volume_farmer_okx.py" 2>/dev/null || true
sleep 3

echo "== 5/7 install v3 unit for MODE=$MODE =="
ARGS="--config $REPO/config/config_volume_farmer_okx_v3.yaml --label okx-v3-$MODE --$MODE --duration-days 365"
cat > "$UNIT_DST" <<UNIT_EOF
[Unit]
Description=vGen OKX v3 farmer ($MODE)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=root
WorkingDirectory=$REPO
EnvironmentFile=-$REPO/.env
Environment="PYTHONUNBUFFERED=1"
ExecStart=$REPO/.venv/bin/python $REPO/scripts/run_volume_farmer_okx.py $ARGS
Restart=on-failure
RestartSec=15
StandardOutput=append:$REPO/logs/farmer_okx.log
StandardError=append:$REPO/logs/farmer_okx.log
KillSignal=SIGINT
TimeoutStopSec=30
[Install]
WantedBy=multi-user.target
UNIT_EOF
mkdir -p "$REPO/logs"
systemctl daemon-reload
systemctl enable "$UNIT"

echo "== 6/7 start =="
systemctl restart "$UNIT"
sleep 6
systemctl --no-pager status "$UNIT" | head -n 8

echo "== 7/7 startup log (watch for the fee-tier line + startup card) =="
tail -n 25 "$REPO/logs/farmer_okx.log" 2>/dev/null | grep -E \
  "MODE|clock drift|fee tier|ACTUAL fee|set leverage|state loaded|polling|HALT|ERROR" || \
  tail -n 15 "$REPO/logs/farmer_okx.log"
echo
echo "DONE ($MODE). Live tail:  journalctl -u $UNIT -f   (or)   tail -f $REPO/logs/farmer_okx.log"
echo "Check Telegram for the startup card — verify it says fee tier verified, leverage 15x, your real wallet."
