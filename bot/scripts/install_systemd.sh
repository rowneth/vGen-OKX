#!/usr/bin/env bash
# Install / refresh the vGen OKX farmer systemd unit on the droplet.
# Run as root on the droplet AFTER pulling the latest code.
#
# Usage (on droplet):
#   sudo bash /root/vGen-OKX/bot/scripts/install_systemd.sh
set -euo pipefail

REPO="/root/vGen-OKX/bot"
UNIT_SRC="$REPO/scripts/vgen-farmer-okx.service"
UNIT_DST="/etc/systemd/system/vgen-farmer-okx.service"
LOG_DIR="$REPO/logs"

if [[ ! -f "$UNIT_SRC" ]]; then
	echo "ERROR: $UNIT_SRC not found. Pull the repo first." >&2
	exit 1
fi

mkdir -p "$LOG_DIR"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable vgen-farmer-okx.service
systemctl restart vgen-farmer-okx.service
sleep 3
systemctl --no-pager status vgen-farmer-okx.service | head -n 30
echo
echo "Tail with: journalctl -u vgen-farmer-okx.service -f   (or)   tail -f $LOG_DIR/farmer_okx.log"
