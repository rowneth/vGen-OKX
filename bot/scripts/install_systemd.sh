#!/usr/bin/env bash
# Install / refresh the vgen-farmer systemd unit on the DigitalOcean droplet.
# Run this script as root on the droplet AFTER syncing the latest code.
#
# Usage (on droplet):
#   sudo bash /root/vGen/mexc-bot/scripts/install_systemd.sh
set -euo pipefail

UNIT_SRC="/root/vGen/mexc-bot/scripts/vgen-farmer.service"
UNIT_DST="/etc/systemd/system/vgen-farmer.service"
LOG_DIR="/root/vGen/mexc-bot/logs"

if [[ ! -f "$UNIT_SRC" ]]; then
	echo "ERROR: $UNIT_SRC not found.  Sync the repo first." >&2
	exit 1
fi

mkdir -p "$LOG_DIR"
install -m 0644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable vgen-farmer.service
systemctl restart vgen-farmer.service
sleep 3
systemctl --no-pager status vgen-farmer.service | head -n 30
echo
echo "Tail with: journalctl -u vgen-farmer.service -f   (or)   tail -f $LOG_DIR/farmer_systemd.log"
