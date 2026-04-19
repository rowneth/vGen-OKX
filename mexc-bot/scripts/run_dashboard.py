"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_dashboard.py

Or directly:
    streamlit run src/monitoring/dashboard_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
	root = Path(__file__).resolve().parents[1]
	app = root / "src" / "monitoring" / "dashboard_app.py"
	cmd = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		str(app),
		"--server.headless=false",
		"--browser.gatherUsageStats=false",
	]
	subprocess.run(cmd, check=False, cwd=str(root))


if __name__ == "__main__":
	main()
