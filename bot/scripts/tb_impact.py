"""Compare trend_break ON vs OFF on the 1m live config.

Shows how the early-exit affects win rate, average loss, end equity
and — critically — how much the live paper-vs-real divergence costs.
"""
from __future__ import annotations
import copy, pathlib, sys
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from backtest_volume_farmer import run_one  # noqa: E402

DATA = PROJECT_ROOT / "data/historical/BTC_USDT_1m.parquet"

BASE_TP = 8.0
BASE_SL = 50.0

VARIANTS = [
    ("TB ON  (live config)  ",  True,  3, 20.0),
    ("TB OFF (no early cut) ", False,  3, 20.0),
    ("TB aggressive 5bps   ",  True,  2,  5.0),
    ("TB conservative 40bps",  True,  3, 40.0),
]

df = pd.read_parquet(DATA)
rows = []
for label, tb_en, tb_min, tb_adv in VARIANTS:
    r = run_one(df, tp_bps=BASE_TP, sl_bps=BASE_SL, capital_usd=30.0,
                tb_enabled=tb_en, tb_min_bars=tb_min, tb_adverse_bps=tb_adv)
    wins      = int(r.get("wins", 0))
    losses    = int(r.get("losses", 0))
    trades    = wins + losses
    wr        = float(r.get("win_rate_pct", 0))
    be_wr     = float(r.get("break_even_wr_pct", 0))
    end       = float(r.get("end_equity_with_rebate", 30.0))
    reb       = float(r.get("total_rebate", 0.0))
    tb_exits  = int(r.get("trend_breaks", 0))
    avg_w     = float(r.get("avg_win_bps", 0))
    avg_l     = float(r.get("avg_loss_bps", 0))
    rows.append({
        "Config"       : label,
        "Trades"       : trades,
        "WR %"         : round(wr, 1),
        "BE-WR %"      : round(be_wr, 1),
        "WR gap"       : round(wr - be_wr, 1),
        "TB exits"     : tb_exits,
        "Avg win bps"  : round(avg_w, 1),
        "Avg loss bps" : round(avg_l, 1),
        "End+Reb $"    : round(end, 2),
    })

out = pd.DataFrame(rows)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
print(out.to_string(index=False))
