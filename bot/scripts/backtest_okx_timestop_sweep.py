"""OKX time-stop / SL-tightness sweep — mirrors the live demo-v2 executor.

Models the live `live_volume_executor_okx.py` behaviour on historical 5m BTC:
  - micro_momentum entry (close>open => long else short), bar-range gate
  - ATR-14 relative bracket: TP = tp_mult x ATR_bps, SL = sl_mult x ATR_bps
    (floors tp>=5bps, sl>=8bps; skip entry when ATR < min_usd, matching live)
  - dynamic leverage from SL band (margin 3%, risk 2.5%, lev in [5,100])
  - exits: intrabar TP / SL (SL-first tie-break => sl_ambiguous), else
    TIME-STOP at max_hold bars (close at bar close), mirroring the live maker
    re-peg time-stop.
  - fee model matches live: open=maker, TP close=maker (limit_tp), time_stop
    close=maker (maker_exit re-peg), SL close=taker (native floor) + slippage.
    Rebate = 40% of total gross fees (OKX), accrued separately like the ledger.
  - per-trade net = gross - open_fee - close_fee  (rebate is a separate pool,
    exactly as the live okx-paper / demo-v2 state JSON tracks it).

Sweeps SL multiplier x max_hold so you can see whether "tighter SL + max_hold=3"
actually holds up versus the current live config (sl_mult=1.5, max_hold=2).

Usage:
    python scripts/backtest_okx_timestop_sweep.py
    python scripts/backtest_okx_timestop_sweep.py --sl-slippage-bps 8
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data/historical/BTC_USDT_5m.parquet"

# ── live demo-v2 params (config_volume_farmer_okx_v2.yaml) ───────────────────
MAKER, TAKER, REBATE = 0.0002, 0.0005, 0.40
CAPITAL = 500.0
MARGIN_FRAC, RISK_PCT = 0.03, 0.025
MAX_LEV, MIN_LEV = 100.0, 5.0
MIN_RANGE_BPS, MAX_RANGE_BPS = 3.0, 40.0
TP_MULT = 0.5
TP_BPS_MIN, SL_BPS_MIN = 5.0, 8.0
ATR_MIN_USD = 120.0          # skip entry when ATR < $120 (live min_usd)
LIMIT_TP = True              # TP fills maker
MAKER_EXIT = True            # time_stop fills maker


@dataclass
class Cfg:
    sl_mult:  float
    max_hold: int
    tp_mult:  float = TP_MULT
    tag:      str = ""


def _atr14(df: pd.DataFrame) -> np.ndarray:
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high", "low", "close"))
    pcl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pcl), np.abs(lo - pcl)))
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13] = tr[:14].mean()
        for k in range(14, len(tr)):
            atr[k] = atr[k - 1] * (13 / 14) + tr[k] * (1 / 14)
    return atr


def simulate(df: pd.DataFrame, atr: np.ndarray, cfg: Cfg, sl_slip_bps: float) -> dict:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    n = len(df)
    rng_bps = np.abs(c - o) / o * 1e4

    # FIXED-capital sizing: notional is sized off the *starting* balance, not a
    # compounding one, so per-trade $ economics are comparable across configs and
    # a negative edge doesn't spiral to ruin and corrupt the averages. Equity is
    # tracked only as a linear cumulative curve for drawdown.
    nets: list[float] = []          # per-trade net (ledger style, no rebate)
    bps_list: list[float] = []      # per-trade net in bps of notional (size-free)
    reason_net: dict[str, list] = {}
    gross_sum = maker_fee = taker_fee = volume = 0.0

    i = 14
    while i < n - 1:
        a = atr[i]
        if np.isnan(a) or a < ATR_MIN_USD:
            i += 1; continue
        if not (MIN_RANGE_BPS <= rng_bps[i] <= MAX_RANGE_BPS):
            i += 1; continue

        side = 1 if c[i] > o[i] else -1          # micro_momentum
        # (alternate_direction = false in v2 — no flip)

        atr_bps = a / o[i] * 1e4
        tp_bps = max(cfg.tp_mult * atr_bps, TP_BPS_MIN)
        sl_bps = max(cfg.sl_mult * atr_bps, SL_BPS_MIN)

        entry = c[i]
        margin = CAPITAL * MARGIN_FRAC           # fixed base, no compounding
        sl_frac = sl_bps / 1e4
        lev = max(MIN_LEV, min((CAPITAL * RISK_PCT) / (margin * sl_frac), MAX_LEV)) if sl_frac else MAX_LEV
        notional = margin * lev
        open_fee = notional * MAKER
        maker_fee += open_fee; volume += notional

        tp = entry * (1 + tp_bps / 1e4) if side == 1 else entry * (1 - tp_bps / 1e4)
        sl = entry * (1 - sl_bps / 1e4) if side == 1 else entry * (1 + sl_bps / 1e4)

        reason = None; px = 0.0; exit_bar = i
        for j in range(i + 1, n):
            bh = j - i
            tp_hit = (h[j] >= tp) if side == 1 else (lo[j] <= tp)
            sl_hit = (lo[j] <= sl) if side == 1 else (h[j] >= sl)
            if tp_hit and sl_hit:
                reason, px, exit_bar = "sl_ambiguous", sl, j; break
            if tp_hit:
                reason, px, exit_bar = "tp", tp, j; break
            if sl_hit:
                reason, px, exit_bar = "sl", sl, j; break
            if bh >= cfg.max_hold:
                reason, px, exit_bar = "time_stop", c[j], j; break
        if reason is None:
            i += 1; continue

        gross = ((px - entry) if side == 1 else (entry - px)) / entry * notional
        if reason in ("sl", "sl_ambiguous"):
            gross -= sl_slip_bps * 1e-4 * notional        # taker SL slips against us
            close_fee = notional * TAKER; taker_fee += close_fee
        elif reason == "tp":
            close_fee = notional * (MAKER if LIMIT_TP else TAKER); maker_fee += close_fee
        else:  # time_stop
            close_fee = notional * (MAKER if MAKER_EXIT else TAKER); maker_fee += close_fee

        net = gross - open_fee - close_fee
        gross_sum += gross; volume += notional
        nets.append(net)
        bps_list.append(net / notional * 1e4)
        reason_net.setdefault(reason, []).append(net)
        i = exit_bar + 1

    nets_a = np.array(nets)
    wins = nets_a[nets_a > 0]; losses = nets_a[nets_a <= 0]
    total_fee = maker_fee + taker_fee
    rebate = REBATE * total_fee
    true_net = nets_a.sum() + rebate
    # drawdown on the linear cumulative-net curve (incl. rebate paid per trade)
    curve = CAPITAL + np.cumsum(nets_a)
    peak = np.maximum.accumulate(curve) if len(curve) else np.array([CAPITAL])
    maxdd = float((curve - peak).min()) if len(curve) else 0.0
    return dict(
        cfg=cfg, trades=len(nets), wr=len(wins) / len(nets) * 100 if nets else 0,
        avg_win=wins.mean() if len(wins) else 0, avg_loss=losses.mean() if len(losses) else 0,
        rr=(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() else float("inf"),
        pf=(wins.sum() / -losses.sum()) if len(losses) and losses.sum() else float("inf"),
        exp=nets_a.mean() if len(nets) else 0,
        exp_bps=float(np.mean(bps_list)) if bps_list else 0.0,
        ledger_net=nets_a.sum(), rebate=rebate, true_net=true_net,
        total_fee=total_fee, taker_fee=taker_fee, volume=volume,
        maxdd=maxdd, maxdd_pct=maxdd / CAPITAL * 100,
        reasons={k: (len(v), float(np.sum(v))) for k, v in reason_net.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl-slippage-bps", type=float, default=5.0,
                    help="adverse slippage on taker SL fills (live SLs gapped; default 5)")
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(DATA); df.columns = [x.lower() for x in df.columns]
    if args.rows:
        df = df.iloc[-args.rows:].reset_index(drop=True)
    t0, t1 = pd.to_datetime(df["open_time"]).min(), pd.to_datetime(df["open_time"]).max()
    span = (t1 - t0).total_seconds() / 86400
    atr = _atr14(df)
    print(f"Data: {len(df):,} bars  {t0.date()} → {t1.date()}  ({span:.0f}d)   SL-slip={args.sl_slippage_bps}bps")
    print(f"Model: ATR-rel TP=0.5xATR, SL=mult xATR | maker TP+time_stop, taker SL | OKX 40% rebate | cap $500\n")

    W = 122
    hdr = (f"  {'TP×ATR':>7} {'SL×ATR':>7} {'maxHold':>8} {'trades':>7} {'WR%':>6} {'avgWin':>7} "
           f"{'avgLoss':>8} {'RR':>5} {'PF':>5} {'edge bps/t':>10} {'net/t$':>7} {'TRUEnet$':>9} "
           f"{'maxDD$':>8} {'tp/ts/sl':>15}")

    def show(rows):
        print("-" * W)
        for r in rows:
            cfg = r["cfg"]
            mh = "—(999)" if cfg.max_hold >= 999 else str(cfg.max_hold)
            rc = r["reasons"]
            mix = f"{rc.get('tp',(0,0))[0]}/{rc.get('time_stop',(0,0))[0]}/{rc.get('sl',(0,0))[0]+rc.get('sl_ambiguous',(0,0))[0]}"
            print(f"  {cfg.tp_mult:>7.2f} {cfg.sl_mult:>7.2f} {mh:>8} {r['trades']:>7,} {r['wr']:>5.1f}% "
                  f"{r['avg_win']:>+7.2f} {r['avg_loss']:>+8.2f} {r['rr']:>5.2f} {r['pf']:>5.2f} "
                  f"{r['exp_bps']:>+10.2f} {r['exp']:>+7.3f} {r['true_net']:>+9.0f} "
                  f"{r['maxdd']:>+8.0f} {mix:>15}  {cfg.tag}")

    # ── Block A: the user's question — SL tightness × max_hold (TP fixed 0.5) ──
    gridA = []
    for sl_mult in (1.5, 1.25, 1.0, 0.8):
        for mh in (2, 3, 4):
            tag = "← CURRENT live" if (sl_mult, mh) == (1.5, 2) else ("← PROPOSED" if (sl_mult, mh) == (1.0, 3) else "")
            gridA.append(Cfg(sl_mult, mh, tag=tag))
    gridA.append(Cfg(1.5, 999, tag="← no time-stop (old paper)"))
    rowsA = [simulate(df, atr, cfg, args.sl_slippage_bps) for cfg in gridA]

    # ── Block B: the real lever — raise TP toward SL (RR rebalance), max_hold=3 ──
    gridB = [Cfg(sl, 3, tp_mult=tp) for tp, sl in
             [(0.5, 1.5), (0.75, 1.5), (1.0, 1.5), (1.0, 1.0), (1.25, 1.25), (1.5, 1.5), (1.0, 0.75)]]
    rowsB = [simulate(df, atr, cfg, args.sl_slippage_bps) for cfg in gridB]

    print("=" * W)
    print("  BLOCK A — SL tightness × max_hold  (TP fixed at 0.5×ATR, as live)")
    print(hdr)
    show(rowsA)
    print("=" * W)
    print("  BLOCK B — TP:SL rebalance  (max_hold=3) — does raising TP toward SL fix the edge?")
    print(hdr)
    show(rowsB)
    print("=" * W)
    print("  edge bps/t = per-trade net as bps of notional (size-free truth) | TRUEnet$ = sum net + 40% rebate, linear (no compounding)")
    print("  tp/ts/sl = exit-reason counts (sl includes sl_ambiguous)")

    cur = next(r for r in rowsA if (r["cfg"].sl_mult, r["cfg"].max_hold) == (1.5, 2))
    best = max((r for r in rowsA + rowsB if r["cfg"].max_hold < 999), key=lambda r: r["true_net"])
    bc = best["cfg"]
    print(f"\n  CURRENT (TP0.5 / SL1.5 / hold2):  edge {cur['exp_bps']:+.2f}bps/t  RR {cur['rr']:.2f}  WR {cur['wr']:.1f}%  TRUEnet ${cur['true_net']:+,.0f}")
    print(f"  BEST in either block:  TP{bc.tp_mult}/SL{bc.sl_mult}/hold{bc.max_hold}  →  "
          f"edge {best['exp_bps']:+.2f}bps/t  RR {best['rr']:.2f}  WR {best['wr']:.1f}%  TRUEnet ${best['true_net']:+,.0f}")


if __name__ == "__main__":
    main()
