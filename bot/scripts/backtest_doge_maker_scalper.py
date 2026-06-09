"""Honest maker-scalper backtest for DOGE-USDT-SWAP volume farming.

Goal: estimate the COST to farm a volume target as a delta-bounded (max 1 clip)
passive market-maker, and whether a small bankroll survives it.

Why single-lot ping-pong: it's the safest volume-farming mode (inventory never
exceeds one clip), which is what a $40 bankroll demands. It also makes the fill
model honest — we never assume we magically captured the spread on both sides of
a bar that actually trended through us.

Fill model (the part naive backtests fake):
  * Flat: rest a passive BUY at the touch (bid) and SELL at the touch (ask).
    Within a 1m bar the price visits its extremes in an assumed order derived
    from close vs open (up bar: open->low->high->close; down bar reversed). The
    first of our quotes the path reaches *may* fill, each gated by `fill_prob`
    (a queue-position haircut: touching your price != getting filled).
  * In a position: rest a passive exit at entry +/- `profit_ticks` (maker). It
    fills only if a LATER bar reaches it (again gated by fill_prob). If the hold
    expires or price runs `stop_ticks` against us, we TAKER-flatten at the close
    (real fee + the real adverse move). Conditioning the loss on "price did not
    come back" is exactly the adverse selection real makers eat.

Everything is parameterised; the key unknown (`fill_prob`) is swept so the output
is a range, not one flattering number.
"""
from __future__ import annotations
import csv, random, statistics, sys
from dataclasses import dataclass

# --- DOGE-USDT-SWAP instrument spec (live) ---
TICK = 1e-5
CTVAL = 1000.0          # 1 contract = 1000 DOGE
MAKER = 0.0002
TAKER = 0.0005
REBATE = 0.40           # 40% rebate on maker fees (their account)


def load(path):
    out = []
    with open(path) as f:
        for x in csv.DictReader(f):
            out.append((int(x["ts"]), float(x["open"]), float(x["high"]),
                        float(x["low"]), float(x["close"])))
    out.sort(key=lambda z: z[0])
    return out


@dataclass
class Result:
    fill_prob: float
    profit_ticks: int
    cycles: int
    captured: int            # exited maker at profit target
    stopped: int             # taker-flattened (timeout/stop)
    capture_rate: float
    net_pnl: float           # realized incl fees, incl rebate
    fees_paid: float
    rebate_earned: float
    volume: float
    bps_per_rt: float        # net cost (neg) or gain (pos) per round trip, bps of notional
    max_dd: float            # worst equity drawdown ($) over the window


def run(bars, *, fill_prob, profit_ticks=1, stop_ticks=12, max_hold=3,
        clip_contracts=0.10, capital=40.0, seed=0) -> Result:
    rng = random.Random(seed)
    pos = 0           # -1 short, 0 flat, +1 long
    entry = 0.0
    age = 0
    realized = 0.0
    fees = 0.0
    rebate = 0.0
    volume = 0.0
    cycles = captured = stopped = 0
    peak_eq = capital
    max_dd = 0.0
    clip_notional_at = lambda px: clip_contracts * CTVAL * px

    def maker_fill(px):
        nonlocal fees, rebate, volume
        n = clip_notional_at(px)
        f = n * MAKER
        fees += f; rebate += f * REBATE; volume += n

    def taker_fill(px):
        nonlocal fees, volume
        n = clip_notional_at(px)
        fees += n * TAKER; volume += n

    for (ts, o, h, l, c) in bars:
        if pos == 0:
            # quote both sides at the touch (1 tick apart straddling open)
            bid = round((o - TICK / 2) / TICK) * TICK
            ask = bid + TICK
            up = c >= o
            # path-ordered first reachable quote
            first = ("buy", bid) if up else ("sell", ask)
            second = ("sell", ask) if up else ("buy", bid)
            took = None
            for side, px in (first, second):
                touched = (l <= px) if side == "buy" else (h >= px)
                if touched and rng.random() < fill_prob:
                    took = (side, px); break
            if took:
                side, px = took
                maker_fill(px)
                pos = 1 if side == "buy" else -1
                entry = px; age = 0; cycles += 1
        else:
            age += 1
            exit_px = entry + profit_ticks * TICK if pos > 0 else entry - profit_ticks * TICK
            hit = (h >= exit_px) if pos > 0 else (l <= exit_px)
            adverse = ((entry - l) if pos > 0 else (h - entry)) / entry * 1e4
            if hit and rng.random() < fill_prob:
                # maker exit at profit target -> spread captured
                maker_fill(exit_px)
                realized += (exit_px - entry) * pos * clip_contracts * CTVAL
                captured += 1; pos = 0
            elif age >= max_hold or adverse >= stop_ticks * (TICK / entry * 1e4):
                # taker flatten at close (real fee + real adverse move)
                taker_fill(c)
                realized += (c - entry) * pos * clip_contracts * CTVAL
                stopped += 1; pos = 0
        # mark-to-market equity for drawdown
        unreal = 0.0 if pos == 0 else (c - entry) * pos * clip_contracts * CTVAL
        eq = capital + realized - (fees - rebate) + unreal
        peak_eq = max(peak_eq, eq)
        max_dd = max(max_dd, peak_eq - eq)

    net = realized - (fees - rebate)
    rts = max(captured + stopped, 1)
    avg_notional = volume / max(cycles + captured + stopped, 1)
    bps_per_rt = net / max(volume, 1e-9) * 1e4 * 2  # per round-trip (2 legs of vol)
    return Result(fill_prob, profit_ticks, cycles, captured, stopped,
                  captured / rts, net, fees, rebate, volume, bps_per_rt, max_dd)


def main():
    bars = load("data/doge_1m.csv")
    days = len(bars) / 1440
    CAP = 40.0
    TARGET = 400_000.0
    CLIP = 0.10  # contracts (~$8.4 notional/clip at $0.084) — 1-clip max inventory

    print(f"DOGE maker-scalper backtest | {len(bars)} bars (~{days:.1f}d) | "
          f"capital ${CAP:.0f} | target ${TARGET:,.0f} | clip={CLIP} ct (~${CLIP*CTVAL*0.084:.1f})")
    print(f"Single-lot ping-pong, max inventory = 1 clip. Fee: maker {MAKER*1e4:.0f}bps "
          f"(rebate {REBATE:.0%}) / taker {TAKER*1e4:.0f}bps\n")

    hdr = (f"{'fill_p':>7}{'pTk':>4}{'cycles':>8}{'cap%':>7}{'net$':>9}"
           f"{'bps/RT':>8}{'vol$(win)':>11}{'maxDD$':>8}  | extrapolated to $400k:")
    print(hdr); print("-" * len(hdr.split('|')[0]) + "|" + "-" * 46)
    for pt in (1, 2):
        for fp in (0.3, 0.5, 0.7, 0.9):
            # average a few seeds to smooth the RNG haircut
            rs = [run(bars, fill_prob=fp, profit_ticks=pt, clip_contracts=CLIP, capital=CAP, seed=s)
                  for s in range(5)]
            r = rs[0]
            bps = statistics.mean(x.bps_per_rt for x in rs)
            net = statistics.mean(x.net_pnl for x in rs)
            vol = statistics.mean(x.volume for x in rs)
            cap_rate = statistics.mean(x.capture_rate for x in rs)
            dd = statistics.mean(x.max_dd for x in rs)
            cyc = statistics.mean(x.cycles for x in rs)
            # extrapolate to target
            cost_400k = bps / 1e4 / 2 * TARGET    # $ cost (neg bps -> negative = cost)
            end_bal = CAP + cost_400k
            scale = TARGET / max(vol, 1e-9)
            days_to = days * scale
            survive = "YES" if end_bal > 0 else "BLOWS UP"
            print(f"{fp:>7.1f}{pt:>4}{cyc:>8.0f}{cap_rate:>7.0%}{net:>9.2f}"
                  f"{bps:>8.2f}{vol:>11.0f}{dd:>8.2f}  | "
                  f"cost ${cost_400k:>7.2f}  end ${end_bal:>6.2f}  {days_to:>4.0f}d  {survive}")
        print()
    print("net$/bps_per_RT are over the ~4.2d window; 'cost to $400k' scales the per-RT")
    print("economics to the full target. bps_per_RT<0 = net cost; >0 = you get PAID.")


if __name__ == "__main__":
    main()
