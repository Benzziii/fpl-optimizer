"""
fpl_optimizer.py — STAGE 2: Optimizer MILP multi-periode (HiGHS via scipy.optimize.milp).

Memodelkan aturan FPL 2026/27 secara eksplisit:
  * skuad 15 (2 GK / 5 DEF / 5 MID / 3 FWD), maks 3 per klub, XI sah, kapten
  * transfer antar-GW dengan BANK & HARGA JUAL (sell price) yang benar
  * FREE TRANSFER bergulir (maks 5), hit -4 per transfer berlebih
  * WILDCARD  : transfer tanpa batas & tanpa hit (permanen)
  * FREE HIT  : skuad 1-minggu bebas, skuad permanen tidak berubah
  * BENCH BOOST: poin 4 pemain cadangan ikut dihitung
  * TRIPLE CAPTAIN: kapten 3x
  * 2 set chip (GW1-19 & GW20-38), 1 chip per GW, set lama hangus di GW19
  * biaya peluang chip (agar chip tidak dibakar untuk keuntungan kecil)

Solver memilih SEKALIGUS: siapa dibeli/dijual tiap GW, kapan memakai tiap chip,
XI, kapten, dan kapan lebih baik membayar hit — memaksimalkan total xP horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

CHIPS = ["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"]
FIRST_HALF_LAST_GW = 19
SQUAD_POS = {1: 2, 2: 5, 3: 5, 4: 3}
INF = np.inf


class MIP:
    def __init__(self):
        self.lb, self.ub, self.integ, self.c = [], [], [], []
        self.R, self.C, self.V, self.lo, self.hi = [], [], [], [], []

    def var(self, lb=0.0, ub=1.0, integer=False, obj=0.0) -> int:
        self.lb.append(lb); self.ub.append(ub)
        self.integ.append(1 if integer else 0); self.c.append(obj)
        return len(self.lb) - 1

    def add_obj(self, j, coef):
        self.c[j] += coef

    def con(self, terms, lo=-INF, hi=INF):
        row = len(self.lo)
        for j, a in terms:
            if a != 0:
                self.R.append(row); self.C.append(j); self.V.append(a)
        self.lo.append(lo); self.hi.append(hi)

    def solve(self, time_limit=60, gap=0.005):
        A = sp.csr_matrix((self.V, (self.R, self.C)), shape=(len(self.lo), len(self.lb)))
        res = milp(c=-np.array(self.c), constraints=LinearConstraint(A, self.lo, self.hi),
                   integrality=np.array(self.integ), bounds=Bounds(self.lb, self.ub),
                   options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
        return res


def half_of(gw: int) -> int:
    return 1 if gw <= FIRST_HALF_LAST_GW else 2


def chip_opportunity_cost(base: float, gw: int) -> float:
    """Biaya peluang chip, mengecil mendekati batas set (use-it-or-lose-it)."""
    deadline = FIRST_HALF_LAST_GW if gw <= FIRST_HALF_LAST_GW else 38
    return base * float(np.clip((deadline - gw) / 10.0, 0.0, 1.0))


def select_candidates(el: pd.DataFrame, xp: pd.DataFrame, current_ids, per_pos=None) -> list:
    """Pangkas pool (solver lebih cepat) tanpa membuang opsi penting."""
    per_pos = per_pos or {1: 8, 2: 30, 3: 38, 4: 20}
    keep = set(current_ids)
    tot = xp.sum(axis=1)
    pos = el["element_type"]
    for p, k in per_pos.items():
        ids = pos[pos == p].index
        keep |= set(tot.reindex(ids).nlargest(k).index)
        for gw in xp.columns:                       # spesialis per-GW (Free Hit / TC / DGW)
            keep |= set(xp[gw].reindex(ids).nlargest(max(4, k // 3)).index)
        val = (tot / el["price"]).reindex(ids)      # value picks
        keep |= set(val[tot.reindex(ids) > 0].nlargest(max(4, k // 4)).index)
        ok = el.loc[ids]
        ok = ok[(ok["status"] == "a") & (tot.reindex(ids) > 0)]
        keep |= set(ok.nsmallest(5, "now_cost").index)   # pengisi bench murah
    return sorted(keep)


def optimize(el: pd.DataFrame, xp: pd.DataFrame, current_ids: list, bank: float, ft: int,
             chips_avail: dict, sell_prices: dict | None = None, *, gamma: float = 0.93,
             bench_w: float = 0.10, max_transfers: int = 5, chip_costs: dict | None = None,
             allow_transfers: bool = True, allowed_chips=None, allow_hits: bool = True,
             time_limit: int = 60, gap: float = 0.005, soft_transfer_cost: float = 0.05,
             chip_plan: dict | None = None, per_pos: dict | None = None) -> dict:
    """
    el     : DataFrame indeks id (kolom element_type, team, now_cost [x0.1m], price)
    xp     : DataFrame indeks id x kolom GW (xP per GW)
    bank   : £m ; chips_avail : {(chip, half): bool}
    """
    gws = list(xp.columns)
    H = len(gws)
    allowed_chips = set(CHIPS if allowed_chips is None else allowed_chips)
    chip_costs = chip_costs or {"Wildcard": 8, "Free Hit": 10, "Bench Boost": 8, "Triple Captain": 5}
    sell_prices = sell_prices or {}

    ids = select_candidates(el, xp, current_ids, per_pos)
    N = len(ids)
    pos = el.loc[ids, "element_type"].values
    team = el.loc[ids, "team"].values
    buy = el.loc[ids, "now_cost"].astype(int).values
    sell = np.array([sell_prices.get(i, b) for i, b in zip(ids, buy)])
    X = xp.reindex(ids).fillna(0.0).values            # N x H
    cur_set = set(current_ids)
    cur = np.array([1 if i in cur_set else 0 for i in ids])
    bank0 = int(round(bank * 10))
    BIG = 6000

    m = MIP()
    sq = [[m.var(integer=True) for _ in range(H)] for _ in range(N)]
    tin = [[m.var(ub=1.0 if allow_transfers else 0.0) for _ in range(H)] for _ in range(N)]
    tout = [[m.var(ub=1.0 if allow_transfers else 0.0) for _ in range(H)] for _ in range(N)]
    xi = [[m.var() for _ in range(H)] for _ in range(N)]
    cap = [[m.var() for _ in range(H)] for _ in range(N)]

    # ---- chip variabel (hanya jika tersedia di paruh tsb & diizinkan) ----
    chip = {k: {} for k in CHIPS}
    for k in CHIPS:
        if k not in allowed_chips:
            continue
        for t, gw in enumerate(gws):
            if gw == 1 and k == "Free Hit":
                continue
            if not chips_avail.get((k, half_of(gw)), True):
                continue
            if chip_plan is not None:
                if chip_plan.get((k, half_of(gw))) != t:
                    continue
                chip[k][t] = m.var(lb=1, ub=1, integer=True,
                                   obj=-chip_opportunity_cost(chip_costs.get(k, 0), gw))
            else:
                chip[k][t] = m.var(integer=True,
                                   obj=-chip_opportunity_cost(chip_costs.get(k, 0), gw))
    for t in range(H):
        m.con([(chip[k][t], 1) for k in CHIPS if t in chip[k]], hi=1)
    for k in CHIPS:
        for h in (1, 2):
            tt = [t for t in chip[k] if half_of(gws[t]) == h]
            if tt:
                m.con([(chip[k][t], 1) for t in tt], hi=1)
        for t in chip[k]:                                 # tidak boleh GW19 lalu GW20
            if gws[t] == FIRST_HALF_LAST_GW and (t + 1) in chip[k]:
                m.con([(chip[k][t], 1), (chip[k][t + 1], 1)], hi=1)

    # ---- FH: skuad 1-minggu ----
    fhs = {}
    ps = [[None] * H for _ in range(N)]
    for i in range(N):
        for t in range(H):
            ps[i][t] = sq[i][t]
    if chip["Free Hit"]:
        for t in chip["Free Hit"]:
            fhv = chip["Free Hit"][t]
            for i in range(N):
                fhs[i, t] = m.var(integer=True)
                ps[i][t] = m.var()                        # kontinu; dipaksa = sq atau fhs
            m.con([(fhs[i, t], 1) for i in range(N)] + [(fhv, -15)], lo=0, hi=0)
            for p, n in SQUAD_POS.items():
                m.con([(fhs[i, t], 1) for i in range(N) if pos[i] == p] + [(fhv, -n)], lo=0, hi=0)
            for tm_ in np.unique(team):
                m.con([(fhs[i, t], 1) for i in range(N) if team[i] == tm_] + [(fhv, -3)], hi=0)
            for i in range(N):
                m.con([(fhs[i, t], 1), (fhv, -1)], hi=0)
                s, f, p_ = sq[i][t], fhs[i, t], ps[i][t]
                m.con([(p_, 1), (s, -1), (fhv, -1)], hi=0)          # ps <= sq + fh
                m.con([(p_, 1), (f, -1), (fhv, 1)], hi=1)           # ps <= fhs + 1 - fh
                m.con([(p_, 1), (s, -1), (fhv, 1)], lo=0)           # ps >= sq - fh
                m.con([(p_, 1), (f, -1), (fhv, -1)], lo=-1)         # ps >= fhs - (1-fh)

    # ---- bank, FT, hit ----
    bank_v = [m.var(lb=0, ub=BIG) for _ in range(H)]
    ft_v = [m.var(lb=ft, ub=ft)] + [m.var(lb=1, ub=5) for _ in range(H)]
    hit = [m.var(lb=0, ub=(INF if allow_hits else 0.0), obj=-4.0 * gamma ** t) for t in range(H)]
    ncnt = [m.var(lb=0, ub=15) for _ in range(H)]
    over = [m.var(integer=True) for _ in range(H)]

    for t in range(H):
        for i in range(N):
            prev = [(sq[i][t - 1], 1)] if t > 0 else []
            rhs = 0 if t > 0 else -cur[i]
            m.con([(tin[i][t], 1), (tout[i][t], -1), (sq[i][t], -1)] + prev, lo=rhs, hi=rhs)
            m.con([(tin[i][t], 1), (tout[i][t], 1)], hi=1)
            m.add_obj(tin[i][t], -soft_transfer_cost * gamma ** t)
        m.con([(sq[i][t], 1) for i in range(N)], lo=15, hi=15)
        for p, n in SQUAD_POS.items():
            m.con([(sq[i][t], 1) for i in range(N) if pos[i] == p], lo=n, hi=n)
        for tm_ in np.unique(team):
            m.con([(sq[i][t], 1) for i in range(N) if team[i] == tm_], hi=3)

        # bank
        terms = [(bank_v[t], 1)] + [(tout[i][t], -sell[i]) for i in range(N)] \
                + [(tin[i][t], buy[i]) for i in range(N)]
        if t > 0:
            terms.append((bank_v[t - 1], -1)); m.con(terms, lo=0, hi=0)
        else:
            m.con(terms, lo=bank0, hi=bank0)

        ntr = [(tin[i][t], 1) for i in range(N)]
        wc = chip["Wildcard"].get(t)
        fh = chip["Free Hit"].get(t)
        cap_n = max_transfers
        if wc is not None:
            m.con(ntr + [(wc, -(15 - cap_n))], hi=cap_n)
            m.con(ntr + [(ncnt[t], -1), (wc, -15)], hi=0)          # ncnt >= n - 15 wc
        else:
            m.con(ntr, hi=cap_n)
            m.con(ntr + [(ncnt[t], -1)], hi=0)
        if fh is not None:
            m.con(ntr + [(fh, 15)], hi=15)                          # FH => tanpa transfer permanen
            # budget skuad FH = nilai jual skuad + bank (sebelum GW ini)
            val_prev = ([(sq[i][t - 1], sell[i]) for i in range(N)] + [(bank_v[t - 1], 1)]) if t > 0 else None
            rhs_const = float(np.dot(sell, cur) + bank0) if t == 0 else 0.0
            terms = [(fhs[i, t], buy[i]) for i in range(N)] + [(fh, BIG * 10)]
            if t > 0:
                terms += [(c_, -a_) for c_, a_ in val_prev]
            m.con(terms, hi=rhs_const + BIG * 10)
        m.con([(hit[t], 1), (ncnt[t], -1), (ft_v[t], 1)], lo=0)     # hit >= ncnt - ft
        m.con([(ft_v[t + 1], 1), (ft_v[t], -1), (ncnt[t], 1), (over[t], -15)], hi=1)
        m.con([(ft_v[t + 1], 1), (over[t], 4)], hi=5)

        # ---- XI, kapten, chip BB/TC ----
        for i in range(N):
            m.con([(xi[i][t], 1), (ps[i][t], -1)], hi=0)
            m.con([(cap[i][t], 1), (xi[i][t], -1)], hi=0)
        m.con([(xi[i][t], 1) for i in range(N)], lo=11, hi=11)
        m.con([(xi[i][t], 1) for i in range(N) if pos[i] == 1], lo=1, hi=1)
        m.con([(xi[i][t], 1) for i in range(N) if pos[i] == 2], lo=3)
        m.con([(xi[i][t], 1) for i in range(N) if pos[i] == 3], lo=2)
        m.con([(xi[i][t], 1) for i in range(N) if pos[i] == 4], lo=1)
        m.con([(cap[i][t], 1) for i in range(N)], lo=1, hi=1)

        g_t = gamma ** t
        for i in range(N):
            x = X[i, t]
            m.add_obj(xi[i][t], g_t * (x - bench_w * x))
            m.add_obj(cap[i][t], g_t * x)
            m.add_obj(ps[i][t], g_t * bench_w * x)
        if t in chip["Bench Boost"]:
            bb = chip["Bench Boost"][t]
            for i in range(N):
                bv = m.var(obj=g_t * (1 - bench_w) * X[i, t])
                m.con([(bv, 1), (ps[i][t], -1), (xi[i][t], 1)], hi=0)
                m.con([(bv, 1), (bb, -1)], hi=0)
        if t in chip["Triple Captain"]:
            tc = chip["Triple Captain"][t]
            for i in range(N):
                w = m.var(obj=g_t * X[i, t])
                m.con([(w, 1), (cap[i][t], -1)], hi=0)
                m.con([(w, 1), (tc, -1)], hi=0)

    res = m.solve(time_limit=time_limit, gap=gap)
    if res.x is None:
        raise RuntimeError(f"Solver tidak menemukan solusi: {res.message}")
    x = res.x
    val = lambda j: float(x[j])

    # ------------------------- ekstraksi & verifikasi ------------------------
    out_gws = []
    ft_now = ft
    prev_sq = set(current_ids)
    bank_now = bank0
    for t, gw in enumerate(gws):
        squad = [ids[i] for i in range(N) if val(sq[i][t]) > 0.5]
        play = [ids[i] for i in range(N) if val(ps[i][t]) > 0.5]
        xi_ids = [ids[i] for i in range(N) if val(xi[i][t]) > 0.5]
        cap_id = ids[max(range(N), key=lambda i: val(cap[i][t]))]
        used = next((k for k in CHIPS if t in chip[k] and val(chip[k][t]) > 0.5), None)
        t_in = sorted(set(squad) - prev_sq)
        t_out = sorted(prev_sq - set(squad))
        n = len(t_in)
        n_cnt = 0 if used == "Wildcard" else n
        hit_pts = 4 * max(0, n_cnt - ft_now)
        xp_t = xp[gw]
        xi_sorted = sorted(xi_ids, key=lambda i: -xp_t[i])
        vice_id = next(i for i in xi_sorted if i != cap_id)
        bench = [i for i in play if i not in xi_ids]
        gk_b = [i for i in bench if el.at[i, "element_type"] == 1]
        out_b = sorted([i for i in bench if el.at[i, "element_type"] != 1], key=lambda i: -xp_t[i])
        bench_ord = gk_b + out_b                            # GK cadangan di slot 12, lalu outfield
        mult = 3 if used == "Triple Captain" else 2
        pts_xi = float(sum(xp_t[i] for i in xi_ids))
        pts_cap = float(xp_t[cap_id] * (mult - 1))
        pts_bench = float(sum(xp_t[i] for i in bench)) if used == "Bench Boost" else 0.0
        out_gws.append(dict(
            gw=gw, squad=squad, play_squad=play, xi=xi_ids, captain=cap_id, vice=vice_id,
            bench=bench_ord, transfers_in=t_in, transfers_out=t_out, n_transfers=n,
            chip=used, hit=hit_pts, ft_available=ft_now, xp_xi=pts_xi, xp_captain_bonus=pts_cap,
            xp_bench_boost=pts_bench, xp_gross=pts_xi + pts_cap + pts_bench,
            xp_net=pts_xi + pts_cap + pts_bench - hit_pts))
        ft_now = min(5, max(ft_now - n_cnt, 0) + 1)
        if used != "Free Hit":
            prev_sq = set(squad)
    total = sum(g["xp_net"] for g in out_gws)
    wtotal = sum(g["xp_net"] * gamma ** t for t, g in enumerate(out_gws))
    return dict(gws=out_gws, total_xp=total, weighted_xp=wtotal, solver_status=int(res.status),
                solver_message=str(res.message), mip_gap=getattr(res, "mip_gap", None),
                n_candidates=N, objective=-float(res.fun))


# ==========================================================================
# PENCARIAN CHIP (screening MILP terkunci + coordinate ascent + solve akhir)
# ==========================================================================
def optimize_with_chips(el, xp, current_ids, bank, ft, chips_avail, sell_prices=None, *,
                        allowed_chips=None, total_time: int = 150, final_time: int = 60,
                        progress=None, **kw) -> dict:
    """
    Menentukan KAPAN memakai tiap chip (WC/FH/BB/TC) di horizon + semua transfer.

    Memasukkan keputusan minggu-chip langsung ke satu MIP membuat bound LP lemah
    (lambat). Sebagai gantinya: tiap penempatan chip dievaluasi lewat MILP dengan
    chip TERKUNCI (cepat), lalu coordinate-ascent mencari kombinasi terbaik, dan
    terakhir solve presisi pada kombinasi terpilih.
    """
    import time
    t0 = time.time()
    gws = list(xp.columns)
    allowed = [k for k in CHIPS if k in (allowed_chips if allowed_chips is not None else CHIPS)]
    slots = []                       # (chip, half) yang relevan di horizon
    for k in allowed:
        for h in (1, 2):
            wk = [t for t, gw in enumerate(gws) if half_of(gw) == h and chips_avail.get((k, h), True)
                  and not (k == "Free Hit" and gw == 1)]
            if wk:
                slots.append(((k, h), wk))
    base_kw = dict(kw)
    base_kw.pop("time_limit", None); base_kw.pop("gap", None)
    screen_pp = {1: 6, 2: 20, 3: 26, 4: 14}
    cache = {}

    def ev(plan: dict, final=False):
        key = tuple(sorted(plan.items()))
        if not final and key in cache:
            return cache[key]
        r = optimize(el, xp, current_ids, bank, ft, chips_avail, sell_prices,
                     allowed_chips=sorted({k for k, _ in plan}), chip_plan=dict(plan),
                     time_limit=(final_time if final else 15), gap=(0.003 if final else 0.01),
                     per_pos=(None if final else screen_pp), **base_kw)
        if not final:
            cache[key] = r
        return r

    base = ev({})
    base_obj = base["objective"]
    gain = {}                                         # (chip, gw) -> gain vs tanpa chip
    plan, cur_obj, n_eval = {}, base_obj, 1
    total = sum(len(w) for _, w in slots)
    # --- pass 1: gain tiap penempatan tunggal (juga dipakai tabel UI)
    for (slot, wk) in slots:
        for t in wk:
            if time.time() - t0 > total_time:
                break
            r = ev({slot: t}); n_eval += 1
            gain[(slot[0], gws[t])] = r["objective"] - base_obj
            if progress:
                progress(min(0.9, n_eval / (total + 6)), f"Evaluasi {slot[0]} di GW{gws[t]}")
    # --- coordinate ascent + pair-move (menangkap sinergi, mis. BB di GW sebelum TC)
    slot_weeks = dict(slots)
    top = {sl: sorted(wk, key=lambda t: -gain.get((sl[0], gws[t]), -1e9))[:3] for sl, wk in slots}
    timeup = lambda: time.time() - t0 > total_time

    def try_plan(trial):
        nonlocal n_eval
        if timeup():
            return -1e18
        n_eval += 1
        return ev(trial)["objective"]

    for _round in range(3):
        improved = False
        # (a) pindah / tambah / lepas satu chip
        for (slot, wk) in slots:
            occupied = {t for s_, t in plan.items() if s_ != slot}
            best_plan, best_obj = None, cur_obj
            for t in wk:
                if t in occupied or t == plan.get(slot):
                    continue
                trial = dict(plan); trial[slot] = t
                o = try_plan(trial)
                if o > best_obj + 0.05:
                    best_plan, best_obj = trial, o
            if slot in plan:
                trial = {k_: v_ for k_, v_ in plan.items() if k_ != slot}
                o = try_plan(trial)
                if o > best_obj + 0.05:
                    best_plan, best_obj = trial, o
            if best_plan is not None:
                plan, cur_obj, improved = best_plan, best_obj, True
        # (b) geser dua chip sekaligus di sekitar minggu-minggu terbaik
        new_slots = [sl for sl, _ in slots if sl not in plan]
        movers = list(plan.keys())
        best_plan, best_obj = None, cur_obj
        for s_new in new_slots:
            for a_ in movers:
                for t_new in top[s_new]:
                    for t_a in set(top[a_]) | {plan[a_]}:
                        others = {t for s_, t in plan.items() if s_ not in (a_, s_new)}
                        if t_new == t_a or t_new in others or t_a in others:
                            continue
                        trial = dict(plan); trial[s_new] = t_new; trial[a_] = t_a
                        o = try_plan(trial)
                        if o > best_obj + 0.05:
                            best_plan, best_obj = trial, o
        if best_plan is not None:
            plan, cur_obj, improved = best_plan, best_obj, True
        if not improved or timeup():
            break
    if progress:
        progress(0.93, "Solve akhir (presisi)")
    final = ev(plan, final=True)
    final["chip_gain_table"] = gain      # gain BERSIH (setelah biaya peluang chip)
    final["chip_cost"] = {(k, g): chip_opportunity_cost((kw.get("chip_costs") or
                          {"Wildcard": 8, "Free Hit": 10, "Bench Boost": 8, "Triple Captain": 5}).get(k, 0), g)
                          for (k, g) in gain}
    final["chip_plan"] = {f"{k[0]}": gws[t] for k, t in plan.items()}
    final["baseline_no_chip_obj"] = base_obj
    final["n_solves"] = n_eval + 1
    final["search_seconds"] = time.time() - t0
    return final
