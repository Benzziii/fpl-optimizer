"""
fpl_model.py — STAGE 1: Expected Points (xP) predictor.

Arsitektur (hybrid, tahan data sedikit di awal musim):

  A. Model STRUKTURAL (dari aturan poin FPL, bukan black-box)
       xP = poin_main + gol + assist + clean sheet - kebobolan + saves
            + bonus + DefCon + kartu
     * xMins / P(main) / P(>=60')  : rata-rata menit berbobot-waktu (2 half-life)
                                      x faktor ketersediaan (injury/suspend/news)
     * rate xG/xA/bonus/saves per-90: shrinkage Bayesian ke prior
                                      (musim lalu + rata-rata posisi)
     * konteks fixture             : kekuatan serang/bertahan tim dari xG tim
                                      (disesuaikan kualitas lawan) -> lambda gol
                                      => P(clean sheet) = exp(-lambda_kebobolan)
                                      => E[floor(kebobolan/2)] dari distribusi Poisson
     * DGW/BGW                     : xP dijumlahkan per pertandingan (BGW = 0)

  B. GBM (HistGradientBoosting) sebagai KOREKSI atas xP struktural
       (monotonic terhadap xP struktural, regularisasi kuat)

  C. Walk-forward BACKTEST: menimbang bobot GBM vs struktural (alpha) dan
       melaporkan RMSE/MAE/korelasi vs baseline "form" agar akurasi terukur jujur.

Semua fitur dihitung "as-of" (hanya info sebelum GW tsb) -> tidak ada kebocoran.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import factorial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

POS_NAME = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = np.array([0, 6, 6, 5, 4], dtype=float)     # index = element_type
CS_PTS = np.array([0, 4, 4, 1, 0], dtype=float)
DC_THR = np.array([999, 999, 10, 12, 12], dtype=float)  # DefCon: DEF 10 CBIT, MID/FWD 12 CBIRT

HL_SLOW, HL_FAST = 8.0, 3.0
D_SLOW, D_FAST = 0.5 ** (1 / HL_SLOW), 0.5 ** (1 / HL_FAST)

DEFAULT_PRIOR = {  # dipakai hanya jika data musim berjalan belum cukup
    "gx90": {1: 0.0, 2: 0.04, 3: 0.12, 4: 0.30},
    "ax90": {1: 0.01, 2: 0.06, 3: 0.11, 4: 0.10},
    "bonus90": {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.25},
    "sv90": {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0},
    "yc90": {1: 0.03, 2: 0.14, 3: 0.12, 4: 0.09},
    "dc": {1: 0.0, 2: 0.30, 3: 0.12, 4: 0.04},
}

GBM_FEATURES = ["struct", "exp_min", "p60", "gx90", "ax90", "bonus90", "pos",
                "lam_for", "lam_against", "home", "price", "form"]

STAT_COLS = ["n", "mins", "apps", "p60", "xg", "goals", "xa", "assists",
             "bonus", "saves", "yc", "dch", "pts"]


# ==========================================================================
# 1. DATA HISTORI
# ==========================================================================
_NUM = ["minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "saves", "bonus", "bps", "expected_goals", "expected_assists", "expected_goals_conceded",
        "defensive_contribution", "yellow_cards", "red_cards", "own_goals", "starts", "value"]


def prepare_elements(bootstrap) -> pd.DataFrame:
    el = pd.DataFrame(bootstrap["elements"])
    for c in ["now_cost", "cost_change_start", "selected_by_percent", "ep_next",
              "chance_of_playing_next_round", "chance_of_playing_this_round",
              "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
              "transfers_in_event", "transfers_out_event", "form", "minutes"]:
        el[c] = pd.to_numeric(el.get(c), errors="coerce")
    teams = {t["id"]: t for t in bootstrap["teams"]}
    el["team_name"] = el["team"].map(lambda t: teams[t]["name"])
    el["team_short"] = el["team"].map(lambda t: teams[t]["short_name"])
    el["pos_name"] = el["element_type"].map(POS_NAME)
    el["price"] = el["now_cost"] / 10.0
    return el


def build_history(summaries: dict, fixtures: list, elements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, s in summaries.items():
        if s and s.get("history"):
            for h in s["history"]:
                d = dict(h)
                d["element"] = int(pid)
                rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in _NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df else 0.0
    df["round"] = df["round"].astype(int)
    df["was_home"] = df["was_home"].astype(int)
    df["opponent_team"] = df["opponent_team"].astype(int)
    df["pos"] = df["element"].map(elements.set_index("id")["element_type"]).astype(int)
    fx_team = {f["id"]: (f["team_h"], f["team_a"]) for f in fixtures}
    cur_team = df["element"].map(elements.set_index("id")["team"])
    tm = [(fx_team[f][0] if h else fx_team[f][1]) if f in fx_team else c
          for f, h, c in zip(df["fixture"], df["was_home"], cur_team)]
    df["team"] = np.array(tm, dtype=int)
    # DefCon: beberapa feed memberi poin (0/2) bukan hitungan -> deteksi otomatis
    dc = df["defensive_contribution"]
    df["dc_hit"] = np.where(dc.max() <= 2, dc >= 2, dc >= DC_THR[df["pos"].values]).astype(float)
    return df.sort_values(["element", "round"]).reset_index(drop=True)


def build_priors(summaries: dict, elements: pd.DataFrame) -> pd.DataFrame:
    """Prior per pemain dari musim lalu (history_past[-1])."""
    out = []
    for pid in elements["id"]:
        hp = ((summaries.get(pid) or {}).get("history_past")) or []
        rec = {"element": pid, "prev_min": 0.0, "prev_gx90": np.nan, "prev_ax90": np.nan,
               "prev_bonus90": np.nan, "prev_sv90": np.nan}
        if hp:
            last = hp[-1]
            f = lambda k: float(pd.to_numeric(last.get(k), errors="coerce") or 0.0)
            m = f("minutes")
            rec["prev_min"] = m
            if m >= 450:
                xg = f("expected_goals") or f("goals_scored")
                xa = f("expected_assists") or f("assists")
                rec["prev_gx90"] = (0.8 * xg + 0.2 * f("goals_scored")) / (m / 90)
                rec["prev_ax90"] = (0.8 * xa + 0.2 * f("assists")) / (m / 90)
                rec["prev_bonus90"] = f("bonus") / (m / 90)
                rec["prev_sv90"] = f("saves") / (m / 90)
        out.append(rec)
    return pd.DataFrame(out).set_index("element")


# ==========================================================================
# 2. KEKUATAN TIM (serang/bertahan dari xG, disesuaikan kualitas lawan)
# ==========================================================================
def build_team_matches(hist: pd.DataFrame, fixtures: list) -> pd.DataFrame:
    xg = hist.groupby(["fixture", "team"])["expected_goals"].sum().rename("xgf").reset_index()
    rows = []
    for f in fixtures:
        if not f.get("finished") or not f.get("event") or f.get("team_h_score") is None:
            continue
        rows.append(dict(fixture=f["id"], team=f["team_h"], opp=f["team_a"], home=1,
                         round=f["event"], gf=f["team_h_score"], ga=f["team_a_score"]))
        rows.append(dict(fixture=f["id"], team=f["team_a"], opp=f["team_h"], home=0,
                         round=f["event"], gf=f["team_a_score"], ga=f["team_h_score"]))
    tm = pd.DataFrame(rows)
    if tm.empty:
        return tm
    tm = tm.merge(xg, on=["fixture", "team"], how="left")
    opp_xg = xg.rename(columns={"team": "opp", "xgf": "xga"})
    tm = tm.merge(opp_xg, on=["fixture", "opp"], how="left")
    tm["mf"] = np.where(tm["xgf"] > 0, 0.65 * tm["xgf"] + 0.35 * tm["gf"], tm["gf"])
    tm["ma"] = np.where(tm["xga"] > 0, 0.65 * tm["xga"] + 0.35 * tm["ga"], tm["ga"])
    # kalibrasi level liga: rata-rata lambda = rata-rata gol sebenarnya
    # (penting agar P(clean sheet) = exp(-lambda) tidak bias)
    k = float(tm["gf"].mean() / max(tm["mf"].mean(), 1e-6))
    tm["mf"] *= k
    tm["ma"] *= k
    return tm


class TeamModel:
    """att/def multiplikatif gaya Dixon-Coles ringan, as-of per round."""

    def __init__(self, tm: pd.DataFrame, n_teams: int = 20, k: float = 3.0, iters: int = 4):
        self.tm, self.k, self.iters = tm, k, iters
        self.T = n_teams + 1               # index 1..20
        self._cache = {}

    def asof(self, r: int) -> dict:
        if r in self._cache:
            return self._cache[r]
        T = self.T
        att, dfn = np.ones(T), np.ones(T)
        raw_f, raw_a = np.full(T, 1.35), np.full(T, 1.35)
        g, hf = 1.35, 1.12
        sub = self.tm[self.tm["round"] < r] if len(self.tm) else self.tm
        if len(sub) >= 10:
            g = float(sub["mf"].mean())
            hm = sub.loc[sub.home == 1, "mf"].mean()
            am = sub.loc[sub.home == 0, "mf"].mean()
            n_h = (sub.home == 1).sum()
            hf_obs = float(np.sqrt(np.clip(hm / max(am, 1e-6), 0.8, 1.6)))
            hf = (n_h * hf_obs + 40 * 1.12) / (n_h + 40)
            home = sub["home"].values == 1
            mf_adj = sub["mf"].values / np.where(home, hf, 1 / hf)
            ma_adj = sub["ma"].values * np.where(home, hf, 1 / hf)
            w = D_SLOW ** (r - 1 - sub["round"].values)
            t_idx, o_idx = sub["team"].values, sub["opp"].values
            for _ in range(self.iters):
                num_a = np.bincount(t_idx, weights=w * mf_adj, minlength=T)
                den_a = np.bincount(t_idx, weights=w * g * dfn[o_idx], minlength=T)
                num_d = np.bincount(t_idx, weights=w * ma_adj, minlength=T)
                den_d = np.bincount(t_idx, weights=w * g * att[o_idx], minlength=T)
                att = (num_a + self.k * g) / (den_a + self.k * g)
                dfn = (num_d + self.k * g) / (den_d + self.k * g)
                att[0] = dfn[0] = 1.0
                att, dfn = att / att[1:].mean(), dfn / dfn[1:].mean()
                att[0] = dfn[0] = 1.0
            sw = np.bincount(t_idx, weights=w, minlength=T)
            raw_f = (np.bincount(t_idx, weights=w * sub["mf"].values, minlength=T) + self.k * g) / (sw + self.k)
            raw_a = (np.bincount(t_idx, weights=w * sub["ma"].values, minlength=T) + self.k * g) / (sw + self.k)
        res = dict(g=g, hf=hf, att=att, dfn=dfn, raw_f=raw_f, raw_a=raw_a)
        self._cache[r] = res
        return res

    def lams(self, team, opp, home, rounds):
        """lambda gol (untuk, kebobolan) + rata-rata raw tim, vektor."""
        team, opp, home, rounds = map(np.asarray, (team, opp, home, rounds))
        n = len(team)
        lf, la, rf, ra = (np.zeros(n) for _ in range(4))
        for r in np.unique(rounds):
            m = rounds == r
            s = self.asof(int(r))
            h = np.where(home[m] == 1, s["hf"], 1 / s["hf"])
            lf[m] = s["g"] * s["att"][team[m]] * s["dfn"][opp[m]] * h
            la[m] = s["g"] * s["att"][opp[m]] * s["dfn"][team[m]] / h
            rf[m], ra[m] = s["raw_f"][team[m]], s["raw_a"][team[m]]
        return lf, la, rf, ra


# ==========================================================================
# 3. STATE PEMAIN (jumlah berbobot-waktu, as-of)
# ==========================================================================
class PlayerStates:
    def __init__(self, hist: pd.DataFrame, element_ids):
        self.idx = {int(p): i for i, p in enumerate(element_ids)}
        self.R = int(hist["round"].max())
        N, R = len(self.idx), self.R
        h = hist
        cols = pd.DataFrame({
            "element": h["element"], "round": h["round"],
            "n": 1.0, "mins": h["minutes"], "apps": (h["minutes"] > 0).astype(float),
            "p60": (h["minutes"] >= 60).astype(float),
            "xg": h["expected_goals"], "goals": h["goals_scored"],
            "xa": h["expected_assists"], "assists": h["assists"],
            "bonus": h["bonus"], "saves": h["saves"],
            "yc": h["yellow_cards"] + 3 * h["red_cards"],
            "dch": h["dc_hit"] * (h["minutes"] >= 60), "pts": h["total_points"],
        })
        agg = cols.groupby(["element", "round"])[STAT_COLS].sum().reset_index()
        ei = agg["element"].map(self.idx).values
        ri = agg["round"].values - 1
        self.slow, self.fast = {}, {}
        for s in STAT_COLS:
            X = np.zeros((N, R))
            np.add.at(X, (ei, ri), agg[s].values)
            S = np.zeros((N, R + 1))
            for r in range(R):
                S[:, r + 1] = D_SLOW * S[:, r] + X[:, r]
            self.slow[s] = S
            if s in ("n", "mins", "apps", "p60"):
                F = np.zeros((N, R + 1))
                for r in range(R):
                    F[:, r + 1] = D_FAST * F[:, r] + X[:, r]
                self.fast[s] = F

    def get(self, stat, element, rounds, fast=False):
        ei = np.array([self.idx[int(e)] for e in element])
        col = np.clip(np.asarray(rounds) - 1, 0, self.R)
        return (self.fast if fast else self.slow)[stat][ei, col]


# ==========================================================================
# 4. FITUR PEMAIN (as-of) + MODEL STRUKTURAL
# ==========================================================================
def position_priors(states: PlayerStates, elements: pd.DataFrame) -> dict:
    """Rata-rata liga per posisi dari data musim berjalan (fallback: default)."""
    pos_of = elements.set_index("id")["element_type"]
    pr = {k: dict(v) for k, v in DEFAULT_PRIOR.items()}
    ids = list(states.idx.keys())
    pos = np.array([pos_of.get(i, 3) for i in ids])
    last = states.R
    tot = {s: states.slow[s][:, last] for s in STAT_COLS}
    for p in (1, 2, 3, 4):
        m = pos == p
        min90 = tot["mins"][m].sum() / 90
        if min90 > 60:
            pr["gx90"][p] = (0.8 * tot["xg"][m].sum() + 0.2 * tot["goals"][m].sum()) / min90
            pr["ax90"][p] = (0.8 * tot["xa"][m].sum() + 0.2 * tot["assists"][m].sum()) / min90
            pr["bonus90"][p] = tot["bonus"][m].sum() / min90
            pr["sv90"][p] = tot["saves"][m].sum() / min90
            pr["yc90"][p] = tot["yc"][m].sum() / min90
            if tot["p60"][m].sum() > 20:
                pr["dc"][p] = tot["dch"][m].sum() / tot["p60"][m].sum()
    return pr


def player_features(ctx, element, rounds, fit_basis=None) -> pd.DataFrame:
    """Fitur per (pemain, round) — semua hanya dari data SEBELUM round tsb."""
    S, el = ctx.states, ctx.el
    element = np.asarray(element)
    rounds = np.asarray(rounds)
    g = lambda s, f=False: S.get(s, element, rounds, fast=f)
    n, mins, apps, p60 = g("n"), g("mins"), g("apps"), g("p60")
    nf, minsf, appsf, p60f = g("n", 1), g("mins", 1), g("apps", 1), g("p60", 1)
    pos = el.loc[element, "element_type"].values
    price = el.loc[element, "price"].values
    pri = ctx.priors.reindex(element)
    prev_min = pri["prev_min"].fillna(0).values

    pm = np.where(prev_min > 0, np.clip(prev_min / 38.0, 0, 85), 25.0)
    Km = 2.0
    mpm_s = (mins + Km * pm) / (n + Km)
    mpm_f = (minsf + 1.0 * pm) / (nf + 1.0)
    exp_min = 0.55 * mpm_s + 0.45 * mpm_f
    pa_prior, p60_prior = np.clip(pm / 60, 0, 1), np.clip(pm / 90, 0, 1) * 0.95
    p_any = 0.55 * (apps + Km * pa_prior) / (n + Km) + 0.45 * (appsf + pa_prior) / (nf + 1)
    p60r = 0.55 * (p60 + Km * p60_prior) / (n + Km) + 0.45 * (p60f + p60_prior) / (nf + 1)
    if fit_basis is not None:      # pemain berstatus cedera/ragu: basis "jika bugar"
        fb = np.asarray(fit_basis, dtype=bool)
        c_min = np.clip((mins + Km * pm) / (apps + Km), 0, 90)
        c_p60 = (p60 + Km * p60_prior) / (apps + Km)
        exp_min = np.where(fb, c_min * 0.95, exp_min)
        p_any = np.where(fb, 0.95, p_any)
        p60r = np.where(fb, np.clip(c_p60, 0, 1) * 0.95, p60r)
    p_any = np.clip(np.maximum(p_any, p60r), 0, 1)

    min90 = mins / 90.0
    pp = ctx.pos_prior
    w_prev = np.clip(prev_min / 1800, 0, 1) * 0.75

    def rate(num, key, prev_col, K):
        base = np.array([pp[key][p] for p in pos])
        prev = pri[prev_col].values
        prior = np.where(np.isnan(prev), base, w_prev * np.nan_to_num(prev) + (1 - w_prev) * base)
        return (num + K * prior) / (min90 + K)

    gx90 = rate(0.8 * g("xg") + 0.2 * g("goals"), "gx90", "prev_gx90", 6.0)
    ax90 = rate(0.8 * g("xa") + 0.2 * g("assists"), "ax90", "prev_ax90", 6.0)
    bonus90 = rate(g("bonus"), "bonus90", "prev_bonus90", 6.0)
    sv90 = rate(g("saves"), "sv90", "prev_sv90", 6.0) * (pos == 1)
    yc90 = (g("yc") + 8.0 * np.array([pp["yc90"][p] for p in pos])) / (min90 + 8.0)
    dc_prior = np.array([pp["dc"][p] for p in pos])
    dc_p = (g("dch") + 4.0 * dc_prior) / (p60 + 4.0)
    form = (g("pts") + 2.0 * 2.0) / (n + 2.0)
    pen1 = (el.loc[element, "penalties_order"].fillna(9).values == 1).astype(float)
    gx90 = gx90 + pen1 * 0.045 * 6.0 / (min90 + 6.0)

    return pd.DataFrame(dict(element=element, round=rounds, pos=pos, price=price,
                             exp_min=exp_min, p_any=p_any, p60=p60r, gx90=gx90, ax90=ax90,
                             bonus90=bonus90, sv90=sv90, yc90=yc90,
                             dc_p=dc_p * (pos != 1), form=form))


def _e_floor_half(lam: np.ndarray) -> np.ndarray:
    ks = np.arange(0, 16)
    fact = np.array([factorial(int(k)) for k in ks], dtype=float)
    pmf = np.exp(-lam)[:, None] * lam[:, None] ** ks[None, :] / fact[None, :]
    return (np.floor(ks / 2)[None, :] * pmf).sum(1)


def structural_components(f: pd.DataFrame) -> pd.DataFrame:
    """xP struktural per-pertandingan, dipecah per komponen (poin FPL)."""
    pos = f["pos"].values.astype(int)
    m = f["exp_min"].values / 90.0
    p_any, p60 = f["p_any"].values, f["p60"].values
    lf, la = f["lam_for"].values, f["lam_against"].values
    atk = np.clip(lf / np.maximum(f["team_for"].values, 0.4), 0.3, 3.0)
    dfs = np.clip(la / np.maximum(f["team_against"].values, 0.4), 0.3, 3.0)
    c = pd.DataFrame(index=f.index)
    c["main"] = p_any + p60
    c["gol"] = GOAL_PTS[pos] * f["gx90"].values * m * atk
    c["assist"] = 3.0 * f["ax90"].values * m * atk
    c["cs"] = CS_PTS[pos] * p60 * np.exp(-la)
    lam_on = la * np.clip(f["exp_min"].values / np.maximum(p_any, 0.05) / 90.0, 0, 1)
    c["kebobolan"] = -np.where(pos <= 2, p_any * _e_floor_half(lam_on), 0.0)
    sv = f["sv90"].values * m * dfs
    c["saves"] = np.where(pos == 1, np.maximum(sv - p_any, 0) / 3.0, 0.0)
    c["bonus"] = np.maximum(f["bonus90"].values * m * (1 + 0.12 * (lf - la)), 0)
    c["defcon"] = 2.0 * f["dc_p"].values * p60
    c["kartu"] = -f["yc90"].values * m
    c["struct"] = c.sum(axis=1)
    return c


# ==========================================================================
# 5. KONTEKS MODEL + TRAINING + BACKTEST
# ==========================================================================
@dataclass
class ModelContext:
    el: pd.DataFrame
    hist: pd.DataFrame
    team_model: TeamModel
    states: PlayerStates
    priors: pd.DataFrame
    pos_prior: dict
    next_gw: int
    gbm: HistGradientBoostingRegressor | None = None
    alpha: float = 0.35
    backtest: dict = field(default_factory=dict)
    fixtures: list = field(default_factory=list)
    events: list = field(default_factory=list)


def _row_frame(ctx: ModelContext, element, rounds, team, opp, home, fit_basis=None) -> pd.DataFrame:
    f = player_features(ctx, element, rounds, fit_basis)
    lf, la, rf, ra = ctx.team_model.lams(team, opp, home, rounds)
    f["home"] = np.asarray(home, dtype=float)
    f["lam_for"], f["lam_against"], f["team_for"], f["team_against"] = lf, la, rf, ra
    comp = structural_components(f)
    f = pd.concat([f, comp], axis=1)
    return f


def _new_gbm():
    mono = [1 if c == "struct" else 0 for c in GBM_FEATURES]
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
        l2_regularization=5.0, monotonic_cst=mono, random_state=0)


def training_frame(ctx: ModelContext) -> pd.DataFrame:
    h = ctx.hist
    tf = _row_frame(ctx, h["element"].values, h["round"].values, h["team"].values,
                    h["opponent_team"].values, h["was_home"].values)
    tf["y"] = h["total_points"].values
    return tf


def _metrics(y, p):
    err = y - p
    return dict(RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(np.abs(err))),
                Korelasi=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else float("nan"))


def run_backtest(tf: pd.DataFrame, n_folds: int = 5) -> tuple[dict, float]:
    rounds = sorted(tf["round"].unique())
    test_rounds = [r for r in rounds if r >= 4][-n_folds:]
    preds = []
    for k in test_rounds:
        tr, te = tf[tf["round"] < k], tf[tf["round"] == k]
        if len(tr) < 600 or len(te) == 0:
            continue
        g = _new_gbm().fit(tr[GBM_FEATURES], tr["y"])
        te = te.assign(gbm=g.predict(te[GBM_FEATURES]))
        preds.append(te)
    if not preds:
        return {"ok": False, "note": "Data belum cukup untuk backtest walk-forward (butuh >= 4 GW)."}, 0.35
    P = pd.concat(preds)
    P = P[P["exp_min"] >= 30]            # fokus pemain yang relevan untuk keputusan
    y = P["y"].values
    best_a, best_rmse = 0.35, 9e9
    for a in np.linspace(0, 1, 11):
        r = _metrics(y, a * P["gbm"].values + (1 - a) * P["struct"].values)["RMSE"]
        if r < best_rmse:
            best_a, best_rmse = float(a), r
    blend = best_a * P["gbm"].values + (1 - best_a) * P["struct"].values
    table = pd.DataFrame({
        "Baseline: form (poin/laga berbobot)": _metrics(y, P["form"].values),
        "Struktural saja": _metrics(y, P["struct"].values),
        "GBM saja": _metrics(y, P["gbm"].values),
        f"Blend final (alpha GBM={best_a:.1f})": _metrics(y, blend),
    }).T
    cal = pd.DataFrame({"pred": blend, "aktual": y})
    cal["desil"] = pd.qcut(cal["pred"].rank(method="first"), 10, labels=False) + 1
    cal = cal.groupby("desil").agg(prediksi=("pred", "mean"), aktual=("aktual", "mean"), n=("pred", "size"))
    return {"ok": True, "table": table, "calibration": cal, "n_rows": int(len(P)),
            "rounds": [int(r) for r in test_rounds]}, best_a


def build_context(bootstrap, fixtures, summaries) -> ModelContext:
    el = prepare_elements(bootstrap).set_index("id", drop=False)
    hist = build_history(summaries, fixtures, el.reset_index(drop=True))
    if hist.empty:
        raise ValueError("Histori pemain kosong — musim belum dimulai.")
    next_gw = next((e["id"] for e in bootstrap["events"] if e.get("is_next")), int(hist["round"].max()) + 1)
    tm = build_team_matches(hist, fixtures)
    states = PlayerStates(hist, el["id"].values)
    ctx = ModelContext(el=el, hist=hist, team_model=TeamModel(tm, n_teams=len(bootstrap["teams"])),
                       states=states, priors=build_priors(summaries, el.reset_index(drop=True)),
                       pos_prior=position_priors(states, el), next_gw=next_gw,
                       fixtures=fixtures, events=bootstrap["events"])
    tf = training_frame(ctx)
    ctx.backtest, ctx.alpha = run_backtest(tf)
    ctx.gbm = _new_gbm().fit(tf[GBM_FEATURES], tf["y"])
    return ctx


# ==========================================================================
# 6. KETERSEDIAAN (injury / suspend / news) & PREDIKSI MULTI-GW
# ==========================================================================
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_BACK = re.compile(r"(?:expected back|back on|until|returns?(?: on)?)\s+(\d{1,2})\s+([A-Za-z]{3})", re.I)


def availability_matrix(ctx: ModelContext, gw_list: list[int], overrides: dict | None = None) -> pd.DataFrame:
    """P(tersedia) per pemain per GW. overrides: {id: 'out'|'fit'}."""
    overrides = overrides or {}
    ev_dates = [(e["id"], pd.Timestamp(e["deadline_time"]).tz_localize(None)) for e in ctx.events
                if e.get("deadline_time")]
    yr0 = ev_dates[0][1].year if ev_dates else 2026
    rows = {}
    for pid, r in ctx.el.iterrows():
        status = r["status"]
        cop = r["chance_of_playing_next_round"]
        a0 = (cop / 100.0) if pd.notna(cop) else {"a": 1.0, "d": 0.5}.get(status, 0.0)
        ret_gw = None
        news = str(r.get("news") or "")
        mm = _BACK.search(news)
        if mm and mm.group(2).lower() in _MONTHS and status in ("i", "d", "s"):
            mo = _MONTHS[mm.group(2).lower()]
            dt = pd.Timestamp(year=yr0 if mo >= 7 else yr0 + 1, month=mo, day=min(int(mm.group(1)), 28))
            cand = [g for g, d in ev_dates if d >= dt - pd.Timedelta(days=2)]
            ret_gw = cand[0] if cand else None
        a = []
        for j, gw in enumerate(gw_list):
            if status == "u":
                v = 0.0
            elif status == "a" and pd.isna(cop):
                v = 1.0
            elif ret_gw is not None:
                v = 0.0 if gw < ret_gw else (0.85 if gw == ret_gw else 1.0)
                if gw == gw_list[0] and a0 > 0:
                    v = a0
            elif status == "d":
                v = 1 - (1 - a0) * 0.4 ** j
            elif status == "i":
                if a0 > 0:       # ada peluang 25/50/75% -> pulih bertahap
                    v = a0 + (1 - a0) * (1 - 0.5 ** j)
                else:            # tanpa info tanggal kembali -> ramp konservatif
                    v = {0: 0.0, 1: 0.0, 2: 0.4, 3: 0.7}.get(j, 0.9)
            elif status == "s":
                v = a0 if j == 0 else 0.9
            else:
                v = a0 if j == 0 else a0 + (1 - a0) * (1 - 0.5 ** j)
            a.append(float(np.clip(v, 0, 1)))
        if overrides.get(pid) == "out":
            a = [0.0] * len(gw_list)
        elif overrides.get(pid) == "fit":
            a = [1.0] * len(gw_list)
        rows[pid] = a
    return pd.DataFrame.from_dict(rows, orient="index", columns=gw_list)


def predict_xp(ctx: ModelContext, gw_list: list[int], overrides: dict | None = None,
               ep_blend: float = 0.0):
    """Return (xp_wide[id x gw], detail per-fixture dengan komponen)."""
    fx = {}
    for f in ctx.fixtures:
        if f.get("finished") or f.get("event") not in gw_list:
            continue
        fx.setdefault((f["team_h"], f["event"]), []).append((f["team_a"], 1))
        fx.setdefault((f["team_a"], f["event"]), []).append((f["team_h"], 0))

    avail = availability_matrix(ctx, gw_list, overrides)
    pid_all = ctx.el["id"].values
    fit_basis = (ctx.el["status"].values != "a") | ctx.el["chance_of_playing_next_round"].notna().values

    recs = []
    for pid, team in zip(pid_all, ctx.el["team"].values):
        for j, gw in enumerate(gw_list):
            for k, (opp, home) in enumerate(fx.get((team, gw), [])):
                recs.append((pid, team, gw, j, k, opp, home))
    if not recs:
        return pd.DataFrame(index=pid_all, columns=gw_list, dtype=float).fillna(0.0), pd.DataFrame()
    R = pd.DataFrame(recs, columns=["element", "team", "gw", "j", "k", "opp", "home"])
    rr = np.full(len(R), ctx.next_gw)
    fb = pd.Series(fit_basis, index=pid_all).reindex(R["element"]).values
    F = _row_frame(ctx, R["element"].values, rr, R["team"].values, R["opp"].values, R["home"].values, fb)
    F = pd.concat([R.reset_index(drop=True), F.drop(columns=["element", "round", "home"])], axis=1)

    # ketersediaan (+ rotasi sedikit lebih besar di laga kedua DGW)
    a = np.array([avail.at[e, g] for e, g in zip(F["element"], F["gw"])])
    sec = np.where(F["k"] > 0, 0.92, 1.0)
    F["avail"] = a
    for c in ("exp_min",):
        F[c] = F[c] * a * sec
    for c in ("p_any", "p60"):
        F[c] = F[c] * a * sec
    comp = structural_components(F)
    F = pd.concat([F.drop(columns=comp.columns), comp], axis=1)
    F["gbm"] = np.clip(ctx.gbm.predict(F[GBM_FEATURES]), 0, None)
    F["xP"] = np.clip(ctx.alpha * F["gbm"] + (1 - ctx.alpha) * F["struct"], 0, None)
    F["xP"] = np.where(F["exp_min"] < 1.0, F["xP"] * F["exp_min"], F["xP"])   # tidak main -> ~0

    if ep_blend > 0:
        ep = ctx.el["ep_next"].reindex(F["element"]).fillna(0).values
        nfix = F.groupby(["element", "gw"])["k"].transform("count").values
        first = (F["gw"] == gw_list[0]).values
        F["xP"] = np.where(first, (1 - ep_blend) * F["xP"] + ep_blend * ep / nfix * a, F["xP"])

    wide = F.groupby(["element", "gw"])["xP"].sum().unstack("gw").reindex(index=pid_all, columns=gw_list).fillna(0.0)
    wide.index.name = "id"
    return wide, F
