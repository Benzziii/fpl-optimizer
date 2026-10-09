import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import pulp
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# FPL OPTIMIZER v2.2 (Fast Engine) — Multi-Period MILP & ML
# ============================================================

st.set_page_config(page_title="FPL Optimizer v2.2", layout="wide")
st.title("⚽ FPL Optimizer v2.2 — Multi-Period MILP & Refined ML")

API = "https://fantasy.premierleague.com/api"

# -----------------------------------------------------------------------------
# 1. OPTIMIZED DATA FETCHING (FAST & FILTERED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_bootstrap():
    r = requests.get(f"{API}/bootstrap-static/", timeout=30)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=1800)
def fetch_fixtures():
    r = requests.get(f"{API}/fixtures/", timeout=30)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_element_summaries_fast(active_player_ids):
    """Mengunduh histori pemain aktif secara efisien untuk menghindari throttling API."""
    def get(pid):
        try:
            r = requests.get(f"{API}/element-summary/{pid}/", timeout=5)
            return pid, (r.json() if r.status_code == 200 else None)
        except Exception:
            return pid, None

    out = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for pid, data in ex.map(get, active_player_ids):
            out[pid] = data
    return out

@st.cache_data(ttl=600)
def fetch_user_entry(entry_id):
    b = fetch_bootstrap()
    if not b:
        return None, "Gagal terhubung ke API FPL."
    nxt = [e["id"] for e in b["events"] if e["is_next"]]
    cur = [e["id"] for e in b["events"] if e["is_current"]]
    gw = (nxt or cur)[0]

    r = requests.get(f"{API}/entry/{entry_id}/event/{gw}/picks/", timeout=20)
    if r.status_code != 200 and gw > 1:
        r = requests.get(f"{API}/entry/{entry_id}/event/{gw-1}/picks/", timeout=20)
    if r.status_code != 200:
        return None, "Entry ID tidak ditemukan / belum ada picks."
    picks = r.json()

    r2 = requests.get(f"{API}/entry/{entry_id}/", timeout=20)
    profile = r2.json() if r2.status_code == 200 else {}
    chip_this_gw = profile.get("active_chip")

    used = []
    r3 = requests.get(f"{API}/entry/{entry_id}/history/", timeout=20)
    if r3.status_code == 200:
        for c in r3.json().get("chips", []):
            m = {"wildcard": "Wildcard", "freehit": "Free Hit",
                 "bboost": "Bench Boost", "3xc": "Triple Captain"}
            if c.get("name") in m:
                used.append(m[c["name"]])
    if chip_this_gw:
        m = {"wildcard": "Wildcard", "freehit": "Free Hit",
             "bboost": "Bench Boost", "3xc": "Triple Captain"}
        nm = m.get(chip_this_gw)
        if nm and nm not in used:
            used.append(nm)

    eh = picks.get("entry_history", {})
    
    picks_data = {}
    for p in picks.get("picks", []):
        sp = p.get("selling_price", 0) / 10.0
        pp = p.get("purchase_price", 0) / 10.0
        picks_data[p["element"]] = {"selling_price": sp, "purchase_price": pp}

    return {
        "player_ids": [p["element"] for p in picks.get("picks", [])],
        "bank": eh.get("bank", 0) / 10.0,
        "ft": max(1, min(5, (eh.get("event_transfers", 0) or 0) + 1)),
        "used_chips": used,
        "active_chip": chip_this_gw,
        "picks_data": picks_data
    }, f"OK — Data GW{gw} diimpor."

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING & ML PREDICTOR
# -----------------------------------------------------------------------------
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

def build_history_df(summaries):
    rows = []
    for pid, s in summaries.items():
        if s and s.get("history"):
            for h in s["history"]:
                rows.append(h)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["element"] = df["element"].astype(int)
    df = df.sort_values(["element", "round"]).reset_index(drop=True)
    num_cols = ["minutes", "total_points", "ict_index", "bonus", "bps",
                "expected_goal_involvements", "expected_goals", "expected_assists",
                "expected_goals_conceded", "goals_scored", "assists", "clean_sheets",
                "saves", "starts"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def add_rolling_features(hist, teams):
    g = hist.groupby("element", group_keys=False)

    for w in (3, 6):
        hist[f"r{w}_xgi"]  = g["expected_goal_involvements"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_ict"]  = g["ict_index"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_min"]  = g["minutes"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_pts"]  = g["total_points"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_xgc"]  = g["expected_goals_conceded"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_starts"] = g["starts"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    hist["season_ppg"] = g["total_points"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean())

    def opp_str(row, kind):
        t = teams.get(int(row["opponent_team"]), {})
        loc = "away" if row["was_home"] else "home"
        return float(t.get(f"strength_{kind}_{loc}", t.get("strength", 3)))
    
    hist["opp_att"] = hist.apply(lambda r: opp_str(r, "attack"), axis=1)
    hist["opp_def"] = hist.apply(lambda r: opp_str(r, "defence"), axis=1)
    return hist

FEATURES = ["pos", "was_home", "opp_att", "opp_def",
            "r3_xgi", "r6_xgi", "r3_ict", "r6_ict",
            "r3_min", "r6_min", "r3_pts", "season_ppg", "r3_xgc", "r3_starts"]

@st.cache_resource(show_spinner=False)
def train_model(train_df):
    df = train_df.dropna(subset=["total_points"]).copy()
    X, y = df[FEATURES], df["total_points"]
    model = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.04, max_depth=5,
        l2_regularization=2.0, random_state=42)
    model.fit(X, y)
    return model

def latest_player_state(hist, elements):
    idx = hist.groupby("element")["round"].idxmax()
    latest = hist.loc[idx, ["element"] + [c for c in FEATURES if c != "pos"]].copy()
    latest = latest.merge(
        elements[["id", "element_type", "status", "chance_of_playing_this_round",
                  "penalties_order", "direct_freekicks_order",
                  "corners_and_indirect_freekicks_order", "ep_next"]],
        left_on="element", right_on="id", how="left")
    latest["pos"] = latest["element_type"]
    for c in ["r3_xgi", "r6_xgi", "r3_ict", "r6_ict", "r3_min", "r6_min",
              "r3_pts", "season_ppg", "r3_xgc", "r3_starts"]:
        latest[c] = latest[c].fillna(0.0)
    
    latest["setpiece_bonus"] = 0.0
    latest.loc[latest["penalties_order"] == 1, "setpiece_bonus"] += 0.5
    latest.loc[latest["direct_freekicks_order"] == 1, "setpiece_bonus"] += 0.2
    latest.loc[latest["corners_and_indirect_freekicks_order"] == 1, "setpiece_bonus"] += 0.2
    return latest

def upcoming_fixtures_by_team(fixtures, gw_list):
    out = {}
    dgw, bgw = {}, {}
    for f in fixtures:
        if f.get("finished") or f["event"] not in gw_list:
            continue
        for team, home in ((f["team_h"], True), (f["team_a"], False)):
            out.setdefault((team, f["event"]), []).append(
                {"home": home, "opp": f["team_a"] if home else f["team_h"]})
    for gw in gw_list:
        for team in set(t for t, _ in out.keys()):
            n = len(out.get((team, gw), []))
            if n == 2:
                dgw[(team, gw)] = True
            elif n == 0:
                bgw[(team, gw)] = True
    return out, dgw, bgw

def predict_xp_all(model, latest, fix_by_team, gw_list, teams, elements):
    team_of = dict(zip(elements["id"], elements["team"]))
    cop = dict(zip(elements["id"], elements["chance_of_playing_this_round"]))
    status = dict(zip(elements["id"], elements["status"]))
    epn = dict(zip(elements["id"], pd.to_numeric(elements["ep_next"], errors="coerce")))

    records = []
    for _, p in latest.iterrows():
        pid, t = int(p["element"]), team_of.get(int(p["element"]))
        
        avail = 1.0
        if status.get(pid) in ("i", "s", "u"):
            avail = float(cop.get(pid) or 0) / 100.0
        elif cop.get(pid) not in (None, 0):
            avail = float(cop.get(pid)) / 100.0
            
        xmins = p["r3_min"] if p["r3_min"] > 0 else 25.0
        min_factor = min(1.0, xmins / 90.0)

        for gw in gw_list:
            fx = fix_by_team.get((t, gw), [])
            if not fx:
                records.append({"id": pid, "gw": gw, "xP": 0.0, "n_fixture": 0})
                continue
            xp_match = 0.0
            for f in fx:
                opp = teams.get(f["opp"], {})
                loc = "away" if f["home"] else "home"
                row = p.copy()
                row["was_home"] = 1.0 if f["home"] else 0.0
                row["opp_att"] = float(opp.get(f"strength_attack_{loc}", opp.get("strength", 3)))
                row["opp_def"] = float(opp.get(f"strength_defence_{loc}", opp.get("strength", 3)))
                xp_match += max(0.0, float(model.predict(row[FEATURES].to_frame().T)[0]))
            
            xp = (xp_match + p["setpiece_bonus"] * len(fx)) * avail * min_factor
            e = epn.get(pid)
            if e and np.isfinite(e):
                xp = 0.82 * xp + 0.18 * float(e) * avail * len(fx)
            records.append({"id": pid, "gw": gw, "xP": round(xp, 2), "n_fixture": len(fx)})
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 3. MULTI-PERIOD MILP SOLVER (FIXED TYPES)
# -----------------------------------------------------------------------------
def solve_multi_period_milp(df, current_ids, bank, free_transfers, active_chip, horizon_gws, picks_data):
    all_ids = [int(x) for x in df["id"].unique()]
    gws = [int(x) for x in horizon_gws]
    current_ids = [int(x) for x in current_ids]
    
    el_df = df.drop_duplicates("id").set_index("id")
    now_cost = {int(k): float(v)/10.0 for k, v in el_df["now_cost"].to_dict().items()}
    team = {int(k): int(v) for k, v in el_df["team"].to_dict().items()}
    pos = {int(k): int(v) for k, v in el_df["element_type"].to_dict().items()}
    
    sell_price = {}
    for i in all_ids:
        nc = now_cost[i]
        if i in picks_data:
            pp = float(picks_data[i]["purchase_price"])
            profit = max(0.0, nc - pp)
            sell_price[i] = round(pp + np.floor(profit * 10) / 20.0, 1)
        else:
            sell_price[i] = nc

    xp_dict = {i: {} for i in all_ids}
    for _, r in df.iterrows():
        xp_dict[int(r["id"])][int(r["gw"])] = float(r["xP"])

    prob = pulp.LpProblem("FPL_MultiPeriod_v2_2", pulp.LpMaximize)

    squad = {(i, t): pulp.LpVariable(f"s_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    xi    = {(i, t): pulp.LpVariable(f"x_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    cap   = {(i, t): pulp.LpVariable(f"c_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    vc    = {(i, t): pulp.LpVariable(f"v_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    tin   = {(i, t): pulp.LpVariable(f"in_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    tout  = {(i, t): pulp.LpVariable(f"out_{int(i)}_g{int(t)}", cat="Binary") for i in all_ids for t in gws}
    
    ft_avail = {t: pulp.LpVariable(f"ft_avail_g{int(t)}", lowBound=1, upBound=5, cat="Integer") for t in gws}
    ft_used  = {t: pulp.LpVariable(f"ft_used_g{int(t)}", lowBound=0, upBound=5, cat="Integer") for t in gws}
    hits     = {t: pulp.LpVariable(f"hits_g{int(t)}", lowBound=0, cat="Integer") for t in gws}

    cur_set = set(current_ids)
    
    for idx_t, t in enumerate(gws):
        for i in all_ids:
            prev_in = 1 if (idx_t == 0 and i in cur_set) else (squad[(i, gws[idx_t-1])] if idx_t > 0 else 0)
            prob += squad[(i, t)] == prev_in + tin[(i, t)] - tout[(i, t)]
            prob += tin[(i, t)] + tout[(i, t)] <= 1
            prob += xi[(i, t)] <= squad[(i, t)]
            prob += cap[(i, t)] <= xi[(i, t)]
            prob += vc[(i, t)] <= xi[(i, t)]
            prob += cap[(i, t)] + vc[(i, t)] <= 1

        n_trans = pulp.lpSum([tin[(i, t)] for i in all_ids])
        prob += pulp.lpSum([squad[(i, t)] for i in all_ids]) == 15
        prob += pulp.lpSum([xi[(i, t)] for i in all_ids]) == 11
        prob += pulp.lpSum([cap[(i, t)] for i in all_ids]) == 1
        prob += pulp.lpSum([vc[(i, t)] for i in all_ids]) == 1

        for p, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
            prob += pulp.lpSum([squad[(i, t)] for i in all_ids if pos[i] == p]) == n
        prob += pulp.lpSum([xi[(i, t)] for i in all_ids if pos[i] == 1]) == 1
        prob += pulp.lpSum([xi[(i, t)] for i in all_ids if pos[i] == 2]) >= 3
        prob += pulp.lpSum([xi[(i, t)] for i in all_ids if pos[i] == 3]) >= 2
        prob += pulp.lpSum([xi[(i, t)] for i in all_ids if pos[i] == 4]) >= 1

        for tm in set(team.values()):
            prob += pulp.lpSum([squad[(i, t)] for i in all_ids if team[i] == tm]) <= 3

        if idx_t == 0:
            prob += ft_avail[t] == free_transfers
        else:
            prev_t = gws[idx_t - 1]
            prob += ft_avail[t] <= ft_avail[prev_t] - ft_used[prev_t] + 1
            prob += ft_avail[t] <= 5

        chip_t = active_chip if idx_t == 0 else "Tanpa Chip"
        if chip_t in ("Wildcard", "Free Hit"):
            prob += hits[t] == 0
            prob += ft_used[t] == 0
        else:
            prob += ft_used[t] <= ft_avail[t]
            prob += ft_used[t] <= n_trans
            prob += hits[t] >= n_trans - ft_used[t]

    cur_squad_value = sum(sell_price.get(i, now_cost[i]) for i in current_ids if i in sell_price)
    max_budget = cur_squad_value + bank
    prob += pulp.lpSum([now_cost[i] * squad[(i, gws[0])] for i in all_ids]) <= max_budget + 0.001

    total_obj = []
    for idx_t, t in enumerate(gws):
        chip_t = active_chip if idx_t == 0 else "Tanpa Chip"
        is_bb = chip_t == "Bench Boost"
        is_tc = chip_t == "Triple Captain"
        cap_mult = 3.0 if is_tc else 2.0
        decay = 1.0 if idx_t == 0 else (0.85 ** idx_t)

        gw_xp = (
            pulp.lpSum([xp_dict[i].get(t, 0) * xi[(i, t)] for i in all_ids])
            + pulp.lpSum([xp_dict[i].get(t, 0) * cap[(i, t)] for i in all_ids]) * (cap_mult - 1)
            + (pulp.lpSum([xp_dict[i].get(t, 0) * (squad[(i, t)] - xi[(i, t)]) for i in all_ids]) if is_bb else 0)
            - hits[t] * 4.0
        )
        total_obj.append(gw_xp * decay)

    prob += pulp.lpSum(total_obj)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    sol_gw1 = lambda var: [i for i in all_ids if pulp.value(var[(i, gws[0])]) == 1]
    
    return {
        "squad": sol_gw1(squad),
        "xi": sol_gw1(xi),
        "captain": sol_gw1(cap),
        "vice": sol_gw1(vc),
        "net_xP": pulp.value(total_obj[0]),
        "n_transfers": int(pulp.value(pulp.lpSum([tin[(i, gws[0])] for i in all_ids]))),
        "hit": int(pulp.value(hits[gws[0]])) * 4,
    }

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------------------------------------------
st.sidebar.header("📥 Import Data FPL")
fpl_id = st.sidebar.text_input("Entry ID FPL (opsional):", placeholder="Contoh: 123456")
st.session_state.setdefault("user_ids", [])
st.session_state.setdefault("bank", 0.5)
st.session_state.setdefault("ft", 1)
st.session_state.setdefault("used_chips", [])
st.session_state.setdefault("picks_data", {})

with st.spinner("Mengunduh data FPL..."):
    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()

if not bootstrap or fixtures is None:
    st.error("Gagal terhubung ke API FPL.")
    st.stop()

elements = pd.DataFrame(bootstrap["elements"])
teams = {t["id"]: t for t in bootstrap["teams"]}
team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
elements["team_name"] = elements["team"].map(team_names)

events = bootstrap["events"]
next_gw = [e["id"] for e in events if e["is_next"]][0]
horizon_gws = [next_gw + k for k in range(3) if next_gw + k <= max(e["id"] for e in events)]

if st.sidebar.button("🔽 Import Skuad Saya") and fpl_id:
    data, msg = fetch_user_entry(fpl_id.strip())
    if data:
        st.session_state.user_ids = data["player_ids"]
        st.session_state.bank = data["bank"]
        st.session_state.ft = data["ft"]
        st.session_state.used_chips = data["used_chips"]
        st.session_state.picks_data = data["picks_data"]
        st.sidebar.success(msg)

bank = st.sidebar.number_input("Sisa budget bank (£m):", 0.0, 30.0, float(st.session_state.bank), 0.1)
ft = st.sidebar.number_input("Free transfer tersedia (Maks 5):", 1, 5, int(st.session_state.ft))
used_chips = st.session_state.used_chips
avail_chips = [c for c in ["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"] if c not in used_chips]
active_chip = st.sidebar.selectbox("⚡ Chip aktif pekan ini:", ["Tanpa Chip"] + avail_chips)
horizon = st.sidebar.slider("Horizon perencanaan (GW ke depan):", 1, 3, 2)

with st.spinner("⚡ Mengambil data histori pemain aktif (Fast Engine)..."):
    active_elements = elements[(elements["total_points"] > 0) | (elements["minutes"] > 0)]
    active_ids = active_elements["id"].tolist()
    
    summaries = fetch_element_summaries_fast(active_ids)
    hist = build_history_df(summaries)
    
    hist_feat = add_rolling_features(hist.copy(), teams)
    hist_feat = hist_feat.merge(elements[["id", "element_type"]], left_on="element", right_on="id", how="left")
    hist_feat["pos"] = hist_feat["element_type"]
    model = train_model(hist_feat)

latest = latest_player_state(hist_feat, elements)
fix_by_team, dgw_set, bgw_set = upcoming_fixtures_by_team(fixtures, horizon_gws)
xp_long = predict_xp_all(model, latest, fix_by_team, horizon_gws, teams, elements)

df_opt_full = xp_long.merge(elements[["id", "now_cost", "team", "element_type", "web_name", "status"]], on="id")

st.subheader("📋 Skuad Saat Ini")
default_names = elements[elements["id"].isin(st.session_state.user_ids)]["web_name"].tolist()
sel_names = st.multiselect("Pilih 15 pemain Anda:", elements["web_name"].tolist(), default=default_names)
current_df = elements[elements["web_name"].isin(sel_names)].copy()

if len(current_df) == 15:
    cur_cost = (current_df["now_cost"] / 10.0).sum()
    st.caption(f"Total harga skuad: £{cur_cost:.1f}m | Bank: £{bank:.1f}m")
    
    if st.button("🎯 JALANKAN OPTIMASI MULTI-PERIOD (MILP)", type="primary", use_container_width=True):
        with st.spinner("MILP Multi-Period solver mencari keputusan transfer paling presisi..."):
            res = solve_multi_period_milp(
                df_opt_full, current_df["id"].tolist(), bank, ft,
                active_chip, horizon_gws[:horizon], st.session_state.picks_data
            )

        out_ids = set(current_df["id"]) - set(res["squad"])
        in_ids = set(res["squad"]) - set(current_df["id"])
        t_out = elements[elements["id"].isin(out_ids)]
        t_in = elements[elements["id"].isin(in_ids)]

        st.divider()
        st.subheader("1. 🔄 Rekomendasi Transfer (Pekan Ini)")
        if active_chip in ("Wildcard", "Free Hit"):
            st.success(f"🎉 **{active_chip} AKTIF** — {res['n_transfers']} perubahan skuad, **tanpa penalti**.")
        elif res["n_transfers"] == 0:
            st.success("✅ **Tidak perlu transfer** — skuad Anda sudah optimal. Simpan Free Transfer untuk pekan depan.")
        else:
            free_used = min(res["n_transfers"], ft)
            paid = res["n_transfers"] - free_used
            st.success(f"✅ **{res['n_transfers']} transfer** ({free_used} gratis, {paid} berbayar → **-{res['hit']} poin**). Total xP multi-GW terbukti lebih tinggi setelah memotong hit.")

        c1, c2 = st.columns(2)
        c1.markdown("🔴 **Keluar:**")
        c1.dataframe(t_out[["web_name", "team_name", "now_cost"]].assign(Harga=t_out["now_cost"]/10.0)[["web_name", "team_name", "Harga"]].rename(columns={"web_name": "Pemain", "team_name": "Klub"}), use_container_width=True, hide_index=True)
        c2.markdown("🟢 **Masuk:**")
        c2.dataframe(t_in[["web_name", "team_name", "now_cost"]].assign(Harga=t_in["now_cost"]/10.0)[["web_name", "team_name", "Harga"]].rename(columns={"web_name": "Pemain", "team_name": "Klub"}), use_container_width=True, hide_index=True)

        gw1_xp_df = xp_long[xp_long["gw"] == next_gw].set_index("id")["xP"]
        elements["xP_GW1"] = elements["id"].map(gw1_xp_df).fillna(0.0)

        xi_df = elements[elements["id"].isin(res["xi"])].copy()
        cap_df = elements[elements["id"].isin(res["captain"])].iloc[0]
        vc_df = elements[elements["id"].isin(res["vice"])].iloc[0]
        bench_df = elements[elements["id"].isin(set(res["squad"]) - set(res["xi"]))].sort_values("xP_GW1", ascending=False)
        
        cap_pts = cap_df["xP_GW1"] * (3.0 if active_chip == "Triple Captain" else 2.0)
        net_pts = xi_df["xP_GW1"].sum() + cap_df["xP_GW1"] - res["hit"] + (bench_df["xP_GW1"].sum() if active_chip == "Bench Boost" else 0)

        st.subheader("2. 🏆 Starting XI, Kapten & Proyeksi")
        k1, k2, k3 = st.columns(3)
        k1.metric("👑 Kapten", cap_df["web_name"], f"{cap_pts:.1f} xP")
        k2.metric("🎖️ Vice-Kapten", vc_df["web_name"], f"{vc_df['xP_GW1']:.1f} xP")
        k3.metric(f"📊 Proyeksi GW{next_gw} (setelah hit)", f"{net_pts:.1f} xP", delta=f"-{res['hit']} hit" if res["hit"] else "tanpa hit")

        st.markdown("**Starting XI:**")
        st.dataframe(xi_df.assign(Posisi=xi_df["element_type"].map(POS)).sort_values(["Posisi", "xP_GW1"], ascending=[True, False])[["web_name", "team_name", "Posisi", "status", "xP_GW1"]].rename(columns={"web_name": "Pemain", "team_name": "Klub", "status": "Status", "xP_GW1": "xP GW1"}), use_container_width=True, hide_index=True)
        
        st.markdown("**Bench:**")
        st.dataframe(bench_df.assign(Posisi=bench_df["element_type"].map(POS))[["web_name", "team_name", "Posisi", "xP_GW1"]].rename(columns={"web_name": "Pemain", "team_name": "Klub", "xP_GW1": "xP GW1"}), use_container_width=True, hide_index=True)
else:
    st.info(f"Pilih tepat **15 pemain** (sekarang: {len(current_df)}). Atau import via Entry ID di sidebar.")