import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import pulp
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# FPL OPTIMIZER v2.0 — Two-Stage Model (ML Predictor + MILP)
# Stage 1: xP Predictor (Gradient Boosting dari histori GW)
# Stage 2: MILP Solver (transfer logic, hit cost, chip, horizon)
# ============================================================

st.set_page_config(page_title="FPL Optimizer v2.0", layout="wide")
st.title("⚽ FPL Optimizer v2.0 — ML Predictor & MILP Solver")

API = "https://fantasy.premierleague.com/api"

# -----------------------------------------------------------------------------
# 1. DATA FETCHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_bootstrap():
    r = requests.get(f"{API}/bootstrap-static/", timeout=30)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=1800)
def fetch_fixtures():
    r = requests.get(f"{API}/fixtures/", timeout=30)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_element_summaries(player_ids):
    """Ambil histori per-gameweek semua pemain (parallel, cached)."""
    def get(pid):
        try:
            r = requests.get(f"{API}/element-summary/{pid}/", timeout=15)
            return pid, (r.json() if r.status_code == 200 else None)
        except Exception:
            return pid, None
    out = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for pid, data in ex.map(get, player_ids):
            out[pid] = data
    return out

@st.cache_data(ttl=600)
def fetch_user_entry(entry_id):
    """Ambil skuad, bank, free transfer, dan chip dari entry ID FPL."""
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
    chip_this_gw = profile.get("active_chip")  # chip yg sedang aktif pekan ini

    # Chip yang sudah dipakai (musim ini) — cek history
    used = []
    r3 = requests.get(f"{API}/entry/{entry_id}/history/", timeout=20)
    if r3.status_code == 200:
        for c in r3.json().get("chips", []):
            m = {"wildcard": "Wildcard", "freehit": "Free Hit",
                 "bboost": "Bench Boost", "3xc": "Triple Captain"}
            if c.get("name") in m:
                used.append(m[c["name"]])
    if chip_this_gw:  # chip aktif tapi mungkin belum masuk history
        m = {"wildcard": "Wildcard", "freehit": "Free Hit",
             "bboost": "Bench Boost", "3xc": "Triple Captain"}
        nm = m.get(chip_this_gw)
        if nm and nm not in used:
            used.append(nm)

    eh = picks.get("entry_history", {})
    return {
        "player_ids": [p["element"] for p in picks.get("picks", [])],
        "bank": eh.get("bank", 0) / 10.0,
        "ft": eh.get("event_transfers", 0),
        "ft_cost": eh.get("event_transfers_cost", 0) / 10.0,
        "used_chips": used,
        "active_chip": chip_this_gw,
    }, f"OK — Data GW{gw} diimpor."

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (dok: Variabel 1 & 2 — Performa + Konteks Fixture)
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
    """Fitur lag (hanya info yang tersedia SEBELUM gameweek dimainkan)."""
    g = hist.groupby("element", group_keys=False)

    for w in (3, 6):
        hist[f"r{w}_xgi"]  = g["expected_goal_involvements"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_ict"]  = g["ict_index"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_min"]  = g["minutes"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_pts"]  = g["total_points"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        hist[f"r{w}_xgc"]  = g["expected_goals_conceded"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    # Poin per game musim berjalan (expanding, shift 1)
    hist["season_ppg"] = g["total_points"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Kekuatan lawan (FDR berbasis team strength resmi FPL)
    def opp_str(row, kind):
        t = teams.get(int(row["opponent_team"]), {})
        loc = "away" if row["was_home"] else "home"  # lawan main di mana
        return float(t.get(f"strength_{kind}_{loc}", t.get("strength", 3)))
    hist["opp_att"] = hist.apply(lambda r: opp_str(r, "attack"), axis=1)
    hist["opp_def"] = hist.apply(lambda r: opp_str(r, "defence"), axis=1)
    return hist

FEATURES = ["pos", "was_home", "opp_att", "opp_def",
            "r3_xgi", "r6_xgi", "r3_ict", "r6_ict",
            "r3_min", "r6_min", "r3_pts", "season_ppg", "r3_xgc"]

@st.cache_resource(show_spinner=False)
def train_model(train_df):
    df = train_df.dropna(subset=["total_points"]).copy()
    X, y = df[FEATURES], df["total_points"]
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=42)
    model.fit(X, y)
    return model

def latest_player_state(hist, elements):
    """Fitur terbaru tiap pemain untuk prediksi GW berikutnya."""
    idx = hist.groupby("element")["round"].idxmax()
    latest = hist.loc[idx, ["element"] + [c for c in FEATURES if c != "pos"]].copy()
    latest = latest.merge(
        elements[["id", "element_type", "status", "chance_of_playing_this_round",
                  "penalties_order", "direct_freekicks_order",
                  "corners_and_indirect_freekicks_order", "ep_next"]],
        left_on="element", right_on="id", how="left")
    latest["pos"] = latest["element_type"]
    # NaN rolling (pemain baru/tanpa histori) → isi 0
    for c in ["r3_xgi", "r6_xgi", "r3_ict", "r6_ict", "r3_min", "r6_min",
              "r3_pts", "season_ppg", "r3_xgc"]:
        latest[c] = latest[c].fillna(0.0)
    # Set-piece status (dok: variabel 1) → bonus xP kecil
    latest["setpiece_bonus"] = 0.0
    latest.loc[latest["penalties_order"] == 1, "setpiece_bonus"] += 0.6
    latest.loc[latest["direct_freekicks_order"] == 1, "setpiece_bonus"] += 0.2
    latest.loc[latest["corners_and_indirect_freekicks_order"] == 1, "setpiece_bonus"] += 0.2
    return latest

def upcoming_fixtures_by_team(fixtures, gw_list):
    """fixture difficulty per tim per GW + deteksi DGW/BGW (dok: variabel 2)."""
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
    """Prediksi xP per pemain per GW. DGW = jumlah xP dua laga."""
    team_of = dict(zip(elements["id"], elements["team"]))
    price = dict(zip(elements["id"], elements["now_cost"] / 10.0))
    cop = dict(zip(elements["id"], elements["chance_of_playing_this_round"]))
    status = dict(zip(elements["id"], elements["status"]))
    epn = dict(zip(elements["id"], pd.to_numeric(elements["ep_next"], errors="coerce")))

    records = []
    for _, p in latest.iterrows():
        pid, t = int(p["element"]), team_of.get(int(p["element"]))
        # Faktor ketersediaan (dok: variabel 3)
        avail = 1.0
        if status.get(pid) in ("i", "s", "u"):
            avail = float(cop.get(pid) or 0) / 100.0
        elif cop.get(pid) not in (None, 0):
            avail = float(cop.get(pid)) / 100.0
        # Faktor menit: rotasi (r3_min rendah -> xP diturunkan proporsional)
        min_factor = min(1.0, (p["r3_min"] or 0) / 65.0) if (p["r3_min"] or 0) > 0 else 0.35

        for gw in gw_list:
            fx = fix_by_team.get((t, gw), [])
            if not fx:  # BGW
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
            # Blend ringan dengan ep_next resmi FPL sebagai sanity anchor
            e = epn.get(pid)
            if e and np.isfinite(e):
                xp = 0.85 * xp + 0.15 * float(e) * avail * len(fx)
            records.append({"id": pid, "gw": gw, "xP": round(xp, 2), "n_fixture": len(fx)})
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# 3. STAGE 2: MILP SOLVER (dok: Optimization Solver)
#    - Transfer in/out dengan biaya hit (-4) vs free transfer
#    - Chip: Wildcard/Free Hit (0 hit), Bench Boost (+xP bench), Triple Captain (3x)
#    - Horizon: transfer dinilai dari total xP beberapa GW ke depan
# -----------------------------------------------------------------------------
def solve_milp(df, current_ids, bank, free_transfers, active_chip, horizon_gws):
    price = dict(zip(df["id"], df["now_cost"] / 10.0))
    team  = dict(zip(df["id"], df["team"]))
    pos   = dict(zip(df["id"], df["element_type"]))
    gw1_xp = dict(zip(df["id"], df["xP_h1"]))
    horizon_xp = dict(zip(df["id"], df["xP_horizon"]))  # sum GW2..N (0 jika hanya 1 GW)

    all_ids = list(df["id"])
    cur_cost = sum(price[i] for i in current_ids if i in price)
    cur_set = set(current_ids)

    prob = pulp.LpProblem("FPL_v2", pulp.LpMaximize)
    squad = {i: pulp.LpVariable(f"s_{i}", cat="Binary") for i in all_ids}
    xi    = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in all_ids}
    cap   = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in all_ids}
    vc    = {i: pulp.LpVariable(f"v_{i}", cat="Binary") for i in all_ids}
    tin   = {i: pulp.LpVariable(f"in_{i}", cat="Binary") for i in all_ids}
    tout  = {i: pulp.LpVariable(f"out_{i}", cat="Binary") for i in all_ids}

    for i in all_ids:
        c = 1 if i in cur_set else 0
        prob += tin[i] >= squad[i] - c
        prob += tout[i] >= c - squad[i]
        prob += tin[i] + tout[i] <= 1
        prob += xi[i] <= squad[i]
        prob += cap[i] <= xi[i]
        prob += vc[i] <= xi[i]
        prob += cap[i] + vc[i] <= 1

    n_trans = pulp.lpSum([tin[i] for i in all_ids])
    prob += n_trans == pulp.lpSum([tout[i] for i in all_ids])
    prob += pulp.lpSum([squad[i] for i in all_ids]) == 15
    prob += pulp.lpSum([xi[i] for i in all_ids]) == 11
    prob += pulp.lpSum([cap[i] for i in all_ids]) == 1
    prob += pulp.lpSum([vc[i] for i in all_ids]) == 1

    # Biaya: harga beli = now_cost (harga jual tidak dimodelkan — catatan UI)
    prob += pulp.lpSum([price[i] * squad[i] for i in all_ids]) <= cur_cost + bank + 0.001

    for p, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
        prob += pulp.lpSum([squad[i] for i in all_ids if pos[i] == p]) == n
    prob += pulp.lpSum([xi[i] for i in all_ids if pos[i] == 1]) == 1
    prob += pulp.lpSum([xi[i] for i in all_ids if pos[i] == 2]) >= 3
    prob += pulp.lpSum([xi[i] for i in all_ids if pos[i] == 3]) >= 2
    prob += pulp.lpSum([xi[i] for i in all_ids if pos[i] == 4]) >= 1

    for t in set(team.values()):
        prob += pulp.lpSum([squad[i] for i in all_ids if team[i] == t]) <= 3

    # Hit cost: -4 poin per transfer melebihi kuota FT, KECUALI WC/FH aktif
    no_hit = active_chip in ("Wildcard", "Free Hit")
    extra = pulp.LpVariable("extra_tr", lowBound=0, cat="Integer")
    if not no_hit:
        prob += extra >= n_trans - free_transfers
        hit_penalty = 4.0 * extra
    else:
        hit_penalty = 0.0

    # Objective: xP GW1 (lineup + kapten) + xP horizon skuad (GW2..N)
    is_bb   = active_chip == "Bench Boost"
    is_tc   = active_chip == "Triple Captain"
    cap_mult = 2.0 + (1.0 if is_tc else 0.0)   # kapten biasa 2x, TC 3x
    obj = (
        pulp.lpSum([gw1_xp[i] * xi[i] for i in all_ids])
        + pulp.lpSum([gw1_xp[i] * cap[i] for i in all_ids]) * (cap_mult - 1)
        + (pulp.lpSum([gw1_xp[i] * (squad[i] - xi[i]) for i in all_ids]) if is_bb else 0)
        + pulp.lpSum([horizon_xp[i] * squad[i] for i in all_ids]) * 0.7
        - hit_penalty
    )
    prob += obj
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    sol = lambda d: [i for i in all_ids if pulp.value(d[i]) == 1]
    return {
        "squad": sol(squad), "xi": sol(xi), "captain": sol(cap), "vice": sol(vc),
        "net_xP": pulp.value(obj),
        "n_transfers": int(pulp.value(n_trans)),
        "hit": 0 if no_hit else int(pulp.value(extra)) * 4,
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

with st.spinner("Mengunduh data FPL (bootstrap, fixtures, histori semua pemain)..."):
    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()

if not bootstrap or fixtures is None:
    st.error("Gagal terhubung ke API Fantasy Premier League. Coba beberapa saat lagi.")
    st.stop()

elements = pd.DataFrame(bootstrap["elements"])
teams = {t["id"]: t for t in bootstrap["teams"]}
team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
elements["team_name"] = elements["team"].map(team_names)

events = bootstrap["events"]
done_gws = [e["id"] for e in events if e["is_current"]]
next_gw = [e["id"] for e in events if e["is_next"]][0]
horizon_gws = [next_gw + k for k in range(3) if next_gw + k <= max(e["id"] for e in events)]

if st.sidebar.button("🔽 Import Skuad Saya") and fpl_id:
    data, msg = fetch_user_entry(fpl_id.strip())
    if data:
        st.session_state.user_ids = data["player_ids"]
        st.session_state.bank = data["bank"]
        st.session_state.ft = max(1, min(5, data["ft"] + 1))
        st.session_state.used_chips = data["used_chips"]
        st.sidebar.success(msg)
        if data["used_chips"]:
            st.sidebar.info(f"Chip terpakai musim ini: {', '.join(data['used_chips'])}")
    else:
        st.sidebar.error(msg)

bank = st.sidebar.number_input("Sisa budget bank (£m):", 0.0, 30.0,
                               float(st.session_state.bank), 0.1)
ft = st.sidebar.number_input("Free transfer tersedia:", 0, 5, int(st.session_state.ft))
used_chips = st.session_state.used_chips
all_chips = ["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"]
avail_chips = [c for c in all_chips if c not in used_chips]
active_chip = st.sidebar.selectbox("⚡ Chip aktif pekan ini:", ["Tanpa Chip"] + avail_chips)
horizon = st.sidebar.slider("Horizon perencanaan (GW ke depan):", 1, 3, 1)
st.sidebar.caption("Model: Gradient Boosting + MILP (solusi matematis optimal)")

# --- Stage 1: bangun & latih model ---
with st.spinner("⏳ Membangun dataset histori & melatih model ML (±30–60 detik pertama kali)..."):
    summaries = fetch_element_summaries(elements["id"].tolist())
    hist = build_history_df(summaries)

if hist.empty:
    st.error("Histori pemain kosong — musim mungkin belum dimulai.")
    st.stop()

hist_feat = add_rolling_features(hist.copy(), teams)
hist_feat = hist_feat.merge(elements[["id", "element_type"]],
                            left_on="element", right_on="id", how="left")
hist_feat["pos"] = hist_feat["element_type"]
model = train_model(hist_feat)
st.success(f"✅ Model dilatih dari **{len(hist):,} baris data per-gameweek** "
           f"(xG, xA, ICT, menit, BPS, home/away, kekuatan lawan).")

# fitur rolling/opp hanya ada di hist_feat (bukan hist mentah)
latest = latest_player_state(hist_feat, elements)
fix_by_team, dgw_set, bgw_set = upcoming_fixtures_by_team(fixtures, horizon_gws)
xp_long = predict_xp_all(model, latest, fix_by_team, horizon_gws, teams, elements)

# pivot xP per GW
xp_pivot = xp_long.pivot_table(index="id", columns="gw", values="xP", aggfunc="sum").fillna(0.0)
xp_pivot = xp_pivot.reindex(columns=sorted(xp_pivot.columns))
# normalisasi posisional: GW berikutnya -> xP_h1, GW+2 -> xP_h2, dst.
xp_pivot.columns = [f"xP_h{k+1}" for k in range(xp_pivot.shape[1])]
elements = elements.merge(xp_pivot, left_on="id", right_index=True, how="left").fillna({"xP_h1": 0})
for c in xp_pivot.columns:
    elements[c] = elements[c].fillna(0.0)
elements["xP_horizon"] = elements[[c for c in xp_pivot.columns if c != "xP_h1"]].sum(axis=1)

# --- Peringatan DGW / BGW (kunci chip BB & TC) ---
dgw_teams = sorted({team_names[t] for (t, gw) in dgw_set if gw == next_gw})
bgw_teams = sorted({team_names[t] for (t, gw) in bgw_set if gw == next_gw})
if dgw_teams:
    st.info(f"🔁 **DGW{next_gw}** — main 2x: {', '.join(dgw_teams)} "
            f"→ kandidat kuat **Triple Captain / Bench Boost**.")
if bgw_teams:
    st.warning(f"🚫 **BGW{next_gw}** — tidak main: {', '.join(bgw_teams)} "
               f"→ hindari beli pemain tim ini; **Free Hit** layak dipertimbangkan.")

# --- Input skuad ---
st.subheader("📋 Skuad Saat Ini")
default_names = elements[elements["id"].isin(st.session_state.user_ids)]["web_name"].tolist()
sel_names = st.multiselect("Pilih 15 pemain Anda (atau import otomatis via Entry ID):",
                           elements["web_name"].tolist(), default=default_names)
current_df = elements[elements["web_name"].isin(sel_names)].copy()

if len(current_df) == 15:
    cur_cost = (current_df["now_cost"] / 10.0).sum()
    st.caption(f"Total harga skuad: £{cur_cost:.1f}m | Bank: £{bank:.1f}m")
    st.dataframe(
        current_df[["web_name", "team_name", "element_type", "now_cost", "status",
                    "xP_h1"]].rename(columns={
            "web_name": "Pemain", "team_name": "Klub",
            "element_type": "Posisi", "now_cost": "Harga (£0.1m)",
            "status": "Status", "xP_h1": f"xP GW{next_gw}"}),
        use_container_width=True, hide_index=True)

    if st.button("🎯 JALANKAN OPTIMASI (MILP)", type="primary", use_container_width=True):
        with st.spinner("MILP solver mencari solusi optimal..."):
            df_opt = elements[elements["xP_h1"] > 0].copy()
            res = solve_milp(df_opt, current_df["id"].tolist(), bank, ft,
                             active_chip, horizon_gws[:horizon])

        out_ids = set(current_df["id"]) - set(res["squad"])
        in_ids = set(res["squad"]) - set(current_df["id"])
        t_out = elements[elements["id"].isin(out_ids)]
        t_in = elements[elements["id"].isin(in_ids)]

        st.divider()
        st.subheader("1. 🔄 Rekomendasi Transfer")
        if active_chip in ("Wildcard", "Free Hit"):
            st.success(f"🎉 **{active_chip} AKTIF** — {res['n_transfers']} perubahan, **tanpa penalti**.")
        elif res["n_transfers"] == 0:
            st.success("✅ **Tidak perlu transfer** — skuad saat ini sudah optimal.")
        else:
            free_used = min(res["n_transfers"], ft)
            paid = res["n_transfers"] - free_used
            st.success(f"✅ **{res['n_transfers']} transfer** "
                       f"({free_used} gratis, {paid} berbayar → **-{res['hit']} poin**). "
                       f"Total xP setelah hit tetap lebih tinggi.")

        c1, c2 = st.columns(2)
        c1.markdown("🔴 **Keluar:**")
        c1.dataframe(t_out[["web_name", "team_name", "xP_h1"]].rename(
            columns={"web_name": "Pemain", "team_name": "Klub", "xP_h1": "xP"}),
            use_container_width=True, hide_index=True)
        c2.markdown("🟢 **Masuk:**")
        c2.dataframe(t_in[["web_name", "team_name", "xP_h1"]].rename(
            columns={"web_name": "Pemain", "team_name": "Klub", "xP_h1": "xP"}),
            use_container_width=True, hide_index=True)

        # Kapten/VC: prioritas pemain DGW & xP tertinggi
        xi_df = elements[elements["id"].isin(res["xi"])].copy()
        cap_df = elements[elements["id"].isin(res["captain"])].iloc[0]
        vc_df = elements[elements["id"].isin(res["vice"])].iloc[0]
        bench_df = elements[elements["id"].isin(set(res["squad"]) - set(res["xi"]))] \
            .sort_values("xP_h1", ascending=False)
        cap_pts = cap_df["xP_h1"] * (3.0 if active_chip == "Triple Captain" else 2.0)
        net_pts = xi_df["xP_h1"].sum() + cap_df["xP_h1"] - res["hit"] \
            + (bench_df["xP_h1"].sum() if active_chip == "Bench Boost" else 0)

        st.subheader("2. 🏆 Starting XI, Kapten & Proyksi")
        k1, k2, k3 = st.columns(3)
        k1.metric("👑 Kapten" + (" (3x!)" if active_chip == "Triple Captain" else ""),
                  cap_df["web_name"], f"{cap_pts:.1f} poin")
        k2.metric("🎖️ Vice-Kapten", vc_df["web_name"], f"{vc_df['xP_gw1']:.1f} poin")
        k3.metric(f"📊 Proyksi GW{next_gw} (setelah hit)", f"{net_pts:.1f} poin",
                  delta=f"-{res['hit']} hit" if res["hit"] else "tanpa hit")

        st.markdown(f"**Starting XI ({POS}-wise):**")
        st.dataframe(
            xi_df.assign(Posisi=xi_df["element_type"].map(POS))
                 .sort_values(["Posisi", "xP_h1"], ascending=[True, False])
                 [["web_name", "team_name", "Posisi", "status", "xP_h1"]]
                 .rename(columns={"web_name": "Pemain", "team_name": "Klub",
                                  "status": "Status", "xP_h1": "xP"}),
            use_container_width=True, hide_index=True)
        st.markdown("**Bench:**")
        st.dataframe(
            bench_df.assign(Posisi=bench_df["element_type"].map(POS))
                    [["web_name", "team_name", "Posisi", "xP_h1"]]
                    .rename(columns={"web_name": "Pemain", "team_name": "Klub",
                                     "xP_h1": "xP"}),
            use_container_width=True, hide_index=True)

        if horizon > 1:
            st.subheader(f"3. 🔭 Proyksi xP per GW (horizon {horizon} GW)")
            show = xi_df[["web_name"] + [f"xP_h{k+1}" for k in range(horizon)]] \
                .rename(columns={"web_name": "Pemain"})
            st.dataframe(show, use_container_width=True, hide_index=True)

        st.caption("Catatan: harga jual memakai now_cost (perubahan harga jual-beli FPL tidak dimodelkan); "
                   "xP = model GBM dari histori GW + konteks fixture; cek injury manual sebelum deadline.")
else:
    st.info(f"Pilih tepat **15 pemain** (sekarang: {len(current_df)}). Atau import via Entry ID di sidebar.")