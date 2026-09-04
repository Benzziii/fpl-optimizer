import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Realistic MILP Optimizer", layout="wide")
st.title("⚡ FPL Ultra Optimizer: Fixtures, Matchup & MILP")

# -----------------------------------------------------------------------------
# 1. FETCH DATA & FIXTURES FROM FPL API
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_fpl_bootstrap_v2():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else None

@st.cache_data(ttl=3600)
def fetch_next_fixtures_v2():
    bootstrap = fetch_fpl_bootstrap_v2()
    if not bootstrap: 
        return {}, None
    
    next_gw = [gw['id'] for gw in bootstrap['events'] if gw['is_next'] or gw['is_current']][0]
    url = f"https://fantasy.premierleague.com/api/fixtures/?event={next_gw}"
    res = requests.get(url)
    if res.status_code != 200: 
        return {}, next_gw
    
    fixtures_data = res.json()
    fixture_map = {}
    
    for f in fixtures_data:
        h_team = f['team_h']
        a_team = f['team_a']
        h_fdr = f['team_h_difficulty']
        a_fdr = f['team_a_difficulty']
        
        fixture_map[h_team] = {'opponent_id': a_team, 'is_home': 1, 'fdr': h_fdr}
        fixture_map[a_team] = {'opponent_id': h_team, 'is_home': 0, 'fdr': a_fdr}
        
    return fixture_map, next_gw

def fetch_user_fpl(entry_id):
    bootstrap = fetch_fpl_bootstrap_v2()
    if not bootstrap:
        return None, "Gagal terhubung ke API FPL."
    
    current_gw = [gw['id'] for gw in bootstrap['events'] if gw['is_current'] or gw['is_next']][0]
    
    picks_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{current_gw}/picks/"
    res = requests.get(picks_url)
    if res.status_code != 200 and current_gw > 1:
        picks_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{current_gw - 1}/picks/"
        res = requests.get(picks_url)
        
    if res.status_code != 200:
        return None, "ID FPL tidak ditemukan."
        
    data = res.json()
    bank = data.get("entry_history", {}).get("bank", 0) / 10.0
    player_ids = [int(p["element"]) for p in data.get("picks", [])]
    
    return {"player_ids": player_ids, "bank": bank}, "Data skuad berhasil diimpor!"

def get_player_role(row):
    roles = []
    if row.get('penalties_order') == 1: roles.append("⚽ Penalty")
    if row.get('corners_and_indirect_freekicks_order') == 1: roles.append("🚩 Corner")
    if row.get('direct_freekicks_order') == 1: roles.append("🎯 Free-Kick")
    if row.get('element_type') == 2 and (row.get('creativity', 0) > 80 or row.get('threat', 0) > 50): roles.append("🏃 Attacking WB")
    return ", ".join(roles) if roles else "-"

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def predict_player_xp_v2(df, teams_df, fixture_map):
    df_feat = df.copy()
    
    df_feat['form'] = pd.to_numeric(df_feat['form'], errors='coerce').fillna(0)
    df_feat['ict_index'] = pd.to_numeric(df_feat['ict_index'], errors='coerce').fillna(0)
    df_feat['creativity'] = pd.to_numeric(df_feat['creativity'], errors='coerce').fillna(0)
    df_feat['threat'] = pd.to_numeric(df_feat['threat'], errors='coerce').fillna(0)
    df_feat['points_per_game'] = pd.to_numeric(df_feat['points_per_game'], errors='coerce').fillna(0)
    df_feat['selected_by_percent'] = pd.to_numeric(df_feat['selected_by_percent'], errors='coerce').fillna(0)
    df_feat['ep_next'] = pd.to_numeric(df_feat['ep_next'], errors='coerce').fillna(0)
    
    df_feat['is_penalty_taker'] = np.where(df_feat['penalties_order'] == 1, 1.0, 0.0)
    df_feat['is_corner_taker'] = np.where(df_feat['corners_and_indirect_freekicks_order'] == 1, 1.0, 0.0)
    df_feat['is_fk_taker'] = np.where(df_feat['direct_freekicks_order'] == 1, 1.0, 0.0)
    df_feat['is_attacking_defender'] = np.where(
        (df_feat['element_type'] == 2) & ((df_feat['creativity'] > 80) | (df_feat['threat'] > 50)), 
        1.0, 0.0
    )

    df_feat['opponent_team_id'] = df_feat['team'].map(lambda t: fixture_map.get(t, {}).get('opponent_id', t))
    df_feat['is_home'] = df_feat['team'].map(lambda t: fixture_map.get(t, {}).get('is_home', 1))
    df_feat['fdr'] = df_feat['team'].map(lambda t: fixture_map.get(t, {}).get('fdr', 3))

    teams_dict = {t["id"]: t["name"] for t in teams_df.to_dict('records')}
    df_feat['opponent_name'] = df_feat['opponent_team_id'].map(teams_dict).fillna("Unknown")
    df_feat['matchup_info'] = np.where(df_feat['is_home'] == 1, "vs " + df_feat['opponent_name'] + " (H)", "vs " + df_feat['opponent_name'] + " (A)")

    team_defense_stats = {}
    for team_id in teams_df['id']:
        team_players = df_feat[df_feat['team'] == team_id]
        gk_players = team_players[team_players['element_type'] == 1].sort_values(by="form", ascending=False)
        gk_form = gk_players.iloc[0]['form'] if not gk_players.empty else 2.5
        def_players = team_players[team_players['element_type'] == 2]
        avg_def_form = def_players['form'].mean() if not def_players.empty else 2.0
        team_defense_stats[team_id] = round((gk_form * 0.4) + (avg_def_form * 0.6), 2)

    avg_league_def = np.mean(list(team_defense_stats.values())) if team_defense_stats else 2.0
    
    df_feat['opp_defense_factor'] = df_feat['opponent_team_id'].map(
        lambda opp_id: round(avg_league_def / max(0.1, team_defense_stats.get(opp_id, avg_league_def)), 2)
    )

    df_feat['fdr_multiplier'] = 1.0 - ((df_feat['fdr'] - 3) * 0.12)

    feature_cols = [
        'form', 'ict_index', 'creativity', 'threat', 'points_per_game', 
        'selected_by_percent', 'ep_next', 'is_penalty_taker', 
        'is_corner_taker', 'is_fk_taker', 'is_attacking_defender',
        'opp_defense_factor', 'is_home', 'fdr_multiplier'
    ]
    X = df_feat[feature_cols]
    
    y_target = (
        df_feat['form'] * 0.20 + 
        df_feat['ict_index'] * 0.05 + 
        df_feat['points_per_game'] * 0.20 + 
        df_feat['ep_next'] * 0.20 +
        df_feat['is_penalty_taker'] * 1.5 + 
        df_feat['is_corner_taker'] * 1.0 + 
        df_feat['is_fk_taker'] * 0.5 + 
        df_feat['is_attacking_defender'] * 1.2
    ) * df_feat['opp_defense_factor'] * df_feat['fdr_multiplier'] * np.where(df_feat['is_home'] == 1, 1.10, 0.95) * np.where(df_feat['status'] == 'a', 1.0, 0.15)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y_target)
    
    df_feat['predicted_xP'] = np.round(rf.predict(X), 2)
    df_feat['special_roles'] = df_feat.apply(get_player_role, axis=1)
    
    return df_feat

# -----------------------------------------------------------------------------
# 3. REALISTIC MILP SOLVER
# -----------------------------------------------------------------------------
def run_realistic_milp(df, current_ids, bank, free_transfers, use_chip_wildcard=False):
    current_ids = [int(x) for x in current_ids]
    players = df.copy()
    players['id'] = players['id'].astype(int)
    all_pids = players['id'].tolist()
    
    current_squad_df = players[players['id'].isin(current_ids)]
    current_cost = (current_squad_df['now_cost'] / 10.0).sum()
    total_budget = current_cost + bank

    prob = pulp.LpProblem("FPL_Realistic_Optimization", pulp.LpMaximize)

    squad_vars = pulp.LpVariable.dicts("Squad", all_pids, cat='Binary')
    start_vars = pulp.LpVariable.dicts("Start", all_pids, cat='Binary')
    cap_vars = pulp.LpVariable.dicts("Cap", all_pids, cat='Binary')
    retain_vars = pulp.LpVariable.dicts("Retain", current_ids, cat='Binary')

    prob += pulp.lpSum([squad_vars[i] for i in all_pids]) == 15
    prob += pulp.lpSum([start_vars[i] for i in all_pids]) == 11
    prob += pulp.lpSum([cap_vars[i] for i in all_pids]) == 1

    for i in all_pids:
        prob += start_vars[i] <= squad_vars[i]
        prob += cap_vars[i] <= start_vars[i]

    prob += pulp.lpSum([(players.loc[players['id'] == i, 'now_cost'].values[0] / 10.0) * squad_vars[i] for i in all_pids]) <= total_budget

    gkp_ids = players[players['element_type'] == 1]['id'].tolist()
    prob += pulp.lpSum([start_vars[i] for i in gkp_ids]) == 1
    prob += pulp.lpSum([squad_vars[i] for i in gkp_ids]) == 2

    def_ids = players[players['element_type'] == 2]['id'].tolist()
    prob += pulp.lpSum([start_vars[i] for i in def_ids]) >= 3
    prob += pulp.lpSum([start_vars[i] for i in def_ids]) <= 5
    prob += pulp.lpSum([squad_vars[i] for i in def_ids]) == 5

    mid_ids = players[players['element_type'] == 3]['id'].tolist()
    prob += pulp.lpSum([start_vars[i] for i in mid_ids]) >= 2
    prob += pulp.lpSum([start_vars[i] for i in mid_ids]) <= 5
    prob += pulp.lpSum([squad_vars[i] for i in mid_ids]) == 5

    fwd_ids = players[players['element_type'] == 4]['id'].tolist()
    prob += pulp.lpSum([start_vars[i] for i in fwd_ids]) >= 1
    prob += pulp.lpSum([start_vars[i] for i in fwd_ids]) <= 3
    prob += pulp.lpSum([squad_vars[i] for i in fwd_ids]) == 3

    for team_id in players['team'].unique():
        team_pids = players[players['team'] == team_id]['id'].tolist()
        prob += pulp.lpSum([squad_vars[i] for i in team_pids]) <= 3

    for i in current_ids:
        prob += retain_vars[i] <= squad_vars[i]

    transfers_made = 15 - pulp.lpSum([retain_vars[i] for i in current_ids])
    extra_transfers = pulp.LpVariable("ExtraTransfers", lowBound=0, cat='Integer')
    prob += extra_transfers >= transfers_made - free_transfers

    if not use_chip_wildcard:
        prob += extra_transfers <= 2

    hit_penalty_cost = 0.0 if use_chip_wildcard else 4.0

    starting_xp = pulp.lpSum([players.loc[players['id'] == i, 'predicted_xP'].values[0] * start_vars[i] for i in all_pids])
    captain_xp = pulp.lpSum([players.loc[players['id'] == i, 'predicted_xP'].values[0] * cap_vars[i] for i in all_pids])
    bench_xp = pulp.lpSum([players.loc[players['id'] == i, 'predicted_xP'].values[0] * (squad_vars[i] - start_vars[i]) for i in all_pids])

    prob += starting_xp + captain_xp + (0.10 * bench_xp) - (hit_penalty_cost * extra_transfers)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    best_squad_ids = [i for i in all_pids if squad_vars[i].varValue == 1]
    starting_ids = [i for i in all_pids if start_vars[i].varValue == 1]
    captain_id = [i for i in all_pids if cap_vars[i].varValue == 1][0]

    return best_squad_ids, starting_ids, captain_id

# -----------------------------------------------------------------------------
# 4. STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.sidebar.header("📥 Import Data FPL")
fpl_id = st.sidebar.text_input("Entry ID FPL:", value="", placeholder="Contoh: 123456")

if "user_ids" not in st.session_state: st.session_state["user_ids"] = []
if "bank" not in st.session_state: st.session_state["bank"] = 0.5

bootstrap = fetch_fpl_bootstrap_v2()
fixture_map, next_gw_id = fetch_next_fixtures_v2()

if bootstrap:
    elements = pd.DataFrame(bootstrap["elements"])
    elements['id'] = elements['id'].astype(int)
    teams_df = pd.DataFrame(bootstrap["teams"])
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements["team_name"] = elements["team"].map(teams)
    elements["display_name"] = elements["web_name"] + " (" + elements["team_name"] + ") - #" + elements["id"].astype(str)
    
    # Run ML Prediction
    elements = predict_player_xp_v2(elements, teams_df, fixture_map)
    
    if st.sidebar.button("Import Skuad FPL") and fpl_id:
        u_data, msg = fetch_user_fpl(fpl_id)
        if u_data:
            st.session_state["user_ids"] = u_data["player_ids"]
            st.session_state["bank"] = u_data["bank"]
            imported_labels = elements[elements["id"].isin(u_data["player_ids"])]["display_name"].tolist()
            st.session_state["selected_squad_key"] = imported_labels
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    bank_money = st.sidebar.number_input("Budget Sisa di Bank (£m):", min_value=0.0, max_value=20.0, value=float(st.session_state["bank"]), step=0.1)
    free_transfers = st.sidebar.number_input("Free Transfer Tersedia:", min_value=1, max_value=5, value=1)
    
    st.sidebar.markdown("---")
    use_wildcard = st.sidebar.checkbox("🚀 Aktifkan Wildcard / Free Hit Pekan Ini")

    st.subheader(f"📋 Skuad Terdaftar (15 Pemain) — Prediction Target: Gameweek {next_gw_id}")
    
    if "selected_squad_key" not in st.session_state:
        st.session_state["selected_squad_key"] = elements[elements["id"].isin(st.session_state["user_ids"])]["display_name"].tolist()
        
    selected_labels = st.multiselect(
        "Daftar Pemain Anda Saat Ini:", 
        options=elements["display_name"].tolist(), 
        key="selected_squad_key"
    )
    
    current_df = elements[elements["display_name"].isin(selected_labels)].copy()

    if not current_df.empty:
        # Tampilan Tabel yang Aman dari KeyError
        display_cols = ["web_name", "team_name", "matchup_info", "fdr", "special_roles", "status", "predicted_xP"]
        safe_cols = [c for c in display_cols if c in current_df.columns]
        
        st.dataframe(
            current_df[safe_cols].assign(Harga=lambda x: x["now_cost"]/10.0 if "now_cost" in x else 0),
            use_container_width=True
        )

        if st.button("🚀 JALANKAN OPTIMASI REALISTIS"):
            current_ids = [int(x) for x in current_df["id"].tolist()]
            
            if len(current_ids) != 15:
                st.error(f"⚠️ Skuad Anda terdeteksi {len(current_ids)} pemain. Pilih tepat 15 pemain di dropdown atas untuk menjalankan optimasi!")
            else:
                with st.spinner("Memproses Optimasi MILP Realistis..."):
                    best_squad_ids, starting_ids, captain_id = run_realistic_milp(
                        elements, current_ids, bank_money, free_transfers, use_chip_wildcard=use_wildcard
                    )
                    
                    final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                    starting_xi = final_squad_df[final_squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
                    bench = final_squad_df[~final_squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
                    
                st.divider()

                transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
                transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
                
                t_out_df = elements[elements['id'].isin(transfers_out_ids)]
                t_in_df = elements[elements['id'].isin(transfers_in_ids)]

                # -------------------------------------------------------------
                # OUTPUT 1: REKOMENDASI TRANSFER
                # -------------------------------------------------------------
                st.subheader("1. 🔄 Hasil Rekomendasi Transfer")
                num_transfers = len(transfers_in_ids)
                extra_transfers = max(0, num_transfers - free_transfers)
                hit_cost = 0 if use_wildcard else (extra_transfers * 4)
                
                if use_wildcard:
                    st.info("💡 **Status Chip:** WILDCARD / FREE HIT AKTIF (0 Penalti Hit untuk seluruh transfer).")
                else:
                    st.info("💡 **Status Chip:** Transfer Normal.")

                if num_transfers == 0:
                    st.success("✅ **0 TRANSFER**. Skuad eksisting Anda sudah berada pada posisi poin paling maksimal.")
                else:
                    st.success(f"✅ Disarankan melakukan **{num_transfers} Transfer** (Penalti Hit: -{hit_cost} Pts):")
                    
                    out_show_cols = [c for c in ["web_name", "team_name", "matchup_info", "special_roles", "predicted_xP"] if c in t_out_df.columns]
                    in_show_cols = [c for c in ["web_name", "team_name", "matchup_info", "special_roles", "predicted_xP"] if c in t_in_df.columns]
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("🔴 **Transfer Out (Pemain Keluar):**")
                        st.dataframe(t_out_df[out_show_cols].assign(Harga=lambda x: x["now_cost"]/10.0 if "now_cost" in x else 0), use_container_width=True)
                    with col_t2:
                        st.markdown("🟢 **Transfer In (Pemain Masuk):**")
                        st.dataframe(t_in_df[in_show_cols].assign(Harga=lambda x: x["now_cost"]/10.0 if "now_cost" in x else 0), use_container_width=True)

                # -------------------------------------------------------------
                # OUTPUT 2: STARTING LINEUP & KAPTEN
                # -------------------------------------------------------------
                st.subheader("2. 🏆 Starting Lineup & Pemilihan Kapten")
                captain = elements[elements['id'] == captain_id].iloc[0]
                
                vc_candidates = starting_xi[starting_xi['id'] != captain_id]
                vice_captain = vc_candidates.iloc[0] if not vc_candidates.empty else captain
                
                net_expected_pts = starting_xi['predicted_xP'].sum() + captain['predicted_xP'] - hit_cost
                
                col_cap1, col_cap2, col_cap3 = st.columns(3)
                with col_cap1:
                    st.metric("👑 CAPTAIN", f"{captain['web_name']}", f"{captain['predicted_xP'] * 2} xP")
                with col_cap2:
                    st.metric("🎖️ VICE-CAPTAIN", f"{vice_captain['web_name']}", f"{vice_captain['predicted_xP']} xP")
                with col_cap3:
                    st.metric("📊 NET PROYEKSI POIN STARTING XI", f"{round(net_expected_pts, 2)} Pts")

                st.markdown("---")
                st.markdown("### 🟢 Starting Eleven (11 Pemain Utama)")
                s11_show = [c for c in ["web_name", "team_name", "matchup_info", "fdr", "element_type", "special_roles", "status", "predicted_xP"] if c in starting_xi.columns]
                st.dataframe(
                    starting_xi[s11_show].rename(columns={"matchup_info": "Lawan GW Ini", "fdr": "FDR", "special_roles": "Peran Khusus"}), 
                    use_container_width=True
                )

                st.markdown("### 🪑 Bench (4 Pemain Cadangan)")
                bench_show = [c for c in ["web_name", "team_name", "matchup_info", "fdr", "element_type", "special_roles", "status", "predicted_xP"] if c in bench.columns]
                st.dataframe(
                    bench[bench_show].rename(columns={"matchup_info": "Lawan GW Ini", "fdr": "FDR", "special_roles": "Peran Khusus"}), 
                    use_container_width=True
                )
else:
    st.error("Gagal terhubung ke API FPL.")
