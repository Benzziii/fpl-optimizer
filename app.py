import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Exact MILP Optimizer", layout="wide")
st.title("⚡ FPL Ultra-Fast MILP & ML Optimizer")

# -----------------------------------------------------------------------------
# 1. FETCH DATA FROM FPL API
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_fpl_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else None

def fetch_user_fpl(entry_id):
    bootstrap = fetch_fpl_bootstrap()
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
def predict_player_xp(df, teams_df):
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

    team_defense_stats = {}
    for team_id in teams_df['id']:
        team_players = df_feat[df_feat['team'] == team_id]
        gk_players = team_players[team_players['element_type'] == 1].sort_values(by="form", ascending=False)
        gk_form = gk_players.iloc[0]['form'] if not gk_players.empty else 2.5
        def_players = team_players[team_players['element_type'] == 2]
        avg_def_form = def_players['form'].mean() if not def_players.empty else 2.0
        team_defense_stats[team_id] = round((gk_form * 0.4) + (avg_def_form * 0.6), 2)

    avg_league_def = np.mean(list(team_defense_stats.values())) if team_defense_stats else 2.0
    
    df_feat['opp_defense_factor'] = df_feat['team'].map(
        lambda t_id: round(avg_league_def / max(0.1, team_defense_stats.get(t_id, avg_league_def)), 2)
    )

    feature_cols = [
        'form', 'ict_index', 'creativity', 'threat', 'points_per_game', 
        'selected_by_percent', 'ep_next', 'is_penalty_taker', 
        'is_corner_taker', 'is_fk_taker', 'is_attacking_defender',
        'opp_defense_factor'
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
    ) * df_feat['opp_defense_factor'] * np.where(df_feat['status'] == 'a', 1.0, 0.15)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y_target)
    
    df_feat['predicted_xP'] = np.round(rf.predict(X), 2)
    df_feat['special_roles'] = df_feat.apply(get_player_role, axis=1)
    
    return df_feat

# -----------------------------------------------------------------------------
# 3. MILP OPTIMIZER ENGINE
# -----------------------------------------------------------------------------
def run_milp_optimization(df, current_ids, bank, free_transfers):
    current_ids = [int(x) for x in current_ids]
    players = df.copy()
    players['id'] = players['id'].astype(int)
    all_pids = players['id'].tolist()
    
    current_squad_df = players[players['id'].isin(current_ids)]
    current_cost = (current_squad_df['now_cost'] / 10.0).sum()
    total_budget = current_cost + bank

    prob = pulp.LpProblem("FPL_Optimal_Transfer", pulp.LpMaximize)

    squad_vars = pulp.LpVariable.dicts("Squad", all_pids, cat='Binary')
    retain_vars = pulp.LpVariable.dicts("Retain", current_ids, cat='Binary')

    prob += pulp.lpSum([squad_vars[i] for i in all_pids]) == 15
    prob += pulp.lpSum([(players.loc[players['id'] == i, 'now_cost'].values[0] / 10.0) * squad_vars[i] for i in all_pids]) <= total_budget

    for pos_code, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pos_ids = players[players['element_type'] == pos_code]['id'].tolist()
        prob += pulp.lpSum([squad_vars[i] for i in pos_ids]) == count

    for team_id in players['team'].unique():
        team_pids = players[players['team'] == team_id]['id'].tolist()
        prob += pulp.lpSum([squad_vars[i] for i in team_pids]) <= 3

    for i in current_ids:
        prob += retain_vars[i] <= squad_vars[i]

    transfers_made = 15 - pulp.lpSum([retain_vars[i] for i in current_ids])
    extra_transfers = pulp.LpVariable("ExtraTransfers", lowBound=0, cat='Integer')
    prob += extra_transfers >= transfers_made - free_transfers

    total_xp = pulp.lpSum([players.loc[players['id'] == i, 'predicted_xP'].values[0] * squad_vars[i] for i in all_pids])
    prob += total_xp - (4.0 * extra_transfers)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    best_squad_ids = [i for i in all_pids if squad_vars[i].varValue == 1]
    return best_squad_ids

def select_starting_xi(squad_df):
    squad_df = squad_df.copy()
    gkps = squad_df[squad_df['element_type'] == 1].sort_values(by="predicted_xP", ascending=False)
    defs = squad_df[squad_df['element_type'] == 2].sort_values(by="predicted_xP", ascending=False)
    mids = squad_df[squad_df['element_type'] == 3].sort_values(by="predicted_xP", ascending=False)
    fwds = squad_df[squad_df['element_type'] == 4].sort_values(by="predicted_xP", ascending=False)
    
    starting_ids = []
    starting_ids.append(gkps.iloc[0]['id'])
    starting_ids.extend(defs.iloc[:3]['id'].tolist())
    starting_ids.extend(mids.iloc[:2]['id'].tolist())
    starting_ids.extend(fwds.iloc[:1]['id'].tolist())
    
    remaining_outfield = squad_df[
        (~squad_df['id'].isin(starting_ids)) & 
        (squad_df['element_type'] != 1)
    ].sort_values(by="predicted_xP", ascending=False)
    
    starting_ids.extend(remaining_outfield.iloc[:4]['id'].tolist())
    
    starting_xi = squad_df[squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
    bench = squad_df[~squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
    
    return starting_xi, bench

# -----------------------------------------------------------------------------
# 4. STREAMLIT INTERFACE
# -----------------------------------------------------------------------------
st.sidebar.header("📥 Import Data FPL")
fpl_id = st.sidebar.text_input("Entry ID FPL:", value="", placeholder="Contoh: 123456")

if "user_ids" not in st.session_state: st.session_state["user_ids"] = []
if "bank" not in st.session_state: st.session_state["bank"] = 0.5

bootstrap = fetch_fpl_bootstrap()

if bootstrap:
    elements = pd.DataFrame(bootstrap["elements"])
    elements['id'] = elements['id'].astype(int)
    teams_df = pd.DataFrame(bootstrap["teams"])
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements["team_name"] = elements["team"].map(teams)
    
    # Buat label unik untuk pilihan pemain
    elements["display_name"] = elements["web_name"] + " (" + elements["team_name"] + ") - #" + elements["id"].astype(str)
    
    elements = predict_player_xp(elements, teams_df)
    
    if st.sidebar.button("Import Skuad FPL") and fpl_id:
        u_data, msg = fetch_user_fpl(fpl_id)
        if u_data:
            st.session_state["user_ids"] = u_data["player_ids"]
            st.session_state["bank"] = u_data["bank"]
            
            # Paksa update widget multiselect
            imported_labels = elements[elements["id"].isin(u_data["player_ids"])]["display_name"].tolist()
            st.session_state["selected_squad_key"] = imported_labels
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    bank_money = st.sidebar.number_input("Budget Sisa di Bank (£m):", min_value=0.0, max_value=20.0, value=float(st.session_state["bank"]), step=0.1)
    free_transfers = st.sidebar.number_input("Free Transfer Tersedia:", min_value=1, max_value=5, value=1)
    chips_available = st.sidebar.multiselect("Chip Tersedia:", ["Wildcard", "Free Hit"], default=["Wildcard", "Free Hit"])

    st.subheader("📋 Skuad Terdaftar (15 Pemain)")
    
    # Inisialisasi default label jika belum ada di session state
    if "selected_squad_key" not in st.session_state:
        st.session_state["selected_squad_key"] = elements[elements["id"].isin(st.session_state["user_ids"])]["display_name"].tolist()
        
    selected_labels = st.multiselect(
        "Daftar Pemain Anda Saat Ini:", 
        options=elements["display_name"].tolist(), 
        key="selected_squad_key"
    )
    
    current_df = elements[elements["display_name"].isin(selected_labels)].copy()

    if not current_df.empty:
        st.dataframe(
            current_df[["web_name", "team_name", "element_type", "now_cost", "status", "special_roles", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0),
            use_container_width=True
        )

        if st.button("🚀 JALANKAN OPTIMASI MILP & ML"):
            current_ids = [int(x) for x in current_df["id"].tolist()]
            
            if len(current_ids) != 15:
                st.error(f"⚠️ Skuad Anda saat ini terdeteksi {len(current_ids)} pemain. Pilih tepat 15 pemain di dropdown atas untuk menjalankan optimasi!")
            else:
                with st.spinner("Memproses Solusi Matematis MILP..."):
                    best_squad_ids = run_milp_optimization(elements, current_ids, bank_money, free_transfers)
                    final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                    starting_xi, bench = select_starting_xi(final_squad_df)
                    
                st.divider()

                transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
                transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
                
                t_out_df = elements[elements['id'].isin(transfers_out_ids)]
                t_in_df = elements[elements['id'].isin(transfers_in_ids)]

                # -------------------------------------------------------------
                # OUTPUT 1: REKOMENDASI TRANSFER & CHIP
                # -------------------------------------------------------------
                st.subheader("1. 🔄 Hasil Rekomendasi Transfer")
                num_transfers = len(transfers_in_ids)
                extra_transfers = max(0, num_transfers - free_transfers)
                hit_cost = extra_transfers * 4
                
                injured_count = len(current_df[current_df['status'] != 'a'])
                chip_msg = "Tidak perlu menggunakan Chip pekan ini."
                
                if num_transfers >= 4 and "Wildcard" in chips_available:
                    chip_msg = "⚠️ Sangat disarankan mengaktifkan **WILDCARD**!"
                elif injured_count >= 3 and "Free Hit" in chips_available:
                    chip_msg = "💡 Pertimbangkan **FREE HIT**."
                    
                st.info(f"**Strategi Chip:** {chip_msg}")

                if num_transfers == 0:
                    st.success("✅ **0 TRANSFER**. Skuad eksisting Anda sudah berada pada posisi poin paling maksimal.")
                else:
                    st.success(f"✅ Disarankan melakukan **{num_transfers} Transfer** (Penalti Hit: -{hit_cost} Pts):")
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown("🔴 **Transfer Out (Pemain Keluar):**")
                        st.dataframe(t_out_df[["web_name", "team_name", "special_roles", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)
                    with col_t2:
                        st.markdown("🟢 **Transfer In (Pemain Masuk):**")
                        st.dataframe(t_in_df[["web_name", "team_name", "special_roles", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)

                # -------------------------------------------------------------
                # OUTPUT 2: STARTING LINEUP & KAPTEN
                # -------------------------------------------------------------
                st.subheader("2. 🏆 Starting Lineup & Pemilihan Kapten")
                captain = starting_xi.iloc[0]
                vice_captain = starting_xi.iloc[1]
                net_expected_pts = starting_xi['predicted_xP'].sum() + captain['predicted_xP'] - hit_cost
                
                col_cap1, col_cap2, col_cap3 = st.columns(3)
                with col_cap1:
                    st.metric("👑 CAPTAIN", f"{captain['web_name']}", f"{captain['predicted_xP'] * 2} xP")
                with col_cap2:
                    st.metric("🎖️ VICE-CAPTAIN", f"{vice_captain['web_name']}", f"{vice_captain['predicted_xP']} xP")
                with col_cap3:
                    st.metric("📊 NET PROYEKSI POIN", f"{round(net_expected_pts, 2)} Pts")

                st.markdown("---")
                st.markdown("### 🟢 Starting Eleven (11 Pemain Utama)")
                st.dataframe(
                    starting_xi[["web_name", "team_name", "element_type", "special_roles", "form", "status", "predicted_xP"]]
                    .rename(columns={"special_roles": "Peran Khusus (Set-Piece)"}), 
                    use_container_width=True
                )

                st.markdown("### 🪑 Bench (4 Pemain Cadangan)")
                st.dataframe(
                    bench[["web_name", "team_name", "element_type", "special_roles", "form", "status", "predicted_xP"]]
                    .rename(columns={"special_roles": "Peran Khusus (Set-Piece)"}), 
                    use_container_width=True
                )
else:
    st.error("Gagal terhubung ke API FPL.")
