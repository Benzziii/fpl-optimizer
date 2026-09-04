import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Fast MILP Optimizer", layout="wide")
st.title("⚡ FPL Ultra-Fast Optimizer (MILP + Machine Learning)")

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
    player_ids = [p["element"] for p in data.get("picks", [])]
    
    return {"player_ids": player_ids, "bank": bank}, "Data skuad berhasil diimpor!"

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE (PREDICT XP VIA RANDOM FOREST)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def predict_player_xp(df):
    df_feat = df.copy()
    
    df_feat['form'] = pd.to_numeric(df_feat['form'], errors='coerce').fillna(0)
    df_feat['ict_index'] = pd.to_numeric(df_feat['ict_index'], errors='coerce').fillna(0)
    df_feat['points_per_game'] = pd.to_numeric(df_feat['points_per_game'], errors='coerce').fillna(0)
    df_feat['selected_by_percent'] = pd.to_numeric(df_feat['selected_by_percent'], errors='coerce').fillna(0)
    df_feat['ep_next'] = pd.to_numeric(df_feat['ep_next'], errors='coerce').fillna(0)
    
    X = df_feat[['form', 'ict_index', 'points_per_game', 'selected_by_percent', 'ep_next']]
    
    y_target = (
        df_feat['form'] * 0.35 + 
        df_feat['ict_index'] * 0.05 + 
        df_feat['points_per_game'] * 0.35 + 
        df_feat['ep_next'] * 0.25
    ) * np.where(df_feat['status'] == 'a', 1.0, 0.15)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y_target)
    
    df['predicted_xP'] = np.round(rf.predict(X), 2)
    return df

# -----------------------------------------------------------------------------
# 3. MILP OPTIMIZER ENGINE (PULP)
# -----------------------------------------------------------------------------
def solve_fpl_milp(df, current_ids, bank, free_transfers):
    players = df.copy()
    player_ids = players['id'].tolist()
    
    # Menghitung budget total yang diizinkan
    current_cost = (players[players['id'].isin(current_ids)]['now_cost'] / 10.0).sum()
    budget = current_cost + bank
    
    # Inisialisasi Problem MILP
    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
    
    # Variabel Keputusan (1 jika dipilih masuk skuad 15 pemain, 0 jika tidak)
    squad_vars = pulp.LpVariable.dicts("Squad", player_ids, cat='Binary')
    
    # Variabel Keputusan (1 jika pemain di-retain dari skuad lama)
    retain_vars = pulp.LpVariable.dicts("Retain", current_ids, cat='Binary')
    
    # 1. Batasan Jumlah Skuad = 15 Pemain
    prob += pulp.lpSum([squad_vars[i] for i in player_ids]) == 15
    
    # 2. Batasan Budget
    prob += pulp.lpSum([(players.loc[players['id'] == i, 'now_cost'].values[0] / 10.0) * squad_vars[i] for i in player_ids]) <= budget
    
    # 3. Batasan Posisi Skuad (2 GKP, 5 DEF, 5 MID, 3 FWD)
    for pos_code, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pos_ids = players[players['element_type'] == pos_code]['id'].tolist()
        prob += pulp.lpSum([squad_vars[i] for i in pos_ids]) == count
        
    # 4. Batasan Maksimal 3 Pemain Per Klub
    for team_id in players['team'].unique():
        team_pids = players[players['team'] == team_id]['id'].tolist()
        prob += pulp.lpSum([squad_vars[i] for i in team_pids]) <= 3
        
    # 5. Hubungan Retain dengan Squad
    for i in current_ids:
        prob += retain_vars[i] <= squad_vars[i]
        
    # Hitung Jumlah Transfer
    transfers_made = 15 - pulp.lpSum([retain_vars[i] for i in current_ids])
    
    # Objective Function: Memaksimalkan Total xP Dikurangi Penalti Hit Transfer (-4 pt per ekstra transfer)
    total_xp = pulp.lpSum([players.loc[players['id'] == i, 'predicted_xP'].values[0] * squad_vars[i] for i in player_ids])
    
    # Pendekatan Linear untuk Penalti Hit
    prob += total_xp - (4.0 * (15 - pulp.lpSum([retain_vars[i] for i in current_ids]) - free_transfers))

    # Solve Optimization Problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Ambil Skuad Terpilih
    selected_squad_ids = [i for i in player_ids if squad_vars[i].varValue == 1]
    
    return selected_squad_ids

def select_starting_xi(squad_df):
    """Memilih 11 Pemain Utama & Formasi Valid FPL"""
    gkps = squad_df[squad_df['element_type'] == 1].sort_values(by="predicted_xP", ascending=False)
    defs = squad_df[squad_df['element_type'] == 2].sort_values(by="predicted_xP", ascending=False)
    mids = squad_df[squad_df['element_type'] == 3].sort_values(by="predicted_xP", ascending=False)
    fwds = squad_df[squad_df['element_type'] == 4].sort_values(by="predicted_xP", ascending=False)
    
    starting_ids = []
    starting_ids.append(gkps.iloc[0]['id']) # 1 Kiper
    starting_ids.extend(defs.iloc[:3]['id'].tolist()) # 3 Bek
    starting_ids.extend(mids.iloc[:2]['id'].tolist()) # 2 Gelandang
    starting_ids.extend(fwds.iloc[:1]['id'].tolist()) # 1 Penyerang
    
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
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements["team_name"] = elements["team"].map(teams)
    
    # ML Prediction
    elements = predict_player_xp(elements)
    
    if st.sidebar.button("Import Skuad FPL") and fpl_id:
        u_data, msg = fetch_user_fpl(fpl_id)
        if u_data:
            st.session_state["user_ids"] = u_data["player_ids"]
            st.session_state["bank"] = u_data["bank"]
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    bank_money = st.sidebar.number_input("Budget Sisa di Bank (£m):", min_value=0.0, max_value=20.0, value=float(st.session_state["bank"]), step=0.1)
    free_transfers = st.sidebar.number_input("Free Transfer Tersedia:", min_value=1, max_value=5, value=1)
    chips_available = st.sidebar.multiselect("Chip Tersedia:", ["Wildcard", "Free Hit"], default=["Wildcard", "Free Hit"])

    st.subheader("📋 Skuad Terdaftar (15 Pemain)")
    default_selected = elements[elements["id"].isin(st.session_state["user_ids"])]["web_name"].tolist()
    
    selected_names = st.multiselect("Daftar Pemain Anda Saat Ini:", options=elements["web_name"].tolist(), default=default_selected)
    current_df = elements[elements["web_name"].isin(selected_names)].copy()

    if not current_df.empty:
        st.dataframe(
            current_df[["web_name", "team_name", "element_type", "now_cost", "status", "form", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0),
            use_container_width=True
        )

        if st.button("🚀 JALANKAN OPTIMASI MILP & ML (SUPER FAST)"):
            current_ids = current_df["id"].tolist()
            
            with st.spinner("Memproses Optimasi Matematika MILP..."):
                best_squad_ids = solve_fpl_milp(elements, current_ids, bank_money, free_transfers)
                final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                starting_xi, bench = select_starting_xi(final_squad_df)
                
            st.divider()

            # --- IDENTIFIKASI PERUBAHAN TRANSFER DARI MILP ---
            transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
            transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
            
            t_out_df = elements[elements['id'].isin(transfers_out_ids)]
            t_in_df = elements[elements['id'].isin(transfers_in_ids)]

            # -----------------------------------------------------------------
            # OUTPUT 1: REKOMENDASI TRANSFER (MILP) & CHIP
            # -----------------------------------------------------------------
            st.subheader("1. 🔄 Hasil Rekomendasi Transfer Optimal")
            
            num_transfers = len(transfers_in_ids)
            extra_transfers = max(0, num_transfers - free_transfers)
            hit_cost = extra_transfers * 4
            
            # Evaluasi Chip
            injured_count = len(current_df[current_df['status'] != 'a'])
            chip_msg = "Saran Chip: Tidak perlu menggunakan Chip pekan ini."
            
            if num_transfers >= 4 and "Wildcard" in chips_available:
                chip_msg = "⚠️ **Saran Chip:** Sangat disarankan mengaktifkan **WILDCARD**! Kombinasi optimal memerlukan banyak transfer."
            elif injured_count >= 3 and "Free Hit" in chips_available:
                chip_msg = "💡 **Saran Chip:** Pertimbangkan **FREE HIT** untuk menghindari pengurangan poin berlebih."
                
            st.info(f"**Strategi Chip:** {chip_msg}")

            if num_transfers == 0:
                st.success("✅ **Saran Transfer:** **0 TRANSFER**. Skuad eksisting Anda sudah berada pada posisi poin paling maksimal.")
            else:
                st.success(f"✅ **Saran Transfer:** Lakukan **{num_transfers} Transfer** berikut (Penalti Hit: -{hit_cost} Pts):")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("🔴 **Pemain Keluar (Transfer Out):**")
                    st.dataframe(t_out_df[["web_name", "team_name", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)
                with col_t2:
                    st.markdown("🟢 **Pemain Masuk (Transfer In):**")
                    st.dataframe(t_in_df[["web_name", "team_name", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)

            # -----------------------------------------------------------------
            # OUTPUT 2: STARTING LINEUP, CAPTAIN & EXPECTED POINTS
            # -----------------------------------------------------------------
            st.subheader("2. 🏆 Starting Lineup & Pemilihan Kapten (Next Week)")
            
            captain = starting_xi.iloc[0]
            vice_captain = starting_xi.iloc[1]
            
            raw_total_pts = starting_xi['predicted_xP'].sum() + captain['predicted_xP']
            net_expected_pts = raw_total_pts - hit_cost
            
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
                starting_xi[["web_name", "team_name", "element_type", "form", "status", "predicted_xP"]]
                .rename(columns={
                    "web_name": "Nama Pemain", 
                    "team_name": "Klub", 
                    "element_type": "Posisi (1:GKP, 2:DEF, 3:MID, 4:FWD)",
                    "predicted_xP": "Expected Points (xP)"
                }),
                use_container_width=True
            )

            st.markdown("### 🪑 Bench (4 Pemain Cadangan)")
            st.dataframe(
                bench[["web_name", "team_name", "element_type", "form", "status", "predicted_xP"]]
                .rename(columns={
                    "web_name": "Nama Pemain", 
                    "team_name": "Klub", 
                    "element_type": "Posisi",
                    "predicted_xP": "Expected Points (xP)"
                }),
                use_container_width=True
            )
else:
    st.error("Gagal terhubung ke API Fantasy Premier League.")
