import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Genetic Algorithm Optimizer", layout="wide")
st.title("⚽ FPL Pure Genetic Algorithm & ML Optimizer")

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
# 3. GENETIC ALGORITHM OPTIMIZER (DYNAMIC TRANSFERS & SQUAD SELECTION)
# -----------------------------------------------------------------------------
def run_genetic_algorithm(df, current_ids, bank, free_transfers, pop_size=80, generations=50):
    all_ids = df['id'].tolist()
    id_to_price = dict(zip(df['id'], df['now_cost'] / 10.0))
    id_to_xp = dict(zip(df['id'], df['predicted_xP']))
    id_to_pos = dict(zip(df['id'], df['element_type'])) # 1: GKP, 2: DEF, 3: MID, 4: FWD
    id_to_team = dict(zip(df['id'], df['team']))

    current_cost = sum(id_to_price[i] for i in current_ids)
    max_budget = current_cost + bank

    # Fungsi Cek Validitas Skuad FPL (15 Pemain)
    def is_valid_squad(squad_ids):
        if len(squad_ids) != 15: return False
        
        # 1. Budget Constraint
        if sum(id_to_price[i] for i in squad_ids) > max_budget: return False
        
        # 2. Position Constraint (2 GKP, 5 DEF, 5 MID, 3 FWD)
        pos_counts = {1:0, 2:0, 3:0, 4:0}
        for i in squad_ids:
            pos_counts[id_to_pos[i]] += 1
        if pos_counts != {1:2, 2:5, 3:5, 4:3}: return False
        
        # 3. Max 3 Players per Team
        team_counts = {}
        for i in squad_ids:
            t = id_to_team[i]
            team_counts[t] = team_counts.get(t, 0) + 1
            if team_counts[t] > 3: return False
            
        return True

    # Fitness Function (Total xP dikurangi Penalti Hit Transfer -4 pt per Extra Transfer)
    def calculate_fitness(squad_ids):
        transfers_made = len(set(squad_ids) - set(current_ids))
        extra_transfers = max(0, transfers_made - free_transfers)
        hit_penalty = extra_transfers * 4.0
        
        total_xp = sum(id_to_xp[i] for i in squad_ids)
        return total_xp - hit_penalty

    # --- INISIALISASI POPULASI ---
    population = []
    if is_valid_squad(current_ids):
        population.append(current_ids)

    # Generate Variasi Skuad Awal (Termasuk 0, 1, 2, 3+ Transfer)
    attempts = 0
    while len(population) < pop_size and attempts < 2000:
        attempts += 1
        num_swaps = random.randint(0, 4) # Menguji variasi 0 hingga 4 transfer
        candidate = list(current_ids)
        
        for _ in range(num_swaps):
            if candidate:
                idx_remove = random.randint(0, len(candidate) - 1)
                candidate.pop(idx_remove)
                
            new_pick = random.choice(all_ids)
            if new_pick not in candidate:
                candidate.append(new_pick)
                
        if is_valid_squad(candidate):
            population.append(candidate)

    if not population:
        population = [current_ids]

    # --- EVOLUSI (GENETIC ALGORITHM ITERATION) ---
    for _ in range(generations):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        survivors = population[:pop_size // 2]
        
        children = []
        while len(survivors) + len(children) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            
            # Crossover (Pindah Silang)
            split = random.randint(1, 14)
            child = list(set(p1[:split] + p2[split:]))
            
            # Perbaiki jika panjang gen berkurang akibat deduplikasi set
            missing = [i for i in all_ids if i not in child]
            random.shuffle(missing)
            while len(child) < 15 and missing:
                child.append(missing.pop())
                
            # Mutasi
            if random.random() < 0.35:
                m_idx = random.randint(0, 14)
                rand_p = random.choice(all_ids)
                if rand_p not in child:
                    child[m_idx] = rand_p
                    
            if is_valid_squad(child):
                children.append(child)
                
        population = survivors + children

    best_squad = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)[0]
    return best_squad

def select_starting_xi(squad_df):
    """Memilih 11 Pemain Utama dengan Formasi Valid FPL (1 GKP, Min 3 DEF, Min 2 MID, Min 1 FWD)"""
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

        if st.button("🧬 JALANKAN OPTIMASI GENETIC ALGORITHM & ML"):
            current_ids = current_df["id"].tolist()
            
            with st.spinner("Genetic Algorithm sedang mensimulasikan evolusi kombinasi transfer & menentukan jumlah transfer paling optimal..."):
                best_squad_ids = run_genetic_algorithm(elements, current_ids, bank_money, free_transfers)
                final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                starting_xi, bench = select_starting_xi(final_squad_df)
                
            st.divider()

            # --- IDENTIFIKASI PERUBAHAN TRANSFER DARI GA ---
            transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
            transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
            
            t_out_df = elements[elements['id'].isin(transfers_out_ids)]
            t_in_df = elements[elements['id'].isin(transfers_in_ids)]

            # -----------------------------------------------------------------
            # OUTPUT 1: REKOMENDASI TRANSFER (GA) & CHIP
            # -----------------------------------------------------------------
            st.subheader("1. 🔄 Hasil Rekomendasi Transfer (Optimasi Genetic Algorithm)")
            
            num_transfers = len(transfers_in_ids)
            extra_transfers = max(0, num_transfers - free_transfers)
            hit_cost = extra_transfers * 4
            
            # Evaluasi Chip
            injured_count = len(current_df[current_df['status'] != 'a'])
            chip_msg = "Saran Chip: Tidak perlu menggunakan Chip pekan ini."
            
            if num_transfers >= 4 and "Wildcard" in chips_available:
                chip_msg = "⚠️ **Saran Chip:** Sangat disarankan mengaktifkan **WILDCARD**! Jumlah pergantian optimal dari GA terlalu banyak untuk transfer biasa."
            elif injured_count >= 3 and "Free Hit" in chips_available:
                chip_msg = "💡 **Saran Chip:** Pertimbangkan **FREE HIT** untuk menghindari pengurangan poin berlebih akibat pemain cedera."
                
            st.info(f"**Strategi Chip:** {chip_msg}")

            if num_transfers == 0:
                st.success("✅ **Saran Transfer dari GA:** **TIDAK ADA TRANSFER (0 Transfer)**. Kombinasi tim eksisting Anda sudah merupakan titik puncak poin optimal pekan ini.")
            else:
                st.success(f"✅ **Saran Transfer dari GA:** Genetic Algorithm menemukan bahwa **{num_transfers} Transfer** adalah opsi paling optimal (Penalti Hit: -{hit_cost} Pts).")
                
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
            
            # Net Expected Points = Total xP Starting XI + (2x xP Captain) - Hit Penalty Transfer
            raw_total_pts = starting_xi['predicted_xP'].sum() + captain['predicted_xP']
            net_expected_pts = raw_total_pts - hit_cost
            
            col_cap1, col_cap2, col_cap3 = st.columns(3)
            with col_cap1:
                st.metric("👑 CAPTAIN", f"{captain['web_name']}", f"{captain['predicted_xP'] * 2} xP")
            with col_cap2:
                st.metric("🎖️ VICE-CAPTAIN", f"{vice_captain['web_name']}", f"{vice_captain['predicted_xP']} xP")
            with col_cap3:
                st.metric("📊 NET PROYEKSI POIN (Setlah Hit)", f"{round(net_expected_pts, 2)} Pts")

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
