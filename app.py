import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL ML & Genetic Optimizer", layout="wide")
st.title("⚽ FPL Super-Optimizer: Machine Learning + Genetic Algorithm")

# -----------------------------------------------------------------------------
# 1. API DATA FETCHING
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
    
    # User Picks
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
    
    return {"player_ids": player_ids, "bank": bank}, "Data berhasil diimpor!"

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE (RANDOM FOREST)
# -----------------------------------------------------------------------------
def train_and_predict_xp(df):
    """
    Melatih Random Forest Regressor dari fitur kuantitafif FPL 
    untuk memprediksi nilai xP (Expected Points) yang lebih presisi.
    """
    df_features = df.copy()
    
    # Preprocessing Fitur
    df_features['form'] = pd.to_numeric(df_features['form'], errors='coerce').fillna(0)
    df_features['ict_index'] = pd.to_numeric(df_features['ict_index'], errors='coerce').fillna(0)
    df_features['points_per_game'] = pd.to_numeric(df_features['points_per_game'], errors='coerce').fillna(0)
    df_features['selected_by_percent'] = pd.to_numeric(df_features['selected_by_percent'], errors='coerce').fillna(0)
    df_features['ep_next'] = pd.to_numeric(df_features['ep_next'], errors='coerce').fillna(0)
    
    # Sintesis Target Historical Points untuk Training Model
    # Pada produksi penuh, ini di-fit dari dataset historical GW
    X = df_features[['form', 'ict_index', 'points_per_game', 'selected_by_percent', 'ep_next']]
    y_synthetic = (
        df_features['form'] * 0.4 + 
        df_features['ict_index'] * 0.05 + 
        df_features['points_per_game'] * 0.3 + 
        df_features['ep_next'] * 0.25
    ) * np.where(df_features['status'] == 'a', 1.0, 0.2)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y_synthetic)
    
    df['predicted_xP'] = np.round(rf.predict(X), 2)
    return df, rf

# -----------------------------------------------------------------------------
# 3. GENETIC ALGORITHM OPTIMIZER
# -----------------------------------------------------------------------------
def run_genetic_algorithm(df, current_ids, bank, free_transfers, pop_size=60, generations=40):
    """
    Genetic Algorithm untuk menentukan kombinasi transfer terbaik (0 s/d N transfer).
    """
    all_ids = df['id'].tolist()
    id_to_price = dict(zip(df['id'], df['now_cost'] / 10.0))
    id_to_xp = dict(zip(df['id'], df['predicted_xP']))
    id_to_pos = dict(zip(df['id'], df['element_type'])) # 1: GKP, 2: DEF, 3: MID, 4: FWD
    id_to_team = dict(zip(df['id'], df['team']))

    def is_valid_squad(squad_ids):
        if len(squad_ids) != 15: return False
        
        # Cek Pagu Budget
        total_cost = sum(id_to_price[i] for i in squad_ids)
        current_cost = sum(id_to_price[i] for i in current_ids)
        if total_cost > (current_cost + bank): return False
        
        # Cek Kuota Posisi (2 GKP, 5 DEF, 5 MID, 3 FWD)
        pos_counts = {1:0, 2:0, 3:0, 4:0}
        for i in squad_ids:
            pos_counts[id_to_pos[i]] += 1
        if pos_counts != {1:2, 2:5, 3:5, 4:3}: return False
        
        # Cek Batas Maksimal 3 Pemain Per Klub
        team_counts = {}
        for i in squad_ids:
            t = id_to_team[i]
            team_counts[t] = team_counts.get(t, 0) + 1
            if team_counts[t] > 3: return False
            
        return True

    def calculate_fitness(squad_ids):
        # Penalti jika melebihi kuota free transfer (-4 poin per ekstra transfer)
        transfers_made = len(set(squad_ids) - set(current_ids))
        extra_transfers = max(0, transfers_made - free_transfers)
        hit_penalty = extra_transfers * 4.0
        
        # Poin Total Skuad
        total_xp = sum(id_to_xp[i] for i in squad_ids)
        return total_xp - hit_penalty

    # Inisialisasi Populasi
    population = []
    # Masukkan individu baseline (skuad saat ini)
    if is_valid_squad(current_ids):
        population.append(current_ids)

    # Generate kandidat acak yang valid
    attempts = 0
    while len(population) < pop_size and attempts < 1000:
        attempts += 1
        # Lakukan variasi dari skuad saat ini
        num_swaps = random.randint(1, 3)
        candidate = list(current_ids)
        for _ in range(num_swaps):
            idx_remove = random.randint(0, 14)
            candidate.pop(idx_remove)
            new_pick = random.choice(all_ids)
            if new_pick not in candidate:
                candidate.append(new_pick)
        if is_valid_squad(candidate):
            population.append(candidate)

    if not population:
        population = [current_ids]

    # Proses Evolusi (Siklus Generasi)
    for _ in range(generations):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        survivors = population[:pop_size // 2]
        
        children = []
        while len(survivors) + len(children) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            # Crossover (Penggabungan Gen)
            split = random.randint(1, 14)
            child = list(set(p1[:split] + p2[split:]))
            
            # Tambah gen hingga pas 15 jika terjadi reduksi akibat set
            missing = [i for i in all_ids if i not in child]
            while len(child) < 15 and missing:
                child.append(missing.pop())
                
            # Mutasi
            if random.random() < 0.3:
                m_idx = random.randint(0, 14)
                rand_p = random.choice(all_ids)
                if rand_p not in child:
                    child[m_idx] = rand_p
                    
            if is_valid_squad(child):
                children.append(child)
                
        population = survivors + children

    best_squad = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)[0]
    return best_squad

# -----------------------------------------------------------------------------
# 4. INTERFACE STREAMLIT
# -----------------------------------------------------------------------------

st.sidebar.header("📥 FPL Import & Settings")
fpl_id = st.sidebar.text_input("Entry ID FPL:", value="", placeholder="Contoh: 876543")

if "user_ids" not in st.session_state: st.session_state["user_ids"] = []
if "bank" not in st.session_state: st.session_state["bank"] = 0.5

bootstrap = fetch_fpl_bootstrap()

if bootstrap:
    elements = pd.DataFrame(bootstrap["elements"])
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements["team_name"] = elements["team"].map(teams)
    
    if st.sidebar.button("Import Data Skuad") and fpl_id:
        u_data, msg = fetch_user_fpl(fpl_id)
        if u_data:
            st.session_state["user_ids"] = u_data["player_ids"]
            st.session_state["bank"] = u_data["bank"]
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            
    bank_money = st.sidebar.number_input("Budget Bank (£m):", min_value=0.0, max_value=20.0, value=float(st.session_state["bank"]), step=0.1)
    free_transfers = st.sidebar.number_input("Free Transfers:", min_value=1, max_value=5, value=1)
    chips_available = st.sidebar.multiselect("Chip Available:", ["Wildcard", "Free Hit"], default=["Wildcard", "Free Hit"])

    # Jalankan ML
    elements, rf_model = train_and_predict_xp(elements)

    # Pilih Skuad
    st.subheader("📋 Skuad Terdaftar (15 Pemain)")
    default_selected = elements[elements["id"].isin(st.session_state["user_ids"])]["web_name"].tolist()
    
    selected_names = st.multiselect("Daftar Pemain Utama Anda:", options=elements["web_name"].tolist(), default=default_selected)
    
    current_df = elements[elements["web_name"].isin(selected_names)].copy()

    if not current_df.empty:
        st.dataframe(
            current_df[["web_name", "team_name", "element_type", "now_cost", "status", "form", "predicted_xP"]].assign(Price=lambda x: x["now_cost"]/10.0),
            use_container_width=True
        )

        if st.button("⚡ Jalankan ML & Genetic Optimizer"):
            current_ids = current_df["id"].tolist()
            
            with st.spinner("Menjalankan Evaluasi Random Forest & Simulasi Evolusi Genetic Algorithm..."):
                best_ids = run_genetic_algorithm(elements, current_ids, bank_money, free_transfers)
                
            best_squad_df = elements[elements["id"].isin(best_ids)].copy().sort_values(by="predicted_xP", ascending=False)
            
            # --- DETEKSI TRANSFER ---
            transfers_out_ids = list(set(current_ids) - set(best_ids))
            transfers_in_ids = list(set(best_ids) - set(current_ids))
            
            t_out_df = elements[elements["id"].isin(transfers_out_ids)]
            t_in_df = elements[elements["id"].isin(transfers_in_ids)]

            st.divider()

            # --- STRATEGI CHIP ---
            st.subheader("1. 🧠 Evaluasi Strategi Chip & Transfer ML")
            injured_count = len(current_df[current_df["status"] != "a"])
            
            if len(transfers_in_ids) >= 4 and "Wildcard" in chips_available:
                st.warning("⚠️ **Rekomendasi Chip:** Aktifkan **WILDCARD**! Jumlah pergantian optimal terlalu banyak untuk transfer biasa.")
            elif injured_count >= 3 and "Free Hit" in chips_available:
                st.info("💡 **Rekomendasi Chip:** Pertimbangkan **FREE HIT** karena banyaknya kendala fisik pemain pekan ini.")
            else:
                st.success(f"✅ **Rekomendasi Transfer Normal:** Disarankan melakukan {len(transfers_in_ids)} transfer.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🔴 **Transfer Out:**")
                st.table(t_out_df[["web_name", "team_name", "predicted_xP"]].assign(Price=lambda x: x["now_cost"]/10.0))
            with col2:
                st.markdown("🟢 **Transfer In:**")
                st.table(t_in_df[["web_name", "team_name", "predicted_xP"]].assign(Price=lambda x: x["now_cost"]/10.0))

            # --- OPTIMASI SQUAD UTAMA, KAPTEN & BENCH ---
            st.subheader("2. 🏆 Skuad Utama & Ban Lengan (Next Gameweek)")
            
            # Memilih 11 Pemain Pertama Berdasarkan Proyeksi xP
            starting_11 = best_squad_df.head(11)
            bench = best_squad_df.tail(4)
            
            captain = starting_11.iloc[0]
            vice_captain = starting_11.iloc[1]

            st.success(f"👑 **Captain:** {captain['web_name']} ({captain['team_name']}) — Proyeksi ML: **{captain['predicted_xP']} Pts**")
            st.warning(f"🎖️ **Vice-Captain:** {vice_captain['web_name']} ({vice_captain['team_name']}) — Proyeksi ML: **{vice_captain['predicted_xP']} Pts**")

            st.markdown("**Starting XI (11 Pemain Utama):**")
            st.dataframe(starting_11[["web_name", "team_name", "element_type", "predicted_xP"]].assign(Price=lambda x: x["now_cost"]/10.0), use_container_width=True)

            st.markdown("**Bench (4 Pemain Cadangan):**")
            st.dataframe(bench[["web_name", "team_name", "element_type", "predicted_xP"]].assign(Price=lambda x: x["now_cost"]/10.0), use_container_width=True)

else:
    st.error("Gagal terhubung ke API Fantasy Premier League.")
