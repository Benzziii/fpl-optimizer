import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import pulp
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Optimizer & ML Predictor", layout="wide")
st.title("⚽ FPL Optimizer: MILP (Pure Solver) & Genetic Algorithm")

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
# 3. OPTIMIZER ENGINES (MILP & GENETIC ALGORITHM)
# -----------------------------------------------------------------------------

# --- A. MILP SOLVER (PULP) ---
def run_realistic_milp(df, current_ids, bank, free_transfers, use_chip_wildcard=False):
    all_pids = df['id'].tolist()
    
    price_dict = dict(zip(df['id'], df['now_cost'] / 10.0))
    xp_dict = dict(zip(df['id'], df['predicted_xP']))
    pos_dict = dict(zip(df['id'], df['element_type'])) # 1: GKP, 2: DEF, 3: MID, 4: FWD
    team_dict = dict(zip(df['id'], df['team']))
    
    current_cost = sum(price_dict[i] for i in current_ids if i in price_dict)
    max_budget = current_cost + bank

    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)

    # Decision Variables
    squad_vars = pulp.LpVariable.dicts("Squad", all_pids, cat='Binary')
    start_vars = pulp.LpVariable.dicts("Start", all_pids, cat='Binary')
    cap_vars = pulp.LpVariable.dicts("Captain", all_pids, cat='Binary')

    transfer_out = pulp.LpVariable.dicts("TransferOut", all_pids, cat='Binary')
    transfer_in = pulp.LpVariable.dicts("TransferIn", all_pids, cat='Binary')

    # Hit Penalty Logic
    if use_chip_wildcard:
        hit_penalty = 0
    else:
        num_transfers = pulp.lpSum([transfer_in[i] for i in all_pids])
        extra_transfers = pulp.LpVariable("ExtraTransfers", lowBound=0, cat='Integer')
        prob += extra_transfers >= num_transfers - free_transfers
        hit_penalty = 4.0 * extra_transfers

    # Objective: Maximize xP
    total_xp = (
        pulp.lpSum([xp_dict[i] * start_vars[i] for i in all_pids]) + 
        pulp.lpSum([xp_dict[i] * cap_vars[i] for i in all_pids]) - 
        hit_penalty
    )
    prob += total_xp

    # Constraints
    prob += pulp.lpSum([squad_vars[i] for i in all_pids]) == 15
    prob += pulp.lpSum([start_vars[i] for i in all_pids]) == 11
    prob += pulp.lpSum([cap_vars[i] for i in all_pids]) == 1

    for i in all_pids:
        prob += start_vars[i] <= squad_vars[i]
        prob += cap_vars[i] <= start_vars[i]

    prob += pulp.lpSum([price_dict[i] * squad_vars[i] for i in all_pids]) <= max_budget

    # Squad Position Constraints
    prob += pulp.lpSum([squad_vars[i] for i in all_pids if pos_dict[i] == 1]) == 2
    prob += pulp.lpSum([squad_vars[i] for i in all_pids if pos_dict[i] == 2]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in all_pids if pos_dict[i] == 3]) == 5
    prob += pulp.lpSum([squad_vars[i] for i in all_pids if pos_dict[i] == 4]) == 3

    # Starting XI Position Constraints
    prob += pulp.lpSum([start_vars[i] for i in all_pids if pos_dict[i] == 1]) == 1
    prob += pulp.lpSum([start_vars[i] for i in all_pids if pos_dict[i] == 2]) >= 3
    prob += pulp.lpSum([start_vars[i] for i in all_pids if pos_dict[i] == 3]) >= 2
    prob += pulp.lpSum([start_vars[i] for i in all_pids if pos_dict[i] == 4]) >= 1

    # Max 3 Players per Team
    teams = set(team_dict.values())
    for t in teams:
        prob += pulp.lpSum([squad_vars[i] for i in all_pids if team_dict[i] == t]) <= 3

    # Transfer Logic
    current_set = set(current_ids)
    for i in all_pids:
        is_current = 1 if i in current_set else 0
        prob += squad_vars[i] == is_current - transfer_out[i] + transfer_in[i]

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    best_squad_ids = [i for i in all_pids if pulp.value(squad_vars[i]) == 1]
    starting_ids = [i for i in all_pids if pulp.value(start_vars[i]) == 1]
    captain_id = [i for i in all_pids if pulp.value(cap_vars[i]) == 1][0]

    return best_squad_ids, starting_ids, captain_id


# --- B. GENETIC ALGORITHM OPTIMIZER ---
def run_genetic_algorithm(df, current_ids, bank, free_transfers, use_chip_wildcard=False, pop_size=80, generations=50):
    all_ids = df['id'].tolist()
    id_to_price = dict(zip(df['id'], df['now_cost'] / 10.0))
    id_to_xp = dict(zip(df['id'], df['predicted_xP']))
    id_to_pos = dict(zip(df['id'], df['element_type']))
    id_to_team = dict(zip(df['id'], df['team']))

    current_cost = sum(id_to_price[i] for i in current_ids if i in id_to_price)
    max_budget = current_cost + bank

    def is_valid_squad(squad_ids):
        if len(squad_ids) != 15: return False
        if sum(id_to_price[i] for i in squad_ids) > max_budget: return False
        
        pos_counts = {1:0, 2:0, 3:0, 4:0}
        for i in squad_ids: pos_counts[id_to_pos[i]] += 1
        if pos_counts != {1:2, 2:5, 3:5, 4:3}: return False
        
        team_counts = {}
        for i in squad_ids:
            t = id_to_team[i]
            team_counts[t] = team_counts.get(t, 0) + 1
            if team_counts[t] > 3: return False
            
        return True

    def calculate_fitness(squad_ids):
        transfers_made = len(set(squad_ids) - set(current_ids))
        if use_chip_wildcard:
            hit_penalty = 0
        else:
            extra_transfers = max(0, transfers_made - free_transfers)
            hit_penalty = extra_transfers * 4.0
            
        total_xp = sum(id_to_xp[i] for i in squad_ids)
        return total_xp - hit_penalty

    population = []
    if is_valid_squad(current_ids): population.append(current_ids)

    attempts = 0
    while len(population) < pop_size and attempts < 2000:
        attempts += 1
        num_swaps = random.randint(0, 10 if use_chip_wildcard else 4)
        candidate = list(current_ids)
        
        for _ in range(num_swaps):
            if candidate:
                candidate.pop(random.randint(0, len(candidate) - 1))
            new_pick = random.choice(all_ids)
            if new_pick not in candidate: candidate.append(new_pick)
                
        if is_valid_squad(candidate): population.append(candidate)

    if not population: population = [current_ids]

    for _ in range(generations):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        survivors = population[:pop_size // 2]
        
        children = []
        while len(survivors) + len(children) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            split = random.randint(1, 14)
            child = list(set(p1[:split] + p2[split:]))
            
            missing = [i for i in all_ids if i not in child]
            random.shuffle(missing)
            while len(child) < 15 and missing: child.append(missing.pop())
                
            if random.random() < 0.35:
                m_idx = random.randint(0, 14)
                rand_p = random.choice(all_ids)
                if rand_p not in child: child[m_idx] = rand_p
                    
            if is_valid_squad(child): children.append(child)
                
        population = survivors + children

    best_squad = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)[0]
    return best_squad


# --- HELPER SELEKSI STARTING XI ---
def select_starting_xi(squad_df):
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
    
    chips_available = st.sidebar.multiselect("Chip Tersedia:", ["Wildcard", "Free Hit", "Bench Boost", "Triple Captain"], default=["Wildcard", "Free Hit"])
    
    # KONTROL BARU: PILIH CHIP UNTUK DIEKSEKUSI PEKAN INI
    active_chip = st.sidebar.selectbox(
        "⚡ Eksekusi Chip Pekan Ini:",
        options=["Tanpa Chip"] + chips_available,
        index=0,
        help="Pilih 'Wildcard' untuk perombakan total tanpa penalti hit -4 poin."
    )

    use_wildcard = (active_chip == "Wildcard")
    use_free_hit = (active_chip == "Free Hit")

    if ("Wildcard" in chips_available or "Free Hit" in chips_available) and active_chip == "Tanpa Chip":
        st.sidebar.warning("💡 **Tips Paruh Pertama:** Jangan lupa gunakan Wildcard / Free Hit sebelum reset paruh musim agar chip Anda tidak hangus!")

    st.subheader("📋 Skuad Terdaftar (15 Pemain)")
    default_selected = elements[elements["id"].isin(st.session_state["user_ids"])]["web_name"].tolist()
    
    selected_names = st.multiselect("Daftar Pemain Anda Saat Ini:", options=elements["web_name"].tolist(), default=default_selected)
    current_df = elements[elements["web_name"].isin(selected_names)].copy()

    if not current_df.empty:
        st.dataframe(
            current_df[["web_name", "team_name", "element_type", "now_cost", "status", "form", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0),
            use_container_width=True
        )

        col_opt1, col_opt2 = st.columns(2)
        
        # --- TOMBOL OPTIMASI 1: REALISTIC MILP ---
        run_milp = col_opt1.button("🎯 JALANKAN OPTIMASI REALISTIS (MILP Solver)", use_container_width=True)
        # --- TOMBOL OPTIMASI 2: GENETIC ALGORITHM ---
        run_ga = col_opt2.button("🧬 JALANKAN OPTIMASI GENETIC ALGORITHM", use_container_width=True)

        if run_milp or run_ga:
            current_ids = current_df["id"].tolist()
            
            if run_milp:
                with st.spinner("MILP Solver sedang mencari solusi matematis terbaik (Global Optimum)..."):
                    best_squad_ids, starting_ids_milp, captain_id_milp = run_realistic_milp(
                        elements, current_ids, bank_money, free_transfers, use_chip_wildcard=use_wildcard
                    )
                    final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                    starting_xi = elements[elements['id'].isin(starting_ids_milp)].sort_values(by="predicted_xP", ascending=False)
                    bench = final_squad_df[~final_squad_df['id'].isin(starting_ids_milp)].sort_values(by="predicted_xP", ascending=False)
                    opt_type = "MILP Solver (Presisi Tinggi)"
            else:
                with st.spinner("Genetic Algorithm sedang mensimulasikan evolusi kombinasi transfer..."):
                    best_squad_ids = run_genetic_algorithm(
                        elements, current_ids, bank_money, free_transfers, use_chip_wildcard=use_wildcard
                    )
                    final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                    starting_xi, bench = select_starting_xi(final_squad_df)
                    opt_type = "Genetic Algorithm"
                
            st.divider()

            # --- IDENTIFIKASI REKOMENDASI TRANSFER ---
            transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
            transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
            
            t_out_df = elements[elements['id'].isin(transfers_out_ids)]
            t_in_df = elements[elements['id'].isin(transfers_in_ids)]

            # -----------------------------------------------------------------
            # OUTPUT 1: REKOMENDASI TRANSFER & CHIP
            # -----------------------------------------------------------------
            st.subheader(f"1. 🔄 Hasil Rekomendasi Transfer ({opt_type})")
            
            num_transfers = len(transfers_in_ids)
            hit_cost = 0 if use_wildcard else max(0, num_transfers - free_transfers) * 4
            
            if use_wildcard:
                st.success(f"🎉 **WILDCARD AKTIF:** Berhasil melakukan perombakan {num_transfers} pemain tanpa penalti Hit (0 Pts Penalty).")
            elif num_transfers == 0:
                st.success("✅ **Saran Transfer:** **TIDAK ADA TRANSFER (0 Transfer)**. Kombinasi tim eksisting Anda sudah optimal.")
            else:
                st.success(f"✅ **Saran Transfer:** {num_transfers} Transfer direkomendasikan (Penalti Hit: -{hit_cost} Pts).")

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("🔴 **Pemain Keluar (Transfer Out):**")
                st.dataframe(t_out_df[["web_name", "team_name", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)
            with col_t2:
                st.markdown("🟢 **Pemain Masuk (Transfer In):**")
                st.dataframe(t_in_df[["web_name", "team_name", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)

            # -----------------------------------------------------------------
            # OUTPUT 2: STARTING LINEUP & KAPTEN
            # -----------------------------------------------------------------
            st.subheader("2. 🏆 Starting Lineup & Pemilihan Kapten")
            
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
                st.metric("📊 NET PROYEKSI POIN (Setelah Hit)", f"{round(net_expected_pts, 2)} Pts")

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