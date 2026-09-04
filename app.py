import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Advanced Defense & GK Matchup Optimizer", layout="wide")
st.title("⚽ FPL Optimizer: Matchup Pertahanan, Kiper Lawan & ML")

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
# HELPER FUNCTION (HARUS DI LUAR SUPAYA CACHING STREAMLIT TIDAK ERROR)
# -----------------------------------------------------------------------------
def get_player_role(row):
    roles = []
    if row.get('penalties_order') == 1: roles.append("⚽ Penalty")
    if row.get('corners_and_indirect_freekicks_order') == 1: roles.append("🚩 Corner")
    if row.get('direct_freekicks_order') == 1: roles.append("🎯 Free-Kick")
    if row.get('element_type') == 2 and (row.get('creativity', 0) > 80 or row.get('threat', 0) > 50): roles.append("🏃 Attacking WB")
    return ", ".join(roles) if roles else "-"

# -----------------------------------------------------------------------------
# 2. ADVANCED ML ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def predict_player_xp(df, teams_df):
    df_feat = df.copy()
    
    # Preprocessing Fitur Standar Pemain
    df_feat['form'] = pd.to_numeric(df_feat['form'], errors='coerce').fillna(0)
    df_feat['ict_index'] = pd.to_numeric(df_feat['ict_index'], errors='coerce').fillna(0)
    df_feat['creativity'] = pd.to_numeric(df_feat['creativity'], errors='coerce').fillna(0)
    df_feat['threat'] = pd.to_numeric(df_feat['threat'], errors='coerce').fillna(0)
    df_feat['points_per_game'] = pd.to_numeric(df_feat['points_per_game'], errors='coerce').fillna(0)
    df_feat['selected_by_percent'] = pd.to_numeric(df_feat['selected_by_percent'], errors='coerce').fillna(0)
    df_feat['ep_next'] = pd.to_numeric(df_feat['ep_next'], errors='coerce').fillna(0)
    
    # Preprocessing Set-Piece & Role
    df_feat['is_penalty_taker'] = np.where(df_feat['penalties_order'] == 1, 1.0, 0.0)
    df_feat['is_corner_taker'] = np.where(df_feat['corners_and_indirect_freekicks_order'] == 1, 1.0, 0.0)
    df_feat['is_fk_taker'] = np.where(df_feat['direct_freekicks_order'] == 1, 1.0, 0.0)
    df_feat['is_attacking_defender'] = np.where(
        (df_feat['element_type'] == 2) & ((df_feat['creativity'] > 80) | (df_feat['threat'] > 50)), 
        1.0, 0.0
    )

    # Opponent Defense Strength
    team_defense_stats = {}
    for team_id in teams_df['id']:
        team_players = df_feat[df_feat['team'] == team_id]
        
        gk_players = team_players[team_players['element_type'] == 1].sort_values(by="form", ascending=False)
        gk_form = gk_players.iloc[0]['form'] if not gk_players.empty else 2.5
        
        def_players = team_players[team_players['element_type'] == 2]
        avg_def_form = def_players['form'].mean() if not def_players.empty else 2.0
        
        defense_strength = (gk_form * 0.4) + (avg_def_form * 0.6)
        team_defense_stats[team_id] = round(defense_strength, 2)

    avg_league_def = np.mean(list(team_defense_stats.values())) if team_defense_stats else 2.0
    
    df_feat['opp_defense_factor'] = df_feat['team'].map(
        lambda t_id: round(avg_league_def / max(0.1, team_defense_stats.get(t_id, avg_league_def)), 2)
    )

    # Matrix Fitur ML
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
    
    rf = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42)
    rf.fit(X, y_target)
    
    df_feat['predicted_xP'] = np.round(rf.predict(X), 2)
    df_feat['special_roles'] = df_feat.apply(get_player_role, axis=1)
    
    return df_feat

# -----------------------------------------------------------------------------
# 3. GENETIC ALGORITHM OPTIMIZER
# -----------------------------------------------------------------------------
def run_genetic_algorithm(df, current_ids, bank, free_transfers, pop_size=40, generations=25):
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
        for i in squad_ids:
            pos_counts[id_to_pos[i]] += 1
        if pos_counts != {1:2, 2:5, 3:5, 4:3}: return False
        
        team_counts = {}
        for i in squad_ids:
            t = id_to_team[i]
            team_counts[t] = team_counts.get(t, 0) + 1
            if team_counts[t] > 3: return False
            
        return True

    def calculate_fitness(squad_ids):
        transfers_made = len(set(squad_ids) - set(current_ids))
        extra_transfers = max(0, transfers_made - free_transfers)
        hit_penalty = extra_transfers * 4.0
        
        total_xp = sum(id_to_xp[i] for i in squad_ids)
        return total_xp - hit_penalty

    population = []
    if is_valid_squad(current_ids):
        population.append(current_ids)

    attempts = 0
    while len(population) < pop_size and attempts < 1000:
        attempts += 1
        candidate = list(current_ids)
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            if candidate:
                candidate.pop(random.randint(0, len(candidate) - 1))
            new_pick = random.choice(all_ids)
            if new_pick not in candidate:
                candidate.append(new_pick)
        if is_valid_squad(candidate) and candidate not in population:
            population.append(candidate)

    while len(population) < 2:
        population.append(current_ids)

    for _ in range(generations):
        population = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)
        cutoff = max(2, len(population) // 2)
        survivors = population[:cutoff]
        
        children = []
        loop_counter = 0
        while len(survivors) + len(children) < pop_size and loop_counter < 100:
            loop_counter += 1
            p1, p2 = random.sample(survivors, 2)
            
            split = random.randint(1, 14)
            child = list(set(p1[:split] + p2[split:]))
            
            missing = [i for i in all_ids if i not in child]
            random.shuffle(missing)
            while len(child) < 15 and missing:
                child.append(missing.pop())
                
            if is_valid_squad(child):
                children.append(child)
                
        population = survivors + children

    best_squad = sorted(population, key=lambda ind: calculate_fitness(ind), reverse=True)[0]
    return best_squad

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
    teams_df = pd.DataFrame(bootstrap["teams"])
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements["team_name"] = elements["team"].map(teams)
    
    elements = predict_player_xp(elements, teams_df)
    
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
            current_df[["web_name", "team_name", "element_type", "now_cost", "status", "special_roles", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0),
            use_container_width=True
        )

        if st.button("🧬 JALANKAN OPTIMASI GENETIC ALGORITHM & ML"):
            current_ids = current_df["id"].tolist()
            
            with st.spinner("Menjalankan Genetic Algorithm..."):
                best_squad_ids = run_genetic_algorithm(elements, current_ids, bank_money, free_transfers)
                final_squad_df = elements[elements['id'].isin(best_squad_ids)].copy()
                starting_xi, bench = select_starting_xi(final_squad_df)
                
            st.divider()

            transfers_out_ids = list(set(current_ids) - set(best_squad_ids))
            transfers_in_ids = list(set(best_squad_ids) - set(current_ids))
            
            t_out_df = elements[elements['id'].isin(transfers_out_ids)]
            t_in_df = elements[elements['id'].isin(transfers_in_ids)]

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
                st.success("✅ **0 TRANSFER**. Skuad eksisting Anda sudah optimal.")
            else:
                st.success(f"✅ Disarankan melakukan **{num_transfers} Transfer** (Hit: -{hit_cost} Pts):")
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("🔴 **Transfer Out:**")
                    st.dataframe(t_out_df[["web_name", "team_name", "special_roles", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)
                with col_t2:
                    st.markdown("🟢 **Transfer In:**")
                    st.dataframe(t_in_df[["web_name", "team_name", "special_roles", "now_cost", "predicted_xP"]].assign(Harga=lambda x: x["now_cost"]/10.0), use_container_width=True)

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
            st.markdown("### 🟢 Starting Eleven")
            st.dataframe(
                starting_xi[["web_name", "team_name", "element_type", "special_roles", "form", "status", "predicted_xP"]]
                .rename(columns={"special_roles": "Peran Khusus (Set-Piece)"}), 
                use_container_width=True
            )

            st.markdown("### 🪑 Bench")
            st.dataframe(
                bench[["web_name", "team_name", "element_type", "special_roles", "form", "status", "predicted_xP"]]
                .rename(columns={"special_roles": "Peran Khusus (Set-Piece)"}), 
                use_container_width=True
            )
else:
    st.error("Gagal terhubung ke API FPL.")
