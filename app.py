import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor

# Config Halaman
st.set_page_config(page_title="FPL Final Optimizer", layout="wide")
st.title("⚽ FPL Final Transfer & Starting Lineup Optimizer")

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
# 2. MACHINE LEARNING ENGINE (PREDICT XP)
# -----------------------------------------------------------------------------
def predict_player_xp(df):
    df_feat = df.copy()
    
    df_feat['form'] = pd.to_numeric(df_feat['form'], errors='coerce').fillna(0)
    df_feat['ict_index'] = pd.to_numeric(df_feat['ict_index'], errors='coerce').fillna(0)
    df_feat['points_per_game'] = pd.to_numeric(df_feat['points_per_game'], errors='coerce').fillna(0)
    df_feat['selected_by_percent'] = pd.to_numeric(df_feat['selected_by_percent'], errors='coerce').fillna(0)
    df_feat['ep_next'] = pd.to_numeric(df_feat['ep_next'], errors='coerce').fillna(0)
    
    X = df_feat[['form', 'ict_index', 'points_per_game', 'selected_by_percent', 'ep_next']]
    
    # Target sintesis berbasis korelasi statistik FPL
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
# 3. OPTIMIZATION ENGINE (RELIABLE GREEDY + REPLACEMENT)
# -----------------------------------------------------------------------------
def optimize_transfers(all_df, current_ids, bank, free_transfers):
    current_df = all_df[all_df['id'].isin(current_ids)].copy()
    available_df = all_df[~all_df['id'].isin(current_ids)].copy()
    
    current_cost = (current_df['now_cost'] / 10.0).sum()
    max_allowed_budget = current_cost + bank
    
    # Urutkan skuad berdasarkan xP terendah (kandidat dijual)
    candidates_out = current_df.sort_values(by="predicted_xP", ascending=True)
    
    best_transfers_out = []
    best_transfers_in = []
    
    # Cari kandidat swap 1 per 1 yang sah & menaikkan total xP
    for i in range(min(free_transfers, len(candidates_out))):
        player_out = candidates_out.iloc[i]
        out_pos = player_out['element_type']
        out_price = player_out['now_cost'] / 10.0
        
        # Cari pengganti di posisi yang sama dan budget cukup
        budget_limit = out_price + bank
        targets = available_df[
            (available_df['element_type'] == out_pos) & 
            (available_df['now_cost'] / 10.0 <= budget_limit) &
            (available_df['status'] == 'a')
        ].sort_values(by="predicted_xP", ascending=False)
        
        if not targets.empty:
            target_in = targets.iloc[0]
            if target_in['predicted_xP'] > player_out['predicted_xP']:
                best_transfers_out.append(player_out)
                best_transfers_in.append(target_in)
                bank -= ((target_in['now_cost'] / 10.0) - out_price)
                # Tandai agar tidak terpakai lagi
                available_df = available_df[available_df['id'] != target_in['id']]

    # Skuad Final Setelah Transfer
    retained_ids = [pid for pid in current_ids if pid not in [p['id'] for p in best_transfers_out]]
    new_ids = [p['id'] for p in best_transfers_in]
    final_ids = retained_ids + new_ids
    
    return best_transfers_out, best_transfers_in, final_ids, bank

def select_starting_xi(squad_df):
    """
    Memilih 11 Pemain Utama dengan Formasi Valid FPL:
    1 GKP, Min 3 DEF, Min 2 MID, Min 1 FWD
    """
    gkps = squad_df[squad_df['element_type'] == 1].sort_values(by="predicted_xP", ascending=False)
    defs = squad_df[squad_df['element_type'] == 2].sort_values(by="predicted_xP", ascending=False)
    mids = squad_df[squad_df['element_type'] == 3].sort_values(by="predicted_xP", ascending=False)
    fwds = squad_df[squad_df['element_type'] == 4].sort_values(by="predicted_xP", ascending=False)
    
    starting_ids = []
    
    # Kebutuhan Wajib
    starting_ids.append(gkps.iloc[0]['id']) # 1 Kiper
    starting_ids.extend(defs.iloc[:3]['id'].tolist()) # 3 Bek
    starting_ids.extend(mids.iloc[:2]['id'].tolist()) # 2 Gelandang
    starting_ids.extend(fwds.iloc[:1]['id'].tolist()) # 1 Penyerang
    
    # Sisa 4 pemain outfield dengan xP tertinggi dari sisa pemain
    remaining_outfield = squad_df[
        (~squad_df['id'].isin(starting_ids)) & 
        (squad_df['element_type'] != 1)
    ].sort_values(by="predicted_xP", ascending=False)
    
    starting_ids.extend(remaining_outfield.iloc[:4]['id'].tolist())
    
    starting_xi = squad_df[squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
    bench = squad_df[~squad_df['id'].isin(starting_ids)].sort_values(by="predicted_xP", ascending=False)
    
    return starting_xi, bench

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI & DISPLAY
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
    
    # Jalankan Prediksi ML untuk seluruh pemain
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
    free_transfers = st.sidebar.number_input("Jumlah Free Transfer:", min_value=1, max_value=5, value=1)
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

        if st.button("🚀 HITUNG REKOMENDASI FINAL"):
            current_ids = current_df["id"].tolist()
            
            with st.spinner("Memproses Model ML & Mengoptimalkan Transfer..."):
                t_out, t_in, final_squad_ids, remaining_bank = optimize_transfers(elements, current_ids, bank_money, free_transfers)
                final_squad_df = elements[elements['id'].isin(final_squad_ids)].copy()
                starting_xi, bench = select_starting_xi(final_squad_df)
                
            st.divider()

            # -----------------------------------------------------------------
            # OUTPUT 1: REKOMENDASI TRANSFER & CHIP
            # -----------------------------------------------------------------
            st.subheader("1. 🔄 Rekomendasi Transfer & Penggunaan Chip")
            
            # Evaluasi Chip
            injured_count = len(current_df[current_df['status'] != 'a'])
            chip_msg = "Saran Chip: Tidak perlu menggunakan Wildcard / Free Hit pekan ini."
            
            if injured_count >= 3 and "Wildcard" in chips_available:
                chip_msg = "⚠️ **Saran Chip:** Gunakan **WILDCARD**! Terdapat 3 atau lebih pemain cedera/halangan di skuad Anda."
            elif injured_count >= 2 and free_transfers == 1 and "Free Hit" in chips_available:
                chip_msg = "💡 **Saran Chip:** Pertimbangkan **FREE HIT** untuk menghindari minus poin (*hit*) pekan ini."
                
            st.info(f"**Strategi Chip:** {chip_msg}")
            
            num_transfers = len(t_out)
            if num_transfers == 0:
                st.success("✅ **Saran Transfer:** **TIDAK ADA TRANSFER** pekan ini. Skuad eksisting Anda sudah dalam kondisi optimal.")
            else:
                st.success(f"✅ **Saran Transfer:** Lakukan **{num_transfers} Transfer** berikut:")
                
                transfer_summary = []
                for idx in range(num_transfers):
                    p_out = t_out[idx]
                    p_in = t_in[idx]
                    transfer_summary.append({
                        "Transfer Out (Keluar)": f"{p_out['web_name']} ({p_out['team_name']}) - £{p_out['now_cost']/10.0}m",
                        "Transfer In (Masuk)": f"{p_in['web_name']} ({p_in['team_name']}) - £{p_in['now_cost']/10.0}m",
                        "Peningkatan xP": f"+{round(p_in['predicted_xP'] - p_out['predicted_xP'], 2)} Pts"
                    })
                st.table(pd.DataFrame(transfer_summary))
                st.caption(f"💰 **Sisa Saldo Bank Setelah Transfer:** £{round(remaining_bank, 2)}m")

            # -----------------------------------------------------------------
            # OUTPUT 2: STARTING LINEUP, CAPTAIN & EXPECTED POINTS
            # -----------------------------------------------------------------
            st.subheader("2. 🏆 Starting Lineup & Pemilihan Kapten (Next Week)")
            
            # Penetapan Kapten & VC dari Starting XI
            captain = starting_xi.iloc[0]
            vice_captain = starting_xi.iloc[1]
            
            # Hitung Total Expected Points (Kapten dihitung 2x)
            total_expected_pts = (starting_xi['predicted_xP'].sum() + captain['predicted_xP'])
            
            col_cap1, col_cap2, col_cap3 = st.columns(3)
            with col_cap1:
                st.metric("👑 CAPTAIN", f"{captain['web_name']}", f"{captain['predicted_xP'] * 2} xP (2x)")
            with col_cap2:
                st.metric("🎖️ VICE-CAPTAIN", f"{vice_captain['web_name']}", f"{vice_captain['predicted_xP']} xP")
            with col_cap3:
                st.metric("📊 PROYEKSI TOTAL POIN SQUAD", f"{round(total_expected_pts, 2)} Pts")

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
