import streamlit as st
import pandas as pd
import requests
from itertools import combinations

st.set_page_config(
    page_title="FPL Multi-Transfer & Squad Optimizer",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_fpl_data():
    base_url = "https://fantasy.premierleague.com/api/"
    bootstrap = requests.get(f"{base_url}bootstrap-static/").json()
    fixtures = requests.get(f"{base_url}fixtures/").json()
    
    players = pd.DataFrame(bootstrap['elements'])
    teams = pd.DataFrame(bootstrap['teams']).set_index('id')['name'].to_dict()
    element_types = pd.DataFrame(bootstrap['element_types']).set_index('id')['singular_name_short'].to_dict()
    
    players['team_name'] = players['team'].map(teams)
    players['position'] = players['element_type'].map(element_types)
    players['now_cost'] = players['now_cost'] / 10.0
    players['form'] = pd.to_numeric(players['form'], errors='coerce').fillna(0)
    
    next_gw = None
    for event in bootstrap['events']:
        if event['is_next']:
            next_gw = event['id']
            break
            
    if not next_gw:
        next_gw = 1

    next_fixtures = [f for f in fixtures if f.get('event') == next_gw]
    
    next_opponent_dict = {}
    fdr_dict = {}
    
    for f in next_fixtures:
        home_team = f['team_h']
        away_team = f['team_a']
        home_difficulty = f['team_h_difficulty']
        away_difficulty = f['team_a_difficulty']
        
        next_opponent_dict[home_team] = f"{teams[away_team]} (H)"
        fdr_dict[home_team] = home_difficulty
        
        next_opponent_dict[away_team] = f"{teams[home_team]} (A)"
        fdr_dict[away_team] = away_difficulty
        
    players['next_match'] = players['team'].map(next_opponent_dict).fillna("Blank / TBA")
    players['next_fdr'] = players['team'].map(fdr_dict).fillna(3)
    
    return players, teams, fixtures, next_gw

try:
    players_df, teams_dict, fixtures_data, next_gw = load_fpl_data()
except Exception as e:
    st.error("Failed to load FPL API data. Check internet connection.")
    st.stop()

def calculate_expected_points(df):
    chance_mult = df['chance_of_playing_next_round'].fillna(100) / 100.0
    fdr_multiplier = df['next_fdr'].apply(lambda fdr: 1.25 if fdr <= 2 else (0.75 if fdr >= 4 else 1.0))
    df['xP'] = (df['form'] * chance_mult * fdr_multiplier).round(2)
    return df

players_df = calculate_expected_points(players_df)

st.title(f"⚽ FPL Auto-Optimal Transfer & Squad Optimizer (GW {next_gw})")
st.markdown("Sistem secara otomatis mengevaluasi kombinasi transfer (1-4 pemain) dan menyajikan skenario paling optimum.")

# Sidebar - Parameter
st.sidebar.header("⚙️ Konfigurasi Tim & Transfer")

st.sidebar.subheader("🆔 Sync Otomatis ID Tim FPL")
team_id_input = st.sidebar.text_input("Masukkan FPL Team ID Anda:", value="4231710")

all_player_names = sorted(players_df['web_name'].unique())
fetched_squad_names = []

if team_id_input:
    try:
        current_gw = next_gw - 1 if next_gw > 1 else 1
        entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id_input}/event/{current_gw}/picks/"
        res = requests.get(entry_url)
        if res.status_code == 200:
            picks_data = res.json().get('picks', [])
            player_ids = [p['element'] for p in picks_data]
            fetched_squad_names = players_df[players_df['id'].isin(player_ids)]['web_name'].tolist()
            st.sidebar.success(f"Berhasil menarik 15 pemain dari ID: {team_id_input}!")
        else:
            st.sidebar.error("ID Tim tidak ditemukan.")
    except Exception:
        st.sidebar.error("Gagal mengambil data dari FPL.")

if fetched_squad_names:
    default_selection = fetched_squad_names
elif 'saved_squad' in st.session_state:
    default_selection = st.session_state['saved_squad']
else:
    default_selection = all_player_names[:15] if len(all_player_names) >= 15 else []

st.sidebar.subheader("📌 Masukkan/Ubah Skuad")
selected_squad_names = st.sidebar.multiselect(
    "Pilih 15 Pemain dalam Tim Anda:",
    options=all_player_names,
    default=default_selection
)

if st.sidebar.button("💾 Simpan Skuad Saat Ini"):
    st.session_state['saved_squad'] = selected_squad_names
    st.sidebar.success("Skuad disimpan!")

free_transfers = st.sidebar.number_input("Jumlah Free Transfer Tersedia", min_value=1, max_value=5, value=1)
bank_budget = st.sidebar.number_input("Sisa Anggaran di Bank (£M)", min_value=0.0, max_value=15.0, value=0.5, step=0.1)

squad_df = players_df[players_df['web_name'].isin(selected_squad_names)].copy()

tab1, tab2, tab3 = st.tabs(["🔄 Rekomendasi Optimal (Auto)", "📋 Skuad Pasca-Transfer & Kapten", "🔍 Database Pemain"])

# ---------------------------------------------------------
# TAB 1: EVALUASI OTOMATIS 1-4 TRANSFER
# ---------------------------------------------------------
with tab1:
    st.subheader("🎯 Hasil Transfer Paling Optimum (Otomatis)")
    st.caption("Algoritma secara otomatis menguji semua opsi (1 s.d. 4 transfer) dan menyajikan kombinasi dengan ROI Net Gain tertinggi.")

    recommendations = []
    if len(squad_df) < 15:
        st.warning(f"Lengkapi 15 pemain di sidebar untuk menjalankan optimasi. (Saat ini: {len(squad_df)}/15)")
    else:
        with st.spinner("Mencari kombinasi transfer paling optimum dari skenario 1-4 pemain..."):
            # Secara otomatis mengevaluasi rentang transfer 1 hingga 4
            for n_transfers in range(1, 5):
                extra_transfers = max(0, n_transfers - free_transfers)
                hit_penalty = extra_transfers * 4

                for out_comb in combinations(squad_df.to_dict('records'), n_transfers):
                    out_cost_sum = sum([p['now_cost'] for p in out_comb])
                    out_xp_sum = sum([p['xP'] for p in out_comb])
                    available_budget = out_cost_sum + bank_budget

                    out_positions = [p['position'] for p in out_comb]
                    
                    candidate_pool = players_df[
                        (~players_df['web_name'].isin(selected_squad_names)) &
                        (players_df['chance_of_playing_next_round'].fillna(100) >= 75) &
                        (players_df['position'].isin(out_positions))
                    ].sort_values(by='xP', ascending=False).head(12).to_dict('records')

                    for in_comb in combinations(candidate_pool, n_transfers):
                        in_positions = [p['position'] for p in in_comb]
                        if sorted(out_positions) == sorted(in_positions):
                            in_cost_sum = sum([p['now_cost'] for p in in_comb])
                            
                            if in_cost_sum <= available_budget:
                                in_xp_sum = sum([p['xP'] for p in in_comb])
                                gross_xp_gain = in_xp_sum - out_xp_sum
                                net_xp_gain = gross_xp_gain - hit_penalty

                                if net_xp_gain > 0.5:
                                    recommendations.append({
                                        'Opsi Transfer': f"{n_transfers} Pemain",
                                        'Penalti Hit': f"-{hit_penalty} pts" if hit_penalty > 0 else "0 pts (Free)",
                                        'Pemain Keluar': ", ".join([p['web_name'] for p in out_comb]),
                                        'Pemain Masuk': ", ".join([p['web_name'] for p in in_comb]),
                                        'Sisa Budget': round(bank_budget - (in_cost_sum - out_cost_sum), 2),
                                        'Gross Gain': round(gross_xp_gain, 2),
                                        'Net Gain (ROI)': round(net_xp_gain, 2)
                                    })

        rec_df = pd.DataFrame(recommendations)

        if not rec_df.empty:
            rec_df = rec_df.sort_values(by='Net Gain (ROI)', ascending=False).drop_duplicates(subset=['Pemain Keluar', 'Pemain Masuk']).reset_index(drop=True)
            
            # Highlight Pilihan Terbaik (Top 1)
            best_option = rec_df.iloc[0]
            st.success(f"🏆 **Rekomendasi Terbaik**: Lakukan **{best_option['Opsi Transfer']}** (Penalti: {best_option['Penalti Hit']}) untuk potensi peningkatan bersih **+{best_option['Net Gain (ROI)']} poin**!")

            st.dataframe(rec_df.head(10), use_container_width=True)
        else:
            st.success("Skuad Anda saat ini sudah optimal. Melakukan transfer tambahan tidak disarankan untuk gameweek ini.")

# ---------------------------------------------------------
# TAB 2: SKUAD PASCA-TRANSFER & SIMULASI KAPTEN
# ---------------------------------------------------------
with tab2:
    st.subheader(f"📋 Analisis Skuad Baru Pasca-Transfer (GW {next_gw})")
    
    if len(squad_df) < 15:
        st.info("Lengkapi 15 pemain di sidebar terlebih dahulu.")
    else:
        st.markdown("#### ⚙️ Pilih Opsi Hasil Transfer")
        
        if not rec_df.empty:
            transfer_options = ["(Tanpa Transfer - Gunakan Skuad Asli)"] + [
                f"[{row['Opsi Transfer']} | {row['Penalti Hit']}] Out: {row['Pemain Keluar']} ➡️ In: {row['Pemain Masuk']} (+{row['Net Gain (ROI)']} pts)"
                for _, row in rec_df.iterrows()
            ]
            # Default ke opsi teratas (terbaik)
            selected_transfer_str = st.selectbox("Terapkan Hasil Transfer ke Skuad:", options=transfer_options, index=1)
        else:
            selected_transfer_str = "(Tanpa Transfer - Gunakan Skuad Asli)"
            st.info("Tidak ada rekomendasi transfer. Menampilkan analisis skuad asli Anda.")

        post_squad_names = selected_squad_names.copy()
        applied_penalty = 0

        if selected_transfer_str != "(Tanpa Transfer - Gunakan Skuad Asli)":
            idx = transfer_options.index(selected_transfer_str) - 1
            chosen_rec = rec_df.iloc[idx]
            
            out_list = chosen_rec['Pemain Keluar'].split(", ")
            in_list = chosen_rec['Pemain Masuk'].split(", ")
            
            for out_p in out_list:
                if out_p in post_squad_names:
                    post_squad_names.remove(out_p)
            for in_p in in_list:
                post_squad_names.append(in_p)
            
            n_tx = int(chosen_rec['Opsi Transfer'].split()[0])
            extra_tx = max(0, n_tx - free_transfers)
            applied_penalty = extra_tx * 4

        post_squad_df = players_df[players_df['web_name'].isin(post_squad_names)].copy()
        post_squad_df = post_squad_df.sort_values(by='xP', ascending=False)

        st.divider()

        top_captain = post_squad_df.iloc[0]['web_name']
        top_vc = post_squad_df.iloc[1]['web_name'] if len(post_squad_df) > 1 else top_captain

        st.markdown("### 👑 Pemilihan Kapten (Skuad Baru)")
        col_c1, col_c2 = st.columns(2)
        
        selected_captain = col_c1.selectbox(
            "Pilih Kapten (2x Poin):",
            options=post_squad_df['web_name'].tolist(),
            index=0
        )
        
        remaining_vc_options = [p for p in post_squad_df['web_name'].tolist() if p != selected_captain]
        selected_vc = col_c2.selectbox(
            "Pilih Vice-Captain:",
            options=remaining_vc_options,
            index=0
        )

        captain_xp = post_squad_df[post_squad_df['web_name'] == selected_captain]['xP'].values[0]
        base_squad_xp = post_squad_df['xP'].sum()
        final_achieved_xp = round(base_squad_xp + captain_xp - applied_penalty, 2)

        st.info(f"💡 **Rekomendasi Kapten Skuad Baru**: **{top_captain}** (Base xP: {post_squad_df.iloc[0]['xP']}) & Vice-Captain **{top_vc}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total xP Achieved (GW Depan)", final_achieved_xp)
        col2.metric("Ekstra Poin Kapten", f"+{captain_xp} pts")
        col3.metric("Penalti Transfer (Hit)", f"-{applied_penalty} pts" if applied_penalty > 0 else "0 pts")
        
        flagged_players = post_squad_df[post_squad_df['chance_of_playing_next_round'] < 100]
        col4.metric("Pemain Cedera/Bermasalah", len(flagged_players))

        post_squad_display = post_squad_df.copy()
        post_squad_display['Role'] = post_squad_display['web_name'].apply(
            lambda name: "👑 Captain" if name == selected_captain else ("🛡️ Vice-Captain" if name == selected_vc else "Player")
        )

        st.dataframe(
            post_squad_display[['Role', 'web_name', 'position', 'team_name', 'next_match', 'next_fdr', 'now_cost', 'chance_of_playing_next_round', 'xP']]
            .rename(columns={
                'web_name': 'Nama',
                'position': 'Pos',
                'team_name': 'Klub',
                'next_match': 'Lawan GW Depan',
                'next_fdr': 'FDR',
                'now_cost': 'Harga (£M)',
                'chance_of_playing_next_round': 'Peluang Main (%)',
                'xP': 'Base xP'
            }),
            hide_index=True,
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 3: DATABASE PEMAIN
# ---------------------------------------------------------
with tab3:
    st.subheader("🔍 Database Pemain & Tren Performa")
    
    col_f1, col_f2 = st.columns(2)
    pos_filter = col_f1.multiselect("Filter Posisi", options=list(players_df['position'].unique()), default=list(players_df['position'].unique()))
    max_price = col_f2.slider("Maksimal Harga (£M)", 4.0, 15.0, 15.0, 0.5)

    filtered_players = players_df[
        (players_df['position'].isin(pos_filter)) &
        (players_df['now_cost'] <= max_price)
    ].sort_values(by='xP', ascending=False)

    st.dataframe(
        filtered_players[['web_name', 'position', 'team_name', 'next_match', 'next_fdr', 'now_cost', 'form', 'selected_by_percent', 'xP']]
        .rename(columns={'web_name': 'Nama', 'position': 'Pos', 'team_name': 'Klub', 'next_match': 'Lawan GW Depan', 'next_fdr': 'FDR', 'now_cost': 'Harga (£M)', 'selected_by_percent': 'Kepemilikan (%)'}),
        hide_index=True,
        use_container_width=True
    )
