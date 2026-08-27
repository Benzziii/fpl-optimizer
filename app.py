import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="FPL Transfer & Squad Optimizer",
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
    
    # -----------------------------------------------------
    # MENENTUKAN FIXTURE & FDR UNTUK GAMEWEEK BERIKUTNYA
    # -----------------------------------------------------
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

# ---------------------------------------------------------
# RUMUS EXPECTED POINTS (XP) BERBASIS FORM, CEDERA & FDR
# ---------------------------------------------------------
def calculate_expected_points(df):
    chance_mult = df['chance_of_playing_next_round'].fillna(100) / 100.0
    
    # Penyesuaian Multiplier Kesulitan Jadwal
    # FDR 1-2: Multiplier 1.25 | FDR 3: 1.0 | FDR 4-5: 0.75
    fdr_multiplier = df['next_fdr'].apply(lambda fdr: 1.25 if fdr <= 2 else (0.75 if fdr >= 4 else 1.0))
    
    # Formula xP Lengkap
    df['xP'] = (df['form'] * chance_mult * fdr_multiplier).round(2)
    return df

players_df = calculate_expected_points(players_df)

st.title(f"⚽ FPL Squad Manager & Transfer Optimizer (Gameweek {next_gw})")
st.markdown("Aplikasi optimasi transfer berbasis statistik *Form*, *Kondisi Cedera*, dan *Tingkat Kesulitan Jadwal (FDR)*.")

st.sidebar.header("⚙️ Konfigurasi Tim & Transfer")

free_transfers = st.sidebar.number_input("Jumlah Free Transfer Tersedia", min_value=1, max_value=5, value=1)
bank_budget = st.sidebar.number_input("Sisa Anggaran di Bank (£M)", min_value=0.0, max_value=15.0, value=0.5, step=0.1)

st.sidebar.subheader("📌 Masukkan Skuad Saat Ini")
all_player_names = sorted(players_df['web_name'].unique())

default_squad = all_player_names[:15] if len(all_player_names) >= 15 else []
selected_squad_names = st.sidebar.multiselect(
    "Pilih 15 Pemain dalam Tim Anda:",
    options=all_player_names,
    default=default_squad
)

squad_df = players_df[players_df['web_name'].isin(selected_squad_names)].copy()

tab1, tab2, tab3 = st.tabs(["📋 Skuad Saat Ini & Analisis", "🔄 Rekomendasi Transfer", "🔍 Exploratory Pemain"])

with tab1:
    st.subheader(f"Kondisi Skuad Gameweek {next_gw}")
    if len(squad_df) < 15:
        st.warning(f"Pilih 15 pemain di sidebar untuk analisis lengkap. (Baru terpilih: {len(squad_df)}/15)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Estimasi xP", round(squad_df['xP'].sum(), 2))
    
    flagged_players = squad_df[squad_df['chance_of_playing_next_round'] < 100]
    col2.metric("Pemain Bermasalah/Cedera", len(flagged_players))
    col3.metric("Sisa Bank Budget", f"£{bank_budget}M")

    st.dataframe(
        squad_df[['web_name', 'position', 'team_name', 'next_match', 'next_fdr', 'now_cost', 'form', 'chance_of_playing_next_round', 'news', 'xP']]
        .rename(columns={
            'web_name': 'Nama',
            'position': 'Pos',
            'team_name': 'Klub',
            'next_match': 'Lawan GW Depan',
            'next_fdr': 'FDR (1-5)',
            'now_cost': 'Harga (£M)',
            'chance_of_playing_next_round': 'Peluang Main (%)',
            'news': 'Catatan Medis/Keluaran',
            'xP': 'Expected Pts'
        }),
        hide_index=True,
        use_container_width=True
    )

with tab2:
    st.subheader("🎯 Rekomendasi Transfer Optimal")
    st.caption("Rekomendasi dihitung berdasarkan kenaikan Expected Points (xP Net Gain), FDR jadwal depan, status ketersediaan, dan sisa budget.")

    if len(squad_df) == 0:
        st.info("Pilih pemain terlebih dahulu di sidebar.")
    else:
        recommendations = []

        for _, out_player in squad_df.iterrows():
            max_affordable_price = out_player['now_cost'] + bank_budget
            
            candidates = players_df[
                (players_df['position'] == out_player['position']) &
                (~players_df['web_name'].isin(selected_squad_names)) &
                (players_df['now_cost'] <= max_affordable_price) &
                (players_df['chance_of_playing_next_round'].fillna(100) >= 75)
            ].copy()

            for _, in_player in candidates.iterrows():
                gross_xp_gain = in_player['xP'] - out_player['xP']
                hit_penalty = 0 if free_transfers > 0 else 4
                net_xp_gain = gross_xp_gain - hit_penalty

                if net_xp_gain > 0.5:
                    recommendations.append({
                        'Pemain Keluar': out_player['web_name'],
                        'Klub Lama': out_player['team_name'],
                        'Pemain Masuk': in_player['web_name'],
                        'Klub Baru': in_player['team_name'],
                        'Lawan Pemain Masuk': in_player['next_match'],
                        'FDR Masuk': in_player['next_fdr'],
                        'Posisi': out_player['position'],
                        'Harga Masuk': f"£{in_player['now_cost']}M",
                        'Selisih Budget': round(bank_budget - (in_player['now_cost'] - out_player['now_cost']), 2),
                        'xP Out': out_player['xP'],
                        'xP In': in_player['xP'],
                        'Estimasi Net Gain Poin': round(net_xp_gain, 2)
                    })

        rec_df = pd.DataFrame(recommendations)

        if not rec_df.empty:
            rec_df = rec_df.sort_values(by='Estimasi Net Gain Poin', ascending=False).reset_index(drop=True)
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.success("Skuad Anda saat ini sudah optimal. Tidak ada usulan transfer yang memberikan ROI poin signifikan.")

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
