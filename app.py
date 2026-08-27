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

st.title(f"⚽ FPL Squad Manager & Transfer Optimizer (GW {next_gw})")
st.markdown("Aplikasi optimasi transfer, komposisi skuad pasca-transfer, dan simulasi kapten.")

# Sidebar - Parameter Skuad & Transfer
st.sidebar.header("⚙️ Konfigurasi Tim & Transfer")

# Fitur Sync via Team ID
st.sidebar.subheader("🆔 Sync Otomatis ID Tim FPL")
team_id_input = st.sidebar.text_input("Masukkan FPL Team ID Anda:", placeholder="Contoh: 1234567")

all_player_names = sorted(players_df['web_name'].unique())
fetched_squad_names = []

if team_id_input:
    try:
        # Ambil data picks Gameweek sebelumnya/terbaru dari API
        current_gw = next_gw - 1 if next_gw > 1 else 1
        entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id_input}/event/{current_gw}/picks/"
        res = requests.get(entry_url)
        if res.status_code == 200:
            picks_data = res.json().get('picks', [])
            player_ids = [p['element'] for p in picks_data]
            fetched_squad_names = players_df[players_df['id'].isin(player_ids)]['web_name'].tolist()
            st.sidebar.success(f"Berhasil menarik 15 pemain dari ID: {team_id_input}!")
        else:
            st.sidebar.error("ID Tim tidak ditemukan atau API tidak merespons.")
    except Exception:
        st.sidebar.error("Gagal mengambil data dari FPL.")

# Set default selection
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

tab1, tab2, tab3 = st.tabs(["🔄 Rekomendasi Transfer", "📋 Skuad Pasca-Transfer & Kapten", "🔍 Database Pemain"])

# TAB 1: REKOMENDASI TRANSFER
with tab1:
    st.subheader("🎯 Rekomendasi Transfer Optimal")
    st.caption("Rekomendasi dihitung berdasarkan kenaikan Expected Points (xP Net Gain), FDR jadwal depan, status ketersediaan, dan sisa budget.")

    recommendations = []
    if len(squad_df) < 15:
        st.warning(f"Pilih 15 pemain di sidebar untuk memulai analisis. (Baru terpilih: {len(squad_df)}/15)")
    else:
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

# TAB 2: SKUAD BARU PASCA-TRANSFER & SIMULASI KAPTEN
with tab2:
    st.subheader(f"📋 Analisis Skuad Baru Pasca-Transfer (GW {next_gw})")
    
    if len(squad_df) < 15:
        st.info("Lengkapi 15 pemain di sidebar terlebih dahulu.")
    else:
        st.markdown("#### ⚙️ Pilih Pemain yang Ditransfer")
        
        if not rec_df.empty:
            transfer_options = ["(Tanpa Transfer - Gunakan Skuad Asli)"] + [
                f"Ganti {row['Pemain Keluar']} ➡️ {row['Pemain Masuk']} (+{row['Estimasi Net Gain Poin']} pts)"
                for _, row in rec_df.iterrows()
            ]
            selected_transfer_str = st.selectbox("Terapkan Rekomendasi Transfer ke Skuad:", options=transfer_options)
        else:
            selected_transfer_str = "(Tanpa Transfer - Gunakan Skuad Asli)"
            st.info("Tidak ada rekomendasi transfer. Menampilkan analisis skuad asli Anda.")

        post_squad_names = selected_squad_names.copy()
        applied_penalty = 0

        if selected_transfer_str != "(Tanpa Transfer - Gunakan Skuad Asli)":
            idx = transfer_options.index(selected_transfer_str) - 1
            chosen_rec = rec_df.iloc[idx]
            
            post_squad_names.remove(chosen_rec['Pemain Keluar'])
            post_squad_names.append(chosen_rec['Pemain Masuk'])
            
            if free_transfers <= 0:
                applied_penalty = 4

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

# TAB 3: DATABASE PEMAIN
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
