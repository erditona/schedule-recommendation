# ==============================================================================
# APLIKASI SIMULASI & OPTIMISASI JADWAL (VERSI FINAL - DENGAN VISUALISASI)
# ==============================================================================
from filecmp import clear_cache
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random
import joblib
import warnings
from datetime import datetime
from deap import base, creator, tools, algorithms

# ==============================================================================
# STEP 1: KONFIGURASI HALAMAN & JUDUL
# ==============================================================================
warnings.filterwarnings("ignore")  # Mengabaikan peringatan yang tidak penting
st.set_page_config(page_title="Simulasi Jadwal Cerdas", page_icon="🧠", layout="wide")
st.title("Ruang Simulasi & Optimisasi Jadwal Armada 🧠")
st.markdown("Masukkan parameter, jalankan simulasi, dan biarkan AI menemukan jadwal terbaik untuk Anda.")

# ==============================================================================
# STEP 2: MEMUAT MODEL, DATA, DAN SETUP GA
# ==============================================================================
@st.cache_resource
def load_model_and_data():
    """Memuat model .pkl dan data bahan baku dari CSV."""
    try:
        # model = joblib.load('model_valid_for_ga(new).pkl')
        # model = joblib.load('model_lgbm_final.pkl')
        bundle = joblib.load('model_lgbm_final.pkl')
        preprocessor = bundle['preprocessor']
        model = bundle['model']
        df_sumber = pd.read_csv('data_consume.csv')
        df_sumber['rute'] = df_sumber['asal'] + ' - ' + df_sumber['tujuan']
        daftar_rute = sorted(df_sumber['rute'].unique())
        daftar_armada = sorted(df_sumber['kd_armada'].unique())
        armada_kapasitas_map = df_sumber.dropna(subset=['kapasitas']).groupby('kd_armada')['kapasitas'].first().to_dict()
        return model, preprocessor, daftar_rute, daftar_armada, armada_kapasitas_map
    except FileNotFoundError as e:
        st.error(f"ERROR: File tidak ditemukan. Detail: {e}")
        return None, None, None, None, None

model_ml, preprocessor, daftar_rute, daftar_armada, armada_kapasitas_map = load_model_and_data()

if not model_ml:
    st.stop()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# ==============================================================================
# STEP 3: MEMBUAT FORMULIR INPUT DI SIDEBAR
# ==============================================================================
slot_waktu = list(range(3*60, 23*60 + 1, 30))
max_possible_trips = len(slot_waktu)

st.sidebar.header("Parameter Simulasi")
tanggal_simulasi = st.sidebar.date_input(
    "Pilih Tanggal Simulasi",
    value=datetime.now(),
    min_value=datetime.now()
)
rute_pilihan = st.sidebar.selectbox("Pilih Rute untuk Disimulasikan", daftar_rute)
jumlah_trip = st.sidebar.slider("Jumlah Perjalanan yang Akan Dibuat", min_value=30, max_value=max_possible_trips, value=35)
run_button = st.sidebar.button("Jalankan Simulasi & Optimisasi", type="primary")

# ==============================================================================
# STEP 4: FUNGSI-FUNGSI UTAMA
# ==============================================================================
waktu_putar_menit = 30
waktu_tempuh_asumsi = 90
waktu_trip_total = waktu_tempuh_asumsi + waktu_putar_menit

def get_time_category(minute_of_day):
    hour = minute_of_day / 60
    if 5 <= hour <= 10: return 'Pagi'
    elif 11 <= hour <= 14: return 'Siang'
    elif 15 <= hour <= 18: return 'Sore'
    else: return 'Malam'

def ambil_fitur_lag(trip, df_histori):
    rute = str(trip['rute']).strip() 
    jam = trip['jam']

    # Filter data histori berdasarkan rute
    df_rute = df_histori[df_histori['rute'] == rute]

    # Cek jika histori untuk rute tersebut tidak ada
    if df_rute.empty:
        return pd.Series({
            'load_factor_lag1': 0.0,
            'load_factor_roll_avg3': 0.0
        })

    # LAG 1: ambil rata-rata load factor dari jam sebelumnya (30 menit sebelum)
    jam_lag1 = jam - 30
    load_factor_lag1 = df_rute[df_rute['jam'] == jam_lag1]['load_factor'].mean()

    # ROLLING 3: ambil rata-rata load factor dari 3 slot sebelumnya
    jam_window = [jam - 30, jam - 60, jam - 90]
    rolling_3 = df_rute[df_rute['jam'].isin(jam_window)]['load_factor'].mean()

    # Fallback jika hasil mean kosong (NaN)
    load_factor_lag1 = load_factor_lag1 if not np.isnan(load_factor_lag1) else 0.0
    rolling_3 = rolling_3 if not np.isnan(rolling_3) else 0.0

    return pd.Series({
        'load_factor_lag1': load_factor_lag1,
        'load_factor_roll_avg3': rolling_3
    })

def evaluate_schedule(individual, nama_hari, is_weekend, df_histori):
    if not individual:
        return (0,)

    df_jadwal = pd.DataFrame(individual)
    bus_availability = {bus: -1 for bus in daftar_armada}

    for index, trip in df_jadwal.sort_values(by='jam').iterrows():
        bus = trip['armada']
        if trip['jam'] < bus_availability[bus]:
            return (0,)
        bus_availability[bus] = trip['jam'] + waktu_trip_total

    df_jadwal['day'] = nama_hari
    df_jadwal['flag_weekend'] = is_weekend
    df_jadwal['kapasitas'] = df_jadwal['armada'].map(armada_kapasitas_map).fillna(19)
    df_jadwal['urutan_keberangkatan'] = range(1, len(df_jadwal) + 1)
    df_jadwal['kategori_waktu'] = df_jadwal['jam'].apply(get_time_category)

    # 🔥 Tambahkan lag & rolling dari histori CSV
    fitur_lag_df = df_jadwal.apply(lambda row: ambil_fitur_lag(row, df_histori), axis=1)
    df_jadwal = pd.concat([df_jadwal, fitur_lag_df], axis=1)

    # Pastikan kolom lengkap
    kolom_prediksi = ['day', 'flag_weekend', 'kategori_waktu', 'rute', 'kapasitas',
                      'urutan_keberangkatan', 'load_factor_lag1', 'load_factor_roll_avg3']

    if df_jadwal[kolom_prediksi].isnull().any().any():
        return (0,)  # Jika ada missing value, penalti penuh

    X_pred = df_jadwal[kolom_prediksi]
    X_pred_transformed = preprocessor.transform(X_pred)
    probabilitas_sukses = model_ml.predict_proba(X_pred_transformed)[:, 1]

    total_fitness = sum(probabilitas_sukses * df_jadwal['kapasitas'])
    return (total_fitness,)

# ==============================================================================
# STEP 5: LOGIKA UTAMA APLIKASI
# ==============================================================================
if run_button:
    st.header(f"Hasil Simulasi untuk {tanggal_simulasi.strftime('%d %B %Y')} (Rute: {rute_pilihan})")

    # --- TAHAP 1: MEMBUAT DAN MENAMPILKAN JADWAL AWAL ---
    nama_hari = tanggal_simulasi.strftime('%A')
    is_weekend = 1 if nama_hari in ['Saturday', 'Sunday'] else 0

    jadwal_simulasi = []
    jam_terpilih = sorted(random.sample(slot_waktu, jumlah_trip))
    for jam in jam_terpilih:
        armada = random.choice(daftar_armada)
        jadwal_simulasi.append({
            'rute': rute_pilihan, 'jam': jam, 'armada': armada, 'day': nama_hari,
            'flag_weekend': is_weekend, 'kapasitas': armada_kapasitas_map.get(armada, 19)
        })

    df_simulasi = pd.DataFrame(jadwal_simulasi)
    df_simulasi['urutan_keberangkatan'] = range(1, len(df_simulasi) + 1)
    df_simulasi['kategori_waktu'] = df_simulasi['jam'].apply(get_time_category)
    df_histori = pd.read_csv('data_consume.csv') 
    df_histori['rute'] = df_histori['asal'] + ' - ' + df_histori['tujuan']
    df_histori['rute'] = df_histori['rute'].astype(str).str.strip()
    df_histori['waktu_keberangkatan'] = pd.to_datetime(df_histori['waktu_keberangkatan'], errors='coerce')
    df_histori['jam'] = df_histori['waktu_keberangkatan'].dt.hour * 60 + df_histori['waktu_keberangkatan'].dt.minute
    fitur_lag_df = df_simulasi.apply(lambda row: ambil_fitur_lag(row, df_histori), axis=1)
    df_simulasi = pd.concat([df_simulasi, fitur_lag_df], axis=1)

    X_pred = df_simulasi[['day', 'flag_weekend', 'kategori_waktu', 'rute', 'kapasitas', 'urutan_keberangkatan', 'load_factor_lag1', 'load_factor_roll_avg3']]
    X_pred_transformed = preprocessor.transform(X_pred)
    probabilitas = model_ml.predict_proba(X_pred_transformed)[:, 1]

    df_simulasi['probabilitas'] = probabilitas
    df_simulasi['estimasi_penumpang'] = np.round(df_simulasi['probabilitas'] * df_simulasi['kapasitas']).astype(int)
    df_simulasi['Load Factor'] = df_simulasi['estimasi_penumpang'] / df_simulasi['kapasitas']

    df_sebelum = df_simulasi[df_simulasi['Load Factor'] >= 0.5].sort_values(by='jam').reset_index(drop=True)
    df_sebelum['jam_keberangkatan'] = df_sebelum['jam'].apply(lambda m: f"{m//60:02d}:{m%60:02d}")
    df_sebelum['Rekomendasi'] = np.where(df_sebelum['Load Factor'] >= 0.7, 'Baik 👍', 'Cukup Baik ✅')

    df_wajib_simpan = df_sebelum[df_sebelum['Load Factor'] >= 0.7].copy()

    df_sebelum_tampil = df_sebelum.copy()
    df_sebelum_tampil['Load Factor'] = df_sebelum_tampil['Load Factor'].apply(lambda x: f"{x:.0%}")
    
    st.subheader("Tabel 1: Jadwal Awal (Hasil Prediksi Model)")
    st.dataframe(df_sebelum_tampil[['jam_keberangkatan', 'armada', 'kapasitas', 'Load Factor', 'Rekomendasi']], use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- TAHAP 2: MENJALANKAN OPTIMISASI DENGAN GA ---
    if len(df_sebelum) == 0:
        st.warning("Tidak ada jadwal awal yang memenuhi kriteria (Load Factor > 50%). Optimisasi tidak dapat dijalankan.")
        st.stop()

    pop_size = max(30, len(df_sebelum) * 2)
    n_gen = max(15, len(df_sebelum))

    with st.spinner(f"Menjalankan optimisasi (Populasi: {pop_size}, Generasi: {n_gen})..."):
        clear_cache()
        toolbox = base.Toolbox()

        # Pastikan kolom 'rute' dan 'jam' tersedia dan bersih
        if 'rute' not in df_histori.columns:
            df_histori['rute'] = df_histori['asal'] + ' - ' + df_histori['tujuan']

        df_histori['rute'] = df_histori['rute'].astype(str).str.strip()

        # Tambahkan kolom 'jam' dari waktu keberangkatan (dalam bentuk menit)
        if 'jam' not in df_histori.columns:
            df_histori['waktu_keberangkatan'] = pd.to_datetime(df_histori['waktu_keberangkatan'], errors='coerce')
            df_histori['jam'] = df_histori['waktu_keberangkatan'].dt.hour * 60 + df_histori['waktu_keberangkatan'].dt.minute

        def create_random_trip():
            return {'rute': rute_pilihan, 'jam': random.choice(slot_waktu), 'armada': random.choice(daftar_armada)}
        toolbox.register("individual", tools.initRepeat, creator.Individual, create_random_trip, n=len(df_sebelum))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_schedule, nama_hari=nama_hari, is_weekend=is_weekend, df_histori=df_histori)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        kolom_untuk_individu = ['rute', 'jam', 'armada']
        df_sebelum_untuk_seed = df_simulasi[df_simulasi['Load Factor'] >= 0.5].sort_values(by='jam').reset_index(drop=True)
        seed_individual_list = df_sebelum_untuk_seed[kolom_untuk_individu].to_dict('records')
        seed_individual = creator.Individual(seed_individual_list)
        pop = toolbox.population(n=pop_size)
        pop[0] = seed_individual
        hof = tools.HallOfFame(1)
        fitnesses_awal = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses_awal): ind.fitness.values = fit
        skor_awal = tools.selBest(pop, 1)[0].fitness.values[0]
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_gen, halloffame=hof, verbose=False)
        best_after_ga = hof[0]
        skor_ga = best_after_ga.fitness.values[0]
        df_hasil_ga = pd.DataFrame(best_after_ga)

        # --- TAHAP 3: MEMPROSES DAN MENGGABUNGKAN HASIL AKHIR ---
        df_hasil_ga['jam_keberangkatan'] = df_hasil_ga['jam'].apply(lambda m: f"{m//60:02d}:{m%60:02d}")
        df_hasil_ga['day'] = nama_hari
        df_hasil_ga['flag_weekend'] = is_weekend
        df_hasil_ga['kategori_waktu'] = df_hasil_ga['jam'].apply(get_time_category)
        df_hasil_ga['kapasitas'] = df_hasil_ga['armada'].map(armada_kapasitas_map).fillna(19)
        df_hasil_ga['urutan_keberangkatan'] = range(1, len(df_hasil_ga) + 1)
        # 🔥 Tambahkan fitur lag dari histori untuk hasil GA
        fitur_lag_ga = df_hasil_ga.apply(lambda row: ambil_fitur_lag(row, df_histori), axis=1)
        df_hasil_ga = pd.concat([df_hasil_ga, fitur_lag_ga], axis=1)
        X_pred_ga = df_hasil_ga[['day', 'flag_weekend', 'kategori_waktu', 'rute', 'kapasitas',
                         'urutan_keberangkatan', 'load_factor_lag1', 'load_factor_roll_avg3']]
        X_pred_ga_transformed = preprocessor.transform(X_pred_ga)
        df_hasil_ga['probabilitas'] = model_ml.predict_proba(X_pred_ga_transformed)[:, 1]
        df_hasil_ga['estimasi_penumpang'] = np.round(df_hasil_ga['probabilitas'] * df_hasil_ga['kapasitas']).astype(int)
        df_hasil_ga['Load Factor'] = df_hasil_ga['estimasi_penumpang'] / df_hasil_ga['kapasitas']
        df_hasil_ga['Rekomendasi'] = np.where(df_hasil_ga['Load Factor'] >= 0.7, 'Baik ✅', 'Cukup Baik 🆗')
        df_dari_ga = df_hasil_ga[df_hasil_ga['Load Factor'] >= 0.7].copy()
        df_final_gabungan = pd.concat([df_wajib_simpan, df_dari_ga], ignore_index=True)
        df_final_gabungan.drop_duplicates(subset=['jam_keberangkatan', 'armada'], keep='last', inplace=True)
        df_final_gabungan.sort_values(by='jam', inplace=True)
        
        # --- TAHAP 4: MENAMPILKAN HASIL AKHIR (TABEL & METRIK) ---
        df_final_tampil = df_final_gabungan.copy()
        df_final_tampil['Load Factor'] = df_final_tampil['Load Factor'].apply(lambda x: f"{x:.0%}")
        
        st.subheader("Tabel 2: Jadwal Akhir (Gabungan & Optimisasi GA)")
        st.dataframe(df_final_tampil[['jam_keberangkatan', 'armada', 'kapasitas', 'Load Factor', 'Rekomendasi']], use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Trip Awal (>=50%)", len(df_sebelum))
        col2.metric("Jumlah Trip Akhir (>=70% & Gabungan)", len(df_final_gabungan))
        col1.metric("Skor Awal (Dari Individu Terbaik Awal)", f"{skor_awal:,.2f}")
        col2.metric("Skor Optimal (Hasil Akhir GA)", f"{skor_ga:,.2f}", delta=f"{skor_ga - skor_awal:,.2f}")
        
    # --- TAHAP 5: VISUALISASI PERBANDINGAN ---
    st.markdown("---")
    st.header("Visualisasi Perbandingan Hasil")

    # Siapkan data untuk chart (gunakan dataframe dengan data numerik)
    df_sebelum_numerik = df_sebelum # Ini masih punya LF numerik
    df_final_numerik = df_final_gabungan # Ini juga masih punya LF numerik

    # 1. Visualisasi Distribusi Trip per Jam
    st.subheader("Distribusi Jumlah Trip per Jam")
    col1_chart, col2_chart = st.columns(2)

    # Tampilkan distribusi untuk Jadwal Awal (df_sebelum_numerik)
    with col1_chart:
        df_sebelum_numerik['jam_kategori'] = df_sebelum_numerik['jam'].apply(lambda m: m // 60)
        distribusi_sebelum = df_sebelum_numerik['jam_kategori'].value_counts().sort_index().reset_index()
        distribusi_sebelum.columns = ['Jam', 'Jumlah Trip']

        # Membuat bar chart dengan warna biru untuk distribusi awal
        chart_sebelum = alt.Chart(distribusi_sebelum).mark_bar(color='#1f77b4').encode(
            x=alt.X('Jam:N', title='Jam'),
            y=alt.Y('Jumlah Trip:Q', title='Jumlah Trip'),
            tooltip=['Jam', 'Jumlah Trip']
        ).properties(
            width=300,
            height=200
        )

        st.altair_chart(chart_sebelum, use_container_width=True)
        st.caption("Gambar 1: Distribusi Jadwal Awal")

    # Tampilkan distribusi untuk Jadwal Optimal (df_final_numerik)
    with col2_chart:
        df_final_numerik['jam_kategori'] = df_final_numerik['jam'].apply(lambda m: m // 60)
        distribusi_sesudah = df_final_numerik['jam_kategori'].value_counts().sort_index().reset_index()
        distribusi_sesudah.columns = ['Jam', 'Jumlah Trip']

        # Membuat bar chart dengan warna oranye untuk distribusi optimal
        chart_sesudah = alt.Chart(distribusi_sesudah).mark_bar(color='#ff7f0e').encode(
            x=alt.X('Jam:N', title='Jam'),
            y=alt.Y('Jumlah Trip:Q', title='Jumlah Trip'),
            tooltip=['Jam', 'Jumlah Trip']
        ).properties(
            width=300,
            height=200
        )

        st.altair_chart(chart_sesudah, use_container_width=True)
        st.caption("Gambar 2: Distribusi Jadwal Optimal")

    # 2. Visualisasi Perbandingan KPI
    st.subheader("Perbandingan Key Performance Indicators (KPI)")

    # Hitung KPI
    total_penumpang_sebelum = df_sebelum_numerik['estimasi_penumpang'].sum()
    total_penumpang_sesudah = df_final_numerik['estimasi_penumpang'].sum()
    avg_lf_sebelum = df_sebelum_numerik['Load Factor'].mean()
    avg_lf_sesudah = df_final_numerik['Load Factor'].mean()

    # Buat DataFrame untuk KPI
    kpi_data = {
        'Metrik': ['Total Estimasi Penumpang', 'Rata-rata Load Factor'],
        'Jadwal Awal': [total_penumpang_sebelum, avg_lf_sebelum],
        'Jadwal Optimal': [total_penumpang_sesudah, avg_lf_sesudah]
    }

    df_kpi = pd.DataFrame({
        'Metrik': ['Total Estimasi Penumpang', 'Rata-rata Load Factor'],
        'Jadwal Awal': [f"{total_penumpang_sebelum:.0f}", f"{avg_lf_sebelum:.0%}"],
        'Jadwal Optimal': [f"{total_penumpang_sesudah:.0f}", f"{avg_lf_sesudah:.0%}"]
    }).set_index('Metrik')

    st.dataframe(df_kpi, use_container_width=True)

    # Data yang akan digunakan
    df_penumpang_chart = pd.DataFrame({
        'Jadwal': ['Awal', 'Optimal'],
        'Total Estimasi Penumpang': [total_penumpang_sebelum, total_penumpang_sesudah]
    }).set_index('Jadwal')

    # Membuat bar chart dengan warna berbeda untuk kategori "Awal" dan "Optimal"
    bar_chart = alt.Chart(df_penumpang_chart.reset_index()).mark_bar().encode(
        x='Jadwal:N',  # Menentukan sumbu x
        y='Total Estimasi Penumpang:Q',  # Menentukan sumbu y
        color=alt.Color('Jadwal:N', scale=alt.Scale(domain=['Awal', 'Optimal'], range=['#1f77b4', '#ff7f0e'])),  # Menentukan warna berdasarkan kategori
        tooltip=['Jadwal', 'Total Estimasi Penumpang']  # Menambahkan tooltip
    ).properties(
        width=300,
        height=200
    )

    # Menampilkan bar chart
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Gambar 3: Perbandingan Total Estimasi Penumpang")


else:
    st.info("Silakan pilih parameter di sidebar kiri dan klik tombol 'Jalankan Simulasi & Optimisasi'.")