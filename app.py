import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Kopi Tokopedia",
    page_icon="☕",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    df['Omzet'] = df['Harga'] * df['Terjual']
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file csv berada dalam folder yang sama!")
    st.stop()

# ---------------------------------------------------------
# HEADER DASHBOARD
# ---------------------------------------------------------
st.title("☕ Dashboard Analisis Pasar Kopi Tokopedia")
st.markdown("Dashboard interaktif untuk menganalisis hubungan antara **Harga**, **Rating**, **Varian Kopi**, dan tingkat **Penjualan (Terjual)**.")

# ---------------------------------------------------------
# SIDEBAR - FILTER DATA
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter Data")

# Filter Varian Kopi
varians = ['Semua Varian'] + sorted(list(df['Varian Kopi'].dropna().unique()))
selected_varian = st.sidebar.selectbox("Pilih Varian Kopi", varians)

# Filter Rentang Harga
min_harga, max_harga = int(df['Harga'].min()), int(df['Harga'].max())
selected_harga = st.sidebar.slider(
    "Rentang Harga (Rp)",
    min_value=min_harga,
    max_value=max_harga,
    value=(min_harga, min(max_harga, 500000)),
    step=100
)

# Filter Rating
min_rating, max_rating = float(df['Rating'].min()), float(df['Rating'].max())
selected_rating = st.sidebar.slider(
    "Rentang Rating",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=0.1
)

# Penerapan Filter
df_filtered = df[
    (df['Harga'] >= selected_harga[0]) & 
    (df['Harga'] <= selected_harga[1]) &
    (df['Rating'] >= selected_rating[0]) & 
    (df['Rating'] <= selected_rating[1])
]

if selected_varian != 'Semua Varian':
    df_filtered = df_filtered[df_filtered['Varian Kopi'] == selected_varian]

# ---------------------------------------------------------
# METRICS METRIK UTAMA (KPI)
# ---------------------------------------------------------
st.markdown("### 📊 Ringkasan Eksekutif")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Produk", f"{len(df_filtered):,} item")
with col2:
    total_terjual = df_filtered['Terjual'].sum() if not df_filtered.empty else 0
    st.metric("Total Terjual", f"{total_terjual:,} unit")
with col3:
    mean_rating = df_filtered['Rating'].mean() if not df_filtered.empty else 0
    st.metric("Rata-Rata Rating", f"⭐ {mean_rating:.2f}")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    mean_harga = df_filtered['Harga'].mean() if not df_filtered.empty else 0
    st.metric("Rata-Rata Harga", f"Rp {mean_harga:,.0f}")
with col2:
    total_omzet = df_filtered['Omzet'].sum() if not df_filtered.empty else 0
    st.metric("Total Estimasi Omzet", f"Rp {total_omzet:,.0f}")
# ---------------------------------------------------------
# TAB ANALISIS & VISUALISASI
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Analisis & Korelasi", "🏆 Top Produk & Toko", "🤖 Simulasi Regresi Linier & Omset"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribusi Varian Kopi")
        df_varian = df_filtered['Varian Kopi'].value_counts().reset_index()
        df_varian.columns = ['Varian Kopi', 'Jumlah Produk']
        fig_varian = px.bar(
            df_varian.head(10), 
            x='Jumlah Produk', 
            y='Varian Kopi', 
            orientation='h',
            title="Top 10 Jumlah Produk per Varian",
            color='Jumlah Produk',
            color_continuous_scale='Blues'
        )
        fig_varian.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_varian, use_container_width=True)

    with col_right:
        st.subheader("Hubungan Harga vs Terjual")
        fig_scatter = px.scatter(
            df_filtered, 
            x='Harga', 
            y='Terjual', 
            color='Varian Kopi',
            hover_data=['Nama Produk', 'Rating'],
            title="Scatter Plot: Harga vs Terjual"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Matriks Korelasi (Harga, Rating, Terjual)")
    if len(df_filtered) > 1:
        corr = df_filtered[['Harga', 'Rating', 'Terjual']].corr()
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Korelasi Antar Variabel Numerik"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("10 Produk Terlaris")
        top_produk = df_filtered.nlargest(10, 'Terjual')[['Nama Produk', 'Varian Kopi', 'Harga', 'Terjual']]
        st.dataframe(top_produk, hide_index=True, use_container_width=True)

    with col_t2:
        st.subheader("10 Toko / Lokasi Penjualan Tertinggi")
        top_toko = df_filtered.groupby('Asal/Lokasi Toko')['Terjual'].sum().reset_index().nlargest(10, 'Terjual')
        fig_toko = px.bar(
            top_toko,
            x='Terjual',
            y='Asal/Lokasi Toko',
            orientation='h',
            title="Total Terjual per Toko",
            color='Terjual',
            color_continuous_scale='Greens'
        )
        fig_toko.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_toko, use_container_width=True)
    # ---------------------------------------------------------
    # FUNGSI HELPER (Ditaruh di bagian atas app.py)
    # ---------------------------------------------------------
    @st.cache_data
    def convert_df_to_csv(df):
        """Mengubah DataFrame ke CSV dengan Caching agar hemat memori/proses."""
        return df.to_csv(index=False).encode('utf-8')
    
    
    # ---------------------------------------------------------
    # TABEL DATA & DOWNLOAD (Gantikan snippet lama dengan ini)
    # ---------------------------------------------------------
    st.markdown("---")
    
    # Layout sejajar untuk Judul + Informasi Jumlah Data dan Tombol Download
    col_header, col_download = st.columns([3, 1], vertical_alignment="bottom")
    
    with col_header:
        st.subheader("📋 Seluruh Data Terfilter")
        st.caption(f"Menampilkan **{len(df_filtered):,}** baris data berdasarkan filter aktif.")
    
    with col_download:
        # Mengonversi CSV menggunakan fungsi ber-cache
        csv_data = convert_df_to_csv(df_filtered)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="data_kopi_tokopedia_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    
    # Tampilan Tabel Interaktif dengan Format Kolom
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga": st.column_config.NumberColumn(
                "Harga",
                format="Rp %'d",
            ),
            "Rating": st.column_config.NumberColumn(
                "Rating",
                format="⭐ %.1f",
            ),
            "Terjual": st.column_config.NumberColumn(
                "Terjual",
                format="%d unit",
            ),
            "Link Produk": st.column_config.LinkColumn(
                "Link Tokopedia",
                display_text="Lihat Produk"
            )
        }
    )

with tab3:
    st.subheader("Model Regresi Linier Berganda")
    st.markdown("Model ini menghitung estimasi **Jumlah Terjual (Y)** berdasarkan **Harga ($X_1$)** dan **Rating ($X_2$)**.")
    
    # Training Model Regresi
    X = df[['Harga', 'Rating']]
    Y = df['Terjual']
    model = LinearRegression()
    model.fit(X, Y)
    
    a = model.intercept_
    b1 = model.coef_[0]
    b2 = model.coef_[1]
    
    st.info("Persamaan Regresi Linier Berganda Terbentuk:")
    st.latex(rf"Y = {a:.2f} + ({b1:.6f}) \times \text{{Harga}} + ({b2:.2f}) \times \text{{Rating}}")
    
    st.markdown("#### 🔮 Simulasi Prediksi Penjualan & Omzet")
    st.caption("Masukkan estimasi harga dan rating untuk memproyeksikan potensi unit terjual dan omzet produk baru Anda.")
    
    # ---------------------------------------------------------
    # 1. INPUT PARAMETER SIMULASI (SINKRON DENGAN RENTANG DATASET)
    # ---------------------------------------------------------
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        # Menggunakan min_harga & max_harga dari dataset dengan step 100
        input_harga = st.number_input(
            "Harga Produk (Rp)", 
            min_value=min_harga, 
            max_value=max_harga, 
            value=min(max(50000, min_harga), max_harga),  # Safe default value
            step=100,                                     # Diselaraskan dengan step=100
            help=f"Masukkan harga produk antara Rp {min_harga:,} - Rp {max_harga:,}"
        )
    
    with col_input2:
        # Rata-rata rating dataset sebagai nilai default yang aman
        avg_rating = float(np.round(df['Rating'].mean(), 1))
        default_rating = max(min_rating, min(avg_rating, max_rating))
        
        # Menggunakan min_rating & max_rating dari dataset
        input_rating = st.slider(
            "Target Rating Produk", 
            min_value=min_rating, 
            max_value=max_rating,
            value=default_rating,
            step=0.1,
            help=f"Rentang rating antara {min_rating:.1f} - {max_rating:.1f}"
        )
    
    # ---------------------------------------------------------
    # 2. KALKULASI PREDIKSI (MENGGUNAKAN DATAFRAME AGAR BEBAS WARNING)
    # ---------------------------------------------------------
    input_data = pd.DataFrame([[input_harga, input_rating]], columns=['Harga', 'Rating'])
    pred_terjual_raw = model.predict(input_data)[0]
    
    # Mencegah angka negatif
    pred_terjual_clean = max(0, int(np.round(pred_terjual_raw)))
    pred_omzet = pred_terjual_clean * input_harga
    
    # ---------------------------------------------------------
    # 3. HASIL PROYEKSI (METRIC CARDS)
    # ---------------------------------------------------------
    st.markdown("##### 📊 Hasil Proyeksi")
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric(
            label="Estimasi Unit Terjual", 
            value=f"{pred_terjual_clean:,} unit"
        )
    
    with col_res2:
        st.metric(
            label="Estimasi Total Omzet", 
            value=f"Rp {pred_omzet:,.0f}"
        )
    
    with col_res3:
        st.metric(
            label="Harga Input Produk", 
            value=f"Rp {input_harga:,.0f}"
        )
    
    # ---------------------------------------------------------
    # 4. INSIGHT OTOMATIS BERDASARKAN MODEL
    # ---------------------------------------------------------
    if pred_terjual_clean == 0:
        st.warning("⚠️ **Peringatan:** Kombinasi harga ini terlalu tinggi atau rating terlalu rendah menurut pola data historis.")
    else:
        b1_harga = model.coef_[0]
        b2_rating = model.coef_[1]
        
        dampak_harga = abs(int(np.round(b1_harga * 10000)))
        dampak_rating = int(np.round(b2_rating * 0.1))
        
        st.info(
            f"💡 **Insight Bisnis:**\n"
            f"* Setiap kenaikan **Rating +0.1** diproyeksikan mengubah penjualan sebesar **± {dampak_rating:,} unit**.\n"
            f"* Setiap perubahan **Harga ± Rp 10.000** memengaruhi potensi penjualan sebesar **± {dampak_harga:,} unit**."
        )
    # Visualisasi Pangsa Pasar Omzet
    st.subheader("Omset Kopi")
    st.latex(rf"\text{{Omzet}} = \text{{Harga}} \times \text{{Terjual}}")
    fig_donut_omzet = px.pie(
        df_filtered, 
        names='Varian Kopi', 
        values='Omzet', 
        hole=0.4,
        title="Market Share Omzet (Rupiah) per Varian Kopi",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_donut_omzet.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_donut_omzet, use_container_width=True)
    
    st.subheader("🏆 Top 10 Toko Berdasarkan Omzet Terbesar")
    top_toko_omzet = df_filtered.groupby('Asal/Lokasi Toko')['Omzet'].sum().reset_index().nlargest(10, 'Omzet')
    
    fig_toko_omzet = px.bar(
        top_toko_omzet,
        x='Omzet',
        y='Asal/Lokasi Toko',
        orientation='h',
        title="Total Omzet (Rp) per Toko/Lokasi",
        color='Omzet',
        color_continuous_scale='Greens'
    )
    fig_toko_omzet.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_toko_omzet, use_container_width=True)
    
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga": st.column_config.NumberColumn("Harga", format="Rp %'d"),
            "Terjual": st.column_config.NumberColumn("Terjual", format="%d unit"),
            "Omzet": st.column_config.NumberColumn("Estimasi Omzet", format="Rp %'d"), # <-- Tambahan Omzet
            "Rating": st.column_config.NumberColumn("Rating", format="⭐ %.1f")
        }
    )
