import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Config Halaman
st.set_page_config(
    page_title="Dashboard Analisis Penjualan Kopi Tokopedia",
    page_icon="☕",
    layout="wide"
)

# Load Data Bersih
@st.cache_data
def load_data():
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memuat file 'data_kopi_tokopedia_clean.csv'. Pastikan file berada dalam folder yang sama.")
    st.stop()

# Judul Utama
st.title("☕ Dashboard Analisis Penjualan Kopi Tokopedia")
st.markdown("Dashboard interaktif untuk menganalisis hubungan antara **Harga**, **Rating**, dan **Jumlah Terjual**.")

# Sidebar Filter
st.sidebar.header("🔍 Filter Data")

# Filter Varian Kopi
varians = df['Varian Kopi'].dropna().unique().tolist()
selected_varians = st.sidebar.multiselect("Pilih Varian Kopi:", options=varians, default=varians)

# Filter Harga
min_harga, max_harga = int(df['Harga'].min()), int(df['Harga'].max())
selected_harga = st.sidebar.slider("Rentang Harga (Rp):", min_value=min_harga, max_value=max_harga, value=(min_harga, max_harga))

# Filter Rating
min_rating, max_rating = float(df['Rating'].min()), float(df['Rating'].max())
selected_rating = st.sidebar.slider("Rentang Rating:", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating), step=0.1)

# Terapkan Filter
df_filtered = df[
    (df['Varian Kopi'].isin(selected_varians)) &
    (df['Harga'].between(selected_harga[0], selected_harga[1])) &
    (df['Rating'].between(selected_rating[0], selected_rating[1]))
]

# Ringkasan KPI Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Produk", f"{len(df_filtered):,} item")
m2.metric("Rata-rata Harga", f"Rp {df_filtered['Harga'].mean():,.0f}" if not df_filtered.empty else "Rp 0")
m3.metric("Rata-rata Rating", f"{df_filtered['Rating'].mean():.2f} ⭐" if not df_filtered.empty else "0")
m4.metric("Total Terjual", f"{df_filtered['Terjual'].sum():,} unit" if not df_filtered.empty else "0")

st.markdown("---")

# Halaman Utama Menggunakan Tab
tab1, tab2, tab3 = st.tabs(["📊 Analisis Regresi & Korelasi", "📈 Market Insight", "📋 Data Table"])

# TAB 1: REGRESI & KORELASI
with tab1:
    st.subheader("1. Model Regresi Linier Berganda")
    
    if len(df_filtered) > 5:
        X = df_filtered[['Harga', 'Rating']]
        Y = df_filtered['Terjual']
        
        # Fit Model Regresi
        model = LinearRegression()
        model.fit(X, Y)
        
        col_reg1, col_reg2 = st.columns([1, 1])
        
        with col_reg1:
            st.markdown("### 📐 Persamaan Regresi")
            const = model.intercept_
            coef_harga = model.coef_[0]
            coef_rating = model.coef_[1]
            
            st.latex(rf"Y = {const:.2f} + ({coef_harga:.6f}) \cdot \text{{Harga}} + ({coef_rating:.2f}) \cdot \text{{Rating}}")
            
            st.info(f"""
            **Keterangan Koefisien:**
            * **Konstanta ($a$):** {const:.2f}
            * **Koefisien Harga ($b_1$):** {coef_harga:.6f}
            * **Koefisien Rating ($b_2$):** {coef_rating:.2f}
            """)
            
        with col_reg2:
            st.markdown("### 🧮 Simulator Prediksi Penjualan")
            input_harga = st.number_input("Masukkan Harga Produk (Rp):", min_value=1000, value=50000, step=5000)
            input_rating = st.slider("Masukkan Rating Produk:", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
            
            pred_terjual = model.predict([[input_harga, input_rating]])[0]
            pred_clean = max(0, int(round(pred_terjual)))
            
            st.success(f"**Estimasi Penjualan:** ± {pred_clean} unit")
        
        st.markdown("---")
        st.subheader("2. Matriks Korelasi & Visualisasi Scatter")
        
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            corr = df_filtered[['Harga', 'Rating', 'Terjual']].corr()
            fig_corr = px.imshow(
                corr, 
                text_auto=".2f", 
                color_continuous_scale="coolwarm",
                title="Matriks Korelasi (Harga vs Rating vs Terjual)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_graph2:
            fig_scatter = px.scatter(
                df_filtered, 
                x="Harga", 
                y="Terjual", 
                color="Rating",
                hover_data=["Nama Produk", "Varian Kopi"],
                title="Hubungan Harga vs Terjual (Scatter Plot)",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    else:
        st.warning("Data terlalu sedikit untuk kalkulasi regresi. Perluas rentang filter di sidebar.")

# TAB 2: MARKET INSIGHT
with tab2:
    st.subheader("📊 Analisis Distribusi Pasar")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        # Total Terjual Berdasarkan Varian Kopi
        varian_sales = df_filtered.groupby("Varian Kopi")["Terjual"].sum().reset_index().sort_values(by="Terjual", ascending=False)
        fig_varian = px.bar(
            varian_sales, 
            x="Terjual", 
            y="Varian Kopi", 
            orientation="h",
            title="Total Penjualan per Varian Kopi",
            color="Terjual",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_varian, use_container_width=True)
        
    with col_m2:
        # Top 10 Toko / Lokasi
        toko_top = df_filtered["Asal/Lokasi Toko"].value_counts().head(10).reset_index()
        toko_top.columns = ["Toko/Lokasi", "Jumlah Produk"]
        fig_toko = px.bar(
            toko_top,
            x="Jumlah Produk",
            y="Toko/Lokasi",
            orientation="h",
            title="Top 10 Toko/Lokasi Produk Terbanyak",
            color="Jumlah Produk",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_toko, use_container_width=True)

# TAB 3: DATA TABLE
with tab3:
    st.subheader("📋 Data Hasil Cleaning")
    st.dataframe(df_filtered, use_container_width=True)
    
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Terfilter (CSV)",
        data=csv_data,
        file_name="data_kopi_tokopedia_filtered.csv",
        mime="text/csv"
    )
