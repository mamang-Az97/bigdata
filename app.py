import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Config Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Analisis Tokopedia Kopi",
    page_icon="☕",
    layout="wide"
)

# Fungsi Ekstraksi Varian Kopi dari Nama Produk
def extract_varian(nama):
    nama_lower = str(nama).lower()
    if 'arabika' in nama_lower or 'arabica' in nama_lower:
        return 'Arabika'
    elif 'robusta' in nama_lower:
        return 'Robusta'
    elif 'blend' in nama_lower or 'house blend' in nama_lower:
        return 'Blend'
    elif 'luwak' in nama_lower:
        return 'Luwak'
    elif 'espresso' in nama_lower:
        return 'Espresso'
    else:
        return 'Lainnya'

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    df['Varian_Kopi'] = df['Nama_Produk'].apply(extract_varian)
    df['Estimasi_Omzet'] = df['Harga'] * df['Terjual']
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memuat 'dataset_kopi_tokopedia_clean.csv'. Pastikan file berada di folder yang sama dengan app.py!")
    st.stop()

# Header Utama
st.title("☕ Dashboard Analisis Penjualan Kopi Tokopedia")
st.markdown("Analisis interaktif mengenai hubungan **Harga**, **Rating**, **Lokasi Toko**, dan **Jumlah Terjual**.")

st.sidebar.header("🔍 Filter Interactive")

# 1. Filter Cari Nama Produk
search_query = st.sidebar.text_input("Cari Kata Kunci Produk:", value="")

# 2. Filter Varian Kopi
all_varians = df['Varian_Kopi'].unique().tolist()
selected_varians = st.sidebar.multiselect("Pilih Varian Kopi:", options=all_varians, default=all_varians)

# 3. Filter Lokasi Toko
all_lokasi = sorted(df['Lokasi'].dropna().unique().tolist())
selected_lokasi = st.sidebar.multiselect("Pilih Lokasi Toko:", options=all_lokasi, default=all_lokasi)

# 4. Filter Rentang Harga
min_h, max_h = int(df['Harga'].min()), int(df['Harga'].max())
selected_harga = st.sidebar.slider("Rentang Harga (Rp):", min_value=min_h, max_value=max_h, value=(min_h, 500000))

# 5. Filter Rentang Rating
min_r, max_r = float(df['Rating'].min()), float(df['Rating'].max())
selected_rating = st.sidebar.slider("Rentang Rating:", min_value=min_r, max_value=max_r, value=(min_r, max_r), step=0.1)

# Terapkan Filter
df_filtered = df[
    (df['Nama_Produk'].str.contains(search_query, case=False, na=False)) &
    (df['Varian_Kopi'].isin(selected_varians)) &
    (df['Lokasi'].isin(selected_lokasi)) &
    (df['Harga'].between(selected_harga[0], selected_harga[1])) &
    (df['Rating'].between(selected_rating[0], selected_rating[1]))
]

# KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Produk", f"{len(df_filtered):,} item")
col2.metric("Rata-rata Harga", f"Rp {df_filtered['Harga'].mean():,.0f}" if not df_filtered.empty else "Rp 0")
col3.metric("Rata-rata Rating", f"{df_filtered['Rating'].mean():.2f} ⭐" if not df_filtered.empty else "0")
col4.metric("Total Terjual", f"{df_filtered['Terjual'].sum():,.0f} unit" if not df_filtered.empty else "0")
col5.metric("Estimasi Omzet", f"Rp {df_filtered['Estimasi_Omzet'].sum():,.0f}" if not df_filtered.empty else "Rp 0")

st.markdown("---")

# Pengelompokan Menggunakan Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Regresi & Korelasi", "📈 Insight Pasar & Lokasi", "📋 Tabel Data & Top Seller"])

# TAB 1: REGRESI & KORELASI
with tab1:
    st.subheader("1. Model Regresi Linier Berganda")
    st.caption("Memprediksi variabel terikat (Jumlah Terjual) berdasarkan variabel bebas (Harga & Rating).")
    
    if len(df_filtered) > 5:
        X = df_filtered[['Harga', 'Rating']]
        Y = df_filtered['Terjual']
        
        model = LinearRegression()
        model.fit(X, Y)
        y_pred = model.predict(X)
        r2 = r2_score(Y, y_pred)
        
        reg_col1, reg_col2 = st.columns([1.2, 1])
        
        with reg_col1:
            st.markdown("### 📐 Persamaan Regresi")
            const = model.intercept_
            coef_harga = model.coef_[0]
            coef_rating = model.coef_[1]
            
            st.latex(rf"\hat{{Y}} = {const:.2f} + ({coef_harga:.6f}) \cdot \text{{Harga}} + ({coef_rating:.2f}) \cdot \text{{Rating}}")
            
            st.info(f"""
            **Hasil Evaluasi Model:**
            * **Koefisien Determinasi ($R^2$):** `{r2:.4f}`
            * **Interpretasi Harga:** Setiap kenaikan harga Rp 1.000, estimasi penjualan berubah sebesar **{coef_harga * 1000:.2f} unit**.
            * **Interpretasi Rating:** Setiap kenaikan 1 poin rating, estimasi penjualan berubah sebesar **{coef_rating:.2f} unit**.
            """)
            
        with reg_col2:
            st.markdown("### 🧮 Simulasi Prediksi Penjualan")
            input_harga = st.number_input("Harga Produk (Rp):", min_value=1000, value=65000, step=5000)
            input_rating = st.slider("Rating Produk:", min_value=1.0, max_value=5.0, value=4.9, step=0.1)
            
            pred_sales = model.predict([[input_harga, input_rating]])[0]
            pred_clean = max(0, int(round(pred_sales)))
            
            st.success(f"📌 **Estimasi Penjualan:** ± **{pred_clean:,}** unit")

        st.markdown("---")
        st.subheader("2. Matriks Korelasi & Scatter Plot")
        
        sc_col1, sc_col2 = st.columns(2)
        
        with sc_col1:
            corr_df = df_filtered[['Harga', 'Rating', 'Terjual']].corr()
            fig_corr = px.imshow(
                corr_df,
                text_auto=".3f",
                color_continuous_scale="RdBu_r",
                title="Heatmap Korelasi Pearson"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with sc_col2:
            use_log = st.checkbox("Gunakan Skala Logaritmik (Harga vs Terjual)", value=False)
            fig_scatter = px.scatter(
                df_filtered,
                x="Harga",
                y="Terjual",
                color="Varian_Kopi",
                hover_data=["Nama_Produk", "Rating", "Lokasi"],
                title="Hubungan Harga vs Penjualan (Terjual)",
                trendline="ols",
                log_x=use_log,
                log_y=use_log
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Data hasil filter terlalu sedikit untuk kalkulasi regresi (minimal 5 data).")

# TAB 2: INSIGHT PASAR & LOKASI
with tab2:
    st.subheader("📊 Analisis Sebaran Varian & Demografi Lokasi")
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        # Total Penjualan Berdasarkan Varian Kopi
        varian_summary = df_filtered.groupby("Varian_Kopi")["Terjual"].sum().reset_index().sort_values(by="Terjual", ascending=False)
        fig_varian = px.bar(
            varian_summary,
            x="Varian_Kopi",
            y="Terjual",
            color="Varian_Kopi",
            title="Total Volume Penjualan per Varian Kopi (Unit)",
            text_auto=",.0f"
        )
        st.plotly_chart(fig_varian, use_container_width=True)
        
    with col_graph2:
        # Top 10 Lokasi Seller
        lokasi_summary = df_filtered.groupby("Lokasi")["Terjual"].sum().reset_index().sort_values(by="Terjual", ascending=False).head(10)
        fig_lokasi = px.bar(
            lokasi_summary,
            x="Terjual",
            y="Lokasi",
            orientation="h",
            color="Terjual",
            color_continuous_scale="Viridis",
            title="Top 10 Lokasi Penjualan Kopi Terbanyak"
        )
        fig_lokasi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_lokasi, use_container_width=True)

# TAB 3: DATA TABLE & TOP SELLER
with tab3:
    st.subheader("🏆 Top 10 Produk Terlaris")
    top_10 = df_filtered.sort_values(by="Terjual", ascending=False).head(10)[['Nama_Produk', 'Varian_Kopi', 'Harga', 'Rating', 'Terjual', 'Lokasi', 'Estimasi_Omzet']]
    st.dataframe(top_10, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Seluruh Data Terfilter")
    st.dataframe(df_filtered, use_container_width=True)
    
    csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Terfilter (CSV)",
        data=csv_bytes,
        file_name="kopi_tokopedia_filtered.csv",
        mime="text/csv"
    )
