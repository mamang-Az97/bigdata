import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Tokopedia Kopi",
    page_icon="☕",
    layout="wide"
)

# 2. Fungsi Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data_kopi_tokopedia_clean.csv')
    # Kalkulasi Estimasi Omzet
    df['Estimasi Omzet'] = df['Harga'] * df['Terjual']
    return df

try:
    df = load_data()
except Exception as e:
    st.error("File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file tersimpan di direktori yang sama dengan app.py.")
    st.stop()

# 3. Header Utama
st.title("☕ Dashboard Analisis Penjualan Kopi Tokopedia")
st.markdown("Analisis interaktif mengenai hubungan **Harga**, **Rating**, **Toko/Lokasi**, dan **Jumlah Terjual**.")

# 4. Sidebar Filter
st.sidebar.header("🔍 Filter Interaktif")

# Filter Search Nama Produk
search_query = st.sidebar.text_input("Cari Nama Produk:", value="")

# Filter Varian Kopi
varians = sorted(df['Varian Kopi'].dropna().unique().tolist())
selected_varians = st.sidebar.multiselect("Pilih Varian Kopi:", options=varians, default=varians)

# Filter Asal / Lokasi Toko
tokos = sorted(df['Asal/Lokasi Toko'].dropna().unique().tolist())
selected_tokos = st.sidebar.multiselect("Pilih Toko / Lokasi:", options=tokos, default=tokos)

# Filter Rentang Harga
min_h, max_h = int(df['Harga'].min()), int(df['Harga'].max())
selected_harga = st.sidebar.slider("Rentang Harga (Rp):", min_value=min_h, max_value=max_h, value=(min_h, 500000))

# Filter Rentang Rating
min_r, max_r = float(df['Rating'].min()), float(df['Rating'].max())
selected_rating = st.sidebar.slider("Rentang Rating:", min_value=min_r, max_value=max_r, value=(min_r, max_r), step=0.1)

# Terapkan Filter Data
df_filtered = df[
    (df['Nama Produk'].str.contains(search_query, case=False, na=False)) &
    (df['Varian Kopi'].isin(selected_varians)) &
    (df['Asal/Lokasi Toko'].isin(selected_tokos)) &
    (df['Harga'].between(selected_harga[0], selected_harga[1])) &
    (df['Rating'].between(selected_rating[0], selected_rating[1]))
]

# 5. Summary KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Produk", f"{len(df_filtered):,} item")
c2.metric("Rata-rata Harga", f"Rp {df_filtered['Harga'].mean():,.0f}" if not df_filtered.empty else "Rp 0")
c3.metric("Rata-rata Rating", f"{df_filtered['Rating'].mean():.2f} ⭐" if not df_filtered.empty else "0")
c4.metric("Total Terjual", f"{df_filtered['Terjual'].sum():,.0f} unit" if not df_filtered.empty else "0")
c5.metric("Estimasi Omzet", f"Rp {df_filtered['Estimasi Omzet'].sum():,.0f}" if not df_filtered.empty else "Rp 0")

st.markdown("---")

# 6. Tab Navigasi Utama
tab1, tab2, tab3 = st.tabs(["📊 Model Regresi & Korelasi", "📈 Market & Toko Insight", "📋 Tabel Data & Top Seller"])

# TAB 1: REGRESI & KORELASI
with tab1:
    st.subheader("1. Model Regresi Linier Berganda")
    st.caption("Memprediksi variabel terikat (**Terjual**) berdasarkan variabel bebas (**Harga** dan **Rating**).")
    
    if len(df_filtered) > 5:
        X = df_filtered[['Harga', 'Rating']]
        Y = df_filtered['Terjual']
        
        model = LinearRegression()
        model.fit(X, Y)
        y_pred = model.predict(X)
        r2 = r2_score(Y, y_pred)
        
        col_reg1, col_reg2 = st.columns([1.2, 1])
        
        with col_reg1:
            st.markdown("### 📐 Persamaan Regresi")
            const = model.intercept_
            coef_harga = model.coef_[0]
            coef_rating = model.coef_[1]
            
            st.latex(rf"\hat{{Y}} = {const:.2f} + ({coef_harga:.6f}) \cdot \text{{Harga}} + ({coef_rating:.2f}) \cdot \text{{Rating}}")
            
            st.info(f"""
            **Ringkasan Koefisien:**
            * **R-Squared ($R^2$):** `{r2:.4f}`
            * **Koefisien Harga:** Kenaikan harga Rp 1.000 berdampak pada perubahan estimasi penjualan sebesar **{coef_harga * 1000:.2f} unit**.
            * **Koefisien Rating:** Kenaikan 0.1 poin rating berdampak pada perubahan estimasi penjualan sebesar **{coef_rating * 0.1:.2f} unit**.
            """)
            
        with col_reg2:
            st.markdown("### 🧮 Simulator Prediksi Penjualan")
            input_harga = st.number_input("Harga Produk (Rp):", min_value=1000, value=75000, step=5000)
            input_rating = st.slider("Rating Produk:", min_value=1.0, max_value=5.0, value=4.9, step=0.1)
            
            pred_terjual = model.predict([[input_harga, input_rating]])[0]
            pred_clean = max(0, int(round(pred_terjual)))
            
            st.success(f"📌 **Estimasi Penjualan:** ± **{pred_clean:,}** unit")

        st.markdown("---")
        st.subheader("2. Matriks Korelasi & Scatter Plot")
        
        col_sc1, col_sc2 = st.columns(2)
        
        with col_sc1:
            corr_matrix = df_filtered[['Harga', 'Rating', 'Terjual']].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".3f",
                color_continuous_scale="coolwarm",
                title="Matriks Korelasi Pearson"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_sc2:
            fig_scatter = px.scatter(
                df_filtered,
                x="Harga",
                y="Terjual",
                color="Varian Kopi",
                hover_data=["Nama Produk", "Rating", "Asal/Lokasi Toko"],
                title="Hubungan Harga vs Jumlah Terjual",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
    else:
        st.warning("Data terlalu sedikit untuk kalkulasi model regresi (minimal 5 data terfilter).")

# TAB 2: INSIGHT PASAR & TOKO
with tab2:
    st.subheader("📊 Distribusi Varian & Performa Toko")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        # Total Penjualan per Varian Kopi
        varian_sales = df_filtered.groupby("Varian Kopi")["Terjual"].sum().reset_index().sort_values(by="Terjual", ascending=False)
        fig_varian = px.bar(
            varian_sales,
            x="Terjual",
            y="Varian Kopi",
            orientation="h",
            color="Terjual",
            color_continuous_scale="Viridis",
            title="Total Terjual Berdasarkan Varian Kopi"
        )
        fig_varian.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_varian, use_container_width=True)
        
    with col_ins2:
        # Top 10 Toko Penjualan Terbanyak
        toko_sales = df_filtered.groupby("Asal/Lokasi Toko")["Terjual"].sum().reset_index().sort_values(by="Terjual", ascending=False).head(10)
        fig_toko = px.bar(
            toko_sales,
            x="Terjual",
            y="Asal/Lokasi Toko",
            orientation="h",
            color="Terjual",
            color_continuous_scale="Blues",
            title="Top 10 Toko/Seller dengan Penjualan Tertinggi"
        )
        fig_toko.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_toko, use_container_width=True)

# TAB 3: DATA TABLE & TOP SELLER
with tab3:
    st.subheader("🏆 Top 10 Produk Terlaris")
    top_10 = df_filtered.sort_values(by="Terjual", ascending=False).head(10)[['Nama Produk', 'Varian Kopi', 'Harga', 'Rating', 'Terjual', 'Asal/Lokasi Toko', 'Estimasi Omzet']]
    st.dataframe(top_10, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Seluruh Data Terfilter")
    st.dataframe(df_filtered, use_container_width=True)
    
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Terfilter (CSV)",
        data=csv_data,
        file_name="data_kopi_tokopedia_filtered.csv",
        mime="text/csv"
    )
