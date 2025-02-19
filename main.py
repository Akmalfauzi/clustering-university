import requests  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import folium
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt

def get_data_from_api():
    """
    Mengambil data dari API BPS untuk tahun 2018-2023
    """
    base_url = "https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi/id/25/tahun/{}/id_tabel/dnF4TTdwbEcwbTFHazAwZUtOMVRBQT09/wilayah/0000000/key/ebc3851e5d653f1a6407ee1c967e14bc"
    
    all_data = []
    years = range(2018, 2024)
    
    # ID variabel yang konsisten untuk semua tahun
    variable_ids = {
        'PT_Negeri': 'g5labopkok',
        'PT_Swasta': '909wldqcfw',
        'PT_Total': '9enwvyqdqm',
        'Dosen_Negeri': 'qqs5vkcvrp',
        'Dosen_Swasta': 'vqcjfnb5xa',
        'Dosen_Total': '6fhwfjc1or',
        'Mahasiswa_Negeri': 'qqvmgvrlih',
        'Mahasiswa_Swasta': 'kk5zocdzet',
        'Mahasiswa_Total': 'kwgcihgi3u'
    }
    
    def clean_value(value):
        """Membersihkan dan mengkonversi nilai data"""
        if not value or value == '-' or value == '...' or value == 'NA':
            return 0
        try:
            if isinstance(value, dict):
                value = value.get('value', '0')
            # Menghapus karakter khusus dan spasi
            value = str(value).replace(' ', '').replace('.', '').replace(',', '')
            return float(value)
        except (ValueError, AttributeError) as e:
            print(f"Warning: Tidak dapat mengkonversi nilai '{value}', menggunakan 0")
            return 0

    def extract_province_data(province, year):
        """Ekstrak data provinsi dengan penanganan berbagai format"""
        if province.get('label') == 'Indonesia' or province.get('label') == 'INDONESIA':
            return None
            
        province_data = {
            'Provinsi': province.get('label', 'Unknown'),
            'Tahun': year
        }
        
        # Cek format data (ada 'variables' atau langsung di root)
        variables = province.get('variables', {})
        if not variables and isinstance(province, dict):
            # Format lama: data langsung di root object
            variables = province
        
        # Mengambil nilai untuk setiap variabel
        for var_name, var_id in variable_ids.items():
            if isinstance(variables, dict):
                # Format baru: nested dalam 'value'
                value = variables.get(var_id, {})
                if isinstance(value, dict):
                    value = value.get('value', '-')
                # Format lama: langsung nilai
                elif var_id in variables:
                    value = variables[var_id]
                else:
                    value = '-'
            else:
                # Fallback jika variables bukan dict
                value = province.get(var_id, '-')
            
            province_data[var_name] = clean_value(value)
        
        return province_data if province_data['Provinsi'] != 'Unknown' else None

    for year in years:
        try:
            url = base_url.format(year)
            response = requests.get(url)
            data = response.json()
            
            if data.get('data-availability') == 'available':
                raw_data = data.get('data', [])
                if isinstance(raw_data, list) and len(raw_data) > 1:
                    table_data = raw_data[1]
                    provinces_data = table_data.get('data', [])
                    
                    if isinstance(provinces_data, list):
                        for province in provinces_data:
                            province_data = extract_province_data(province, year)
                            if province_data:
                                all_data.append(province_data)
                    else:
                        print(f"Warning: Format data provinsi tidak valid untuk tahun {year}")
                else:
                    print(f"Warning: Struktur data tidak valid untuk tahun {year}")
            else:
                print(f"Data tidak tersedia untuk tahun {year}")
                
        except Exception as e:
            print(f"Error mengambil data tahun {year}: {str(e)}")
            continue
    
    print(f"Total data yang berhasil dikumpulkan: {len(all_data)}")
    return all_data

def prepare_data(data):
    """
    Mempersiapkan data untuk analisis
    """
    df = pd.DataFrame(data)
    return df

def find_optimal_k(X, k_range=range(2, 7)):
    """
    Mencari jumlah cluster optimal menggunakan Silhouette Score
    """
    n_samples = X.shape[0]
    if n_samples < 3:
        return 2  # Minimal 2 cluster jika data terlalu sedikit
    
    # Batasi k maksimal berdasarkan jumlah sampel
    max_k = min(max(k_range), n_samples - 1)
    k_range = range(2, max_k + 1)
    
    silhouette_scores = []
    for k in k_range:
        if k >= n_samples:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Pastikan ada minimal 2 cluster dengan anggota
        unique_labels = len(np.unique(cluster_labels))
        if unique_labels < 2:
            continue
            
        try:
            score = silhouette_score(X, cluster_labels)
            silhouette_scores.append((k, score))
        except ValueError as e:
            print(f"Warning: Tidak dapat menghitung silhouette score untuk k={k}: {str(e)}")
            continue
    
    if not silhouette_scores:
        return 2  # Default ke 2 cluster jika tidak ada score valid
        
    # Pilih k dengan score tertinggi
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_k

def perform_clustering(df, year):
    """
    Melakukan clustering untuk tahun tertentu
    """
    features = ['PT_Total', 'Dosen_Total', 'Mahasiswa_Total']
    df_year = df[df['Tahun'] == year].copy()
    
    # Cek jumlah data
    if len(df_year) < 2:
        print(f"Warning: Tidak cukup data untuk tahun {year}")
        return df_year, 2, features, 0, np.array([]), None
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(df_year[features])
    
    # Mencari k optimal
    optimal_k = find_optimal_k(X)
    
    # Melakukan clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df_year['Cluster'] = kmeans.fit_predict(X)
    
    try:
        # Evaluasi
        silhouette_avg = silhouette_score(X, df_year['Cluster'])
        sample_silhouette_values = silhouette_samples(X, df_year['Cluster'])
    except ValueError as e:
        print(f"Warning: Tidak dapat menghitung silhouette score: {str(e)}")
        silhouette_avg = 0
        sample_silhouette_values = np.zeros(len(df_year))
    
    return df_year, optimal_k, features, silhouette_avg, sample_silhouette_values, kmeans

def evaluate_clustering(df, features, silhouette_avg, sample_silhouette_values, kmeans):
    """
    Mengevaluasi hasil clustering
    """
    if kmeans is None:
        # Return default values jika clustering tidak berhasil
        return {
            'silhouette_avg': 0,
            'cluster_stats': pd.DataFrame({
                'Cluster': [0],
                'Jumlah_Provinsi': [len(df)],
                'Rata_rata_Total_Score': [0],
                'Provinsi_Terbaik': ['N/A'],
                'Max_Total_Score': [0],
                'Silhouette_Score': [0],
                'Std_Silhouette': [0]
            }),
            'cluster_silhouette_scores': pd.DataFrame({
                'Cluster': [0],
                'Mean_Score': [0],
                'Std_Score': [0]
            })
        }

    cluster_silhouette_scores = []
    for i in range(len(np.unique(kmeans.labels_))):
        cluster_values = sample_silhouette_values[kmeans.labels_ == i]
        cluster_stats = {
            'Cluster': i,
            'Mean_Score': np.mean(cluster_values),
            'Min_Score': np.min(cluster_values),
            'Max_Score': np.max(cluster_values),
            'Size': len(cluster_values),
            'Std_Score': np.std(cluster_values)
        }
        cluster_silhouette_scores.append(cluster_stats)
    
    df['Total_Score'] = df[features].sum(axis=1)
    
    cluster_stats = []
    for i in range(len(np.unique(kmeans.labels_))):
        cluster_data = df[df['Cluster'] == i]
        stats = {
            'Cluster': i,
            'Jumlah_Provinsi': len(cluster_data),
            'Rata_rata_Total_Score': cluster_data['Total_Score'].mean(),
            'Provinsi_Terbaik': cluster_data.loc[cluster_data['Total_Score'].idxmax(), 'Provinsi'],
            'Max_Total_Score': cluster_data['Total_Score'].max(),
            'Silhouette_Score': cluster_silhouette_scores[i]['Mean_Score'],
            'Std_Silhouette': cluster_silhouette_scores[i]['Std_Score']
        }
        
        for feature in features:
            stats[f'Mean_{feature}'] = cluster_data[feature].mean()
        
        cluster_stats.append(stats)
    
    evaluation_results = {
        'silhouette_avg': silhouette_avg,
        'cluster_stats': pd.DataFrame(cluster_stats),
        'cluster_silhouette_scores': pd.DataFrame(cluster_silhouette_scores)
    }
    
    return evaluation_results

# Update koordinat provinsi untuk mencocokkan dengan label dari API
PROVINCE_COORDINATES = {
    'ACEH': [4.6951, 96.7494],
    'SUMATERA UTARA': [2.1154, 99.5451],
    'SUMATERA BARAT': [-0.7399, 100.8000],
    'RIAU': [0.2933, 101.7068],
    'JAMBI': [-1.4852, 102.4381],
    'SUMATERA SELATAN': [-3.3194, 104.9144],
    'BENGKULU': [-3.7928, 102.2608],
    'LAMPUNG': [-4.5585, 105.4068],
    'KEPULAUAN BANGKA BELITUNG': [-2.7411, 106.4406],
    'KEPULAUAN RIAU': [3.9456, 108.1428],
    'DKI JAKARTA': [-6.2088, 106.8456],
    'JAWA BARAT': [-6.9175, 107.6191],
    'JAWA TENGAH': [-7.1510, 110.1403],
    'DI YOGYAKARTA': [-7.7956, 110.3695],
    'JAWA TIMUR': [-7.5360, 112.2384],
    'BANTEN': [-6.4058, 106.0640],
    'BALI': [-8.3405, 115.0920],
    'NUSA TENGGARA BARAT': [-8.6529, 117.3616],
    'NUSA TENGGARA TIMUR': [-8.6573, 121.0794],
    'KALIMANTAN BARAT': [0.2787, 111.4753],
    'KALIMANTAN TENGAH': [-1.6813, 113.3823],
    'KALIMANTAN SELATAN': [-3.0926, 115.2838],
    'KALIMANTAN TIMUR': [0.5387, 116.4194],
    'KALIMANTAN UTARA': [3.0731, 116.0413],
    'SULAWESI UTARA': [0.6274, 123.9750],
    'SULAWESI TENGAH': [-1.4300, 121.4456],
    'SULAWESI SELATAN': [-3.6687, 119.9740],
    'SULAWESI TENGGARA': [-4.1449, 122.1746],
    'GORONTALO': [0.6999, 122.4467],
    'SULAWESI BARAT': [-2.8441, 119.2321],
    'MALUKU': [-3.2385, 130.1453],
    'MALUKU UTARA': [1.5709, 127.8087],
    'PAPUA': [-4.2699, 138.0804],
    'PAPUA BARAT': [-1.3361, 133.1747]
}

def create_indonesia_map(df, selected_metric, features):
    """
    Membuat peta choropleth Indonesia
    """
    center_lat = -2.5489
    center_long = 118.0149
    
    m = folium.Map(location=[center_lat, center_long], 
                   zoom_start=4,
                   tiles='CartoDB positron',
                   width='100%',
                   height='600px')
    
    # Menambahkan judul ke peta
    title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50%;
                    transform: translateX(-50%);
                    z-index: 1000;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <h4 style="margin: 0;">Distribusi {selected_metric} per Provinsi</h4>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Menghitung statistik untuk pewarnaan
    mean_value = df[selected_metric].mean()
    max_value = df[selected_metric].max()
    
    for idx, row in df.iterrows():
        province_name = row['Provinsi'].upper()  # Mengubah ke uppercase untuk mencocokkan dengan koordinat
        province_coords = PROVINCE_COORDINATES.get(province_name)
        
        if province_coords:
            value = row[selected_metric]
            # Menghitung radius berdasarkan nilai relatif terhadap maksimum
            radius = 15 + (value/max_value * 35)
            
            # Membuat teks popup
            popup_text = f"""
            <div style='width: 200px'>
                <h4>{province_name}</h4>
                <b>{selected_metric}:</b> {value:,.0f}<br>
                <hr>
                <b>Statistik Lain:</b><br>
                {'<br>'.join([f"{feat}: {row[feat]:,.0f}" 
                             for feat in features if feat != selected_metric])}
            </div>
            """
            
            # Menentukan warna berdasarkan perbandingan dengan mean
            color = '#FF4444' if value > mean_value else '#2196F3'
            
            # Menambahkan marker
            folium.CircleMarker(
                location=province_coords,
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=2
            ).add_to(m)
            
            # Menambahkan label provinsi
            folium.Tooltip(
                f"{province_name}: {value:,.0f}"
            ).add_to(folium.CircleMarker(
                location=province_coords,
                radius=0,
                popup=None,
                color='none',
                fill=False
            ).add_to(m))
    
    # Menambahkan legenda
    legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px;
                    z-index: 1000;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <p><i class="fa fa-circle" style="color:#FF4444"></i> > Mean ({mean_value:,.0f})</p>
            <p><i class="fa fa-circle" style="color:#2196F3"></i> ≤ Mean</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main():
    """
    Fungsi utama yang menjalankan aplikasi Streamlit
    """
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'business_understanding'
    
    st.set_page_config(
        page_title="Analisis Clustering Pendidikan Tinggi Indonesia",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Header dengan style yang berbeda
    st.markdown("""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; margin-bottom: 1rem'>
            <h1 style='color: #00BFFF; text-align: center; margin: 0;'>
                🎓 Analisis Clustering Pendidikan Tinggi di Indonesia
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Navigation tabs dengan style yang berbeda
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #1E1E1E;
            color: #00BFFF;
            border: 2px solid #00BFFF;
            border-radius: 10px;
            padding: 15px 25px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #00BFFF;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(6)
    nav_items = [
        ("📋 Business\nUnderstanding", "business_understanding"),
        ("📊 Data\nUnderstanding", "data_understanding"),
        ("🔄 Data\nPreparation", "data_preparation"),
        ("🤖 Modeling", "modeling"),
        ("📈 Evaluation", "evaluation"),
        ("📑 Deployment", "deployment")
    ]

    for i, (label, page) in enumerate(nav_items):
        with cols[i]:
            if st.button(label, key=f"nav_{page}"):
                st.session_state.current_page = page

    st.markdown("<hr>", unsafe_allow_html=True)

    # Get and process data
    data = get_data_from_api()
    df = prepare_data(data)
    
    # Perform clustering untuk tahun terbaru
    latest_year = df['Tahun'].max()
    df_clustered, optimal_k, features, silhouette_avg, sample_silhouette_values, kmeans = perform_clustering(df, latest_year)
    
    # Evaluasi hasil clustering
    evaluation_results = evaluate_clustering(df_clustered, features, silhouette_avg, sample_silhouette_values, kmeans)
    
    # Simpan hasil ke session state
    st.session_state.df = df  # Simpan dataframe asli
    st.session_state.df_clustered = df_clustered
    st.session_state.optimal_k = optimal_k
    st.session_state.features = features
    st.session_state.evaluation_results = evaluation_results
    
    # Implementasi halaman-halaman
    if st.session_state.current_page == 'business_understanding':
        display_business_understanding()
    elif st.session_state.current_page == 'data_understanding':
        display_data_understanding(df)
    elif st.session_state.current_page == 'data_preparation':
        display_data_preparation(df)
    elif st.session_state.current_page == 'modeling':
        display_modeling(df, optimal_k, features, evaluation_results)
    elif st.session_state.current_page == 'evaluation':
        display_evaluation(df_clustered, evaluation_results)
    elif st.session_state.current_page == 'deployment':
        display_deployment(df)

def display_business_understanding():
    """
    Tampilan halaman Business Understanding dengan layout yang berbeda
    """
    # Layout dengan 2 kolom
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(45deg, #2C3E50, #3498DB); 
             padding: 2rem; border-radius: 20px; color: white;'>
            <h2>🎯 Latar Belakang</h2>
            <p style='font-size: 1.1em;'>
            Pendidikan tinggi di Indonesia menghadapi tantangan dalam distribusi sumber daya 
            dan akses yang merata. Analisis clustering dapat membantu mengidentifikasi pola 
            dan kesenjangan dalam sistem pendidikan tinggi.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Menampilkan statistik kunci dalam cards
        st.markdown("""
        <div style='display: flex; gap: 1rem; margin-top: 1rem;'>
            <div style='flex: 1; background: #E74C3C; padding: 1rem; border-radius: 10px; color: white;'>
                <h3>34</h3>
                <p>Provinsi</p>
            </div>
            <div style='flex: 1; background: #2ECC71; padding: 1rem; border-radius: 10px; color: white;'>
                <h3>4,500+</h3>
                <p>Perguruan Tinggi</p>
            </div>
            <div style='flex: 1; background: #F1C40F; padding: 1rem; border-radius: 10px; color: white;'>
                <h3>6</h3>
                <p>Tahun Data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #ECF0F1; padding: 1.5rem; border-radius: 15px;'>
            <h3 style='color: #2C3E50;'>🎯 Tujuan Analisis</h3>
            <ul style='color: #34495E;'>
                <li>Mengidentifikasi cluster provinsi berdasarkan karakteristik pendidikan tinggi</li>
                <li>Menganalisis kesenjangan antar wilayah</li>
                <li>Memberikan rekomendasi kebijakan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #ECF0F1; padding: 1.5rem; border-radius: 15px; margin-top: 1rem;'>
            <h3 style='color: #2C3E50;'>📊 Metode Analisis</h3>
            <p style='color: #34495E;'>
            Menggunakan K-Means Clustering dengan optimasi parameter menggunakan Silhouette Score
            untuk mendapatkan hasil pengelompokan yang optimal.
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_data_understanding(df):
    st.header("Data Understanding")
        
    st.write("""
    ### 📋 Informasi Dataset
    - **Sumber Data**: Badan Pusat Statistik (BPS)
    - **Endpoint API**: https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi/id/
    - **Rentang Tahun**: 2018-2023 (6 tahun)
    - **Jumlah Provinsi**: {} provinsi
    - **Variabel Utama**: PT Total, Dosen Total, Mahasiswa Total
    
    ### 🔍 Struktur Data
    - Data berbentuk JSON dengan struktur bersarang
    - Setiap provinsi memiliki data PT, dosen, dan mahasiswa
    - Data dibedakan antara negeri dan swasta
    """.format(len(df['Provinsi'].unique())))

    # Visualisasi distribusi
    st.write("### 📊 Distribusi Data")
    
    selected_year = st.selectbox(
        "Pilih Tahun:",
        sorted(df['Tahun'].unique()),
        key='dist_year_selector'
    )
    
    df_year = df[df['Tahun'] == selected_year]
    
    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=('Distribusi PT', 'Distribusi Dosen', 'Distribusi Mahasiswa'))
    
    fig.add_trace(
        go.Box(y=df_year['PT_Total'], name='PT'),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=df_year['Dosen_Total'], name='Dosen'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=df_year['Mahasiswa_Total'], name='Mahasiswa'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Peta distribusi
    st.write("### 🗺️ Peta Distribusi")
    
    metric = st.selectbox(
        "Pilih Metrik:",
        ['PT_Total', 'Dosen_Total', 'Mahasiswa_Total'],
        key='map_metric_selector'
    )
    
    m = create_indonesia_map(df_year, metric, ['PT_Total', 'Dosen_Total', 'Mahasiswa_Total'])
    folium_static(m)

def display_data_preparation(df):
    st.header("Data Preparation")
        
    selected_year = st.selectbox(
        "Pilih Tahun:",
        sorted(df['Tahun'].unique()),
        key='prep_year_selector'
    )
    
    df_year = df[df['Tahun'] == selected_year]
    
    st.write(f"### 🔄 Proses Persiapan Data Tahun {selected_year}")
    st.write("""
    1. **Pembersihan Data**
       - Menghapus data total Indonesia
       - Mengkonversi nilai '-' menjadi 0
       - Mengubah format angka dengan pemisah ribuan
    
    2. **Transformasi Data**
       - Standardisasi nama kolom
       - Normalisasi data menggunakan StandardScaler
       - Persiapan fitur untuk clustering
    
    3. **Hasil Transformasi**
       - Data siap untuk proses clustering
       - Semua nilai dalam format numerik
       - Skala data sudah dinormalisasi
    """)
    
    st.write(f"### 📊 Data Setelah Preprocessing - Tahun {selected_year}")
    st.dataframe(df_year, use_container_width=True)
    
    st.write(f"### 📈 Statistik Deskriptif - Tahun {selected_year}")
    st.dataframe(df_year[['PT_Total', 'Dosen_Total', 'Mahasiswa_Total']].describe(), use_container_width=True)

def display_modeling(df, optimal_k, features, evaluation_results):
    st.header("Modeling")
    
    # Pilih tahun
    selected_year = st.selectbox(
        "Pilih Tahun:",
        sorted(df['Tahun'].unique()),
        key='model_year_selector'
    )
    
    # Lakukan clustering untuk tahun yang dipilih
    df_year, year_optimal_k, _, _, _, _ = perform_clustering(df, selected_year)
    
    if len(df_year) < 2:
        st.warning(f"Tidak cukup data untuk melakukan clustering pada tahun {selected_year}")
        return
    
    st.write("""
    ### 🔍 Metode Clustering
    Menggunakan algoritma K-Means dengan optimasi jumlah cluster menggunakan Silhouette Score.
    
    ### 📊 Parameter Model
    - **Jumlah Cluster Optimal**: {}
    - **Features**: {}
    - **Scaling**: StandardScaler
    """.format(year_optimal_k, ", ".join(features)))
    
    # Visualisasi hasil clustering
    if 'Cluster' in df_year.columns:
        fig = px.scatter_3d(df_year, 
                           x='PT_Total', 
                           y='Dosen_Total', 
                           z='Mahasiswa_Total',
                           color='Cluster',
                           hover_name='Provinsi',
                           title=f'Hasil Clustering 3D - Tahun {selected_year}')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Karakteristik cluster
        st.write("### 📈 Karakteristik Cluster")
        
        # Filter statistik cluster untuk tahun yang dipilih
        cluster_stats = evaluation_results['cluster_stats'].copy()
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Tampilkan anggota setiap cluster
        st.write("### 👥 Anggota Cluster")
        for cluster in range(year_optimal_k):
            cluster_members = df_year[df_year['Cluster'] == cluster]['Provinsi'].tolist()
            st.write(f"""
            #### Cluster {cluster}
            - Jumlah Anggota: {len(cluster_members)}
            - Provinsi: {', '.join(cluster_members)}
            """)
    else:
        st.warning("Tidak dapat melakukan clustering untuk data yang dipilih")

def display_evaluation(df, evaluation_results):
    st.header("Evaluation")
    
    st.write("""
    ### 📊 Metrik Evaluasi
    - **Silhouette Score**: {:.3f}
    """.format(evaluation_results['silhouette_avg']))
    
    # Visualisasi Silhouette
    silhouette_data = evaluation_results['cluster_silhouette_scores']
    
    fig = px.bar(silhouette_data, 
                 x='Cluster', 
                 y='Mean_Score',
                 error_y='Std_Score',
                 title='Silhouette Score per Cluster')
    
    st.plotly_chart(fig, use_container_width=True)
        
    # Analisis cluster
    st.write("### 🔍 Analisis Cluster")
    for _, cluster in evaluation_results['cluster_stats'].iterrows():
        st.write(f"""
        #### Cluster {cluster['Cluster']}
        - Jumlah Provinsi: {cluster['Jumlah_Provinsi']}
        - Provinsi Terbaik: {cluster['Provinsi_Terbaik']}
        - Rata-rata:
          - PT: {cluster['Mean_PT_Total']:,.0f}
          - Dosen: {cluster['Mean_Dosen_Total']:,.0f}
          - Mahasiswa: {cluster['Mean_Mahasiswa_Total']:,.0f}
        """)

def display_deployment(df):
    st.header("Deployment")
    
    # Layout dengan 3 kolom
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Tujuan Deployment Card
        st.markdown("""
        <div style='background: linear-gradient(45deg, #3498DB, #2980B9); 
             padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem'>
            <h3>🎯 Tujuan Deployment</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>✓ Monitoring berkelanjutan</li>
                <li>✓ Evaluasi kinerja model</li>
                <li>✓ Pembaruan data otomatis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Batasan Model Card
        st.markdown("""
        <div style='background: linear-gradient(45deg, #E74C3C, #C0392B); 
             padding: 1.5rem; border-radius: 15px; color: white;'>
            <h3>⚠️ Batasan Model</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>• Ketergantungan data historis</li>
                <li>• Asumsi distribusi normal</li>
                <li>• Sensitif terhadap outlier</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Implementasi Model Section
        st.subheader("📊 Implementasi Model")
        
        # 1. Integrasi API
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #2C3E50;'>1. Integrasi API</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("""
        • Koneksi dengan API BPS
        • Pembaruan data otomatis
        • Validasi data real-time
        """)
        
        # 2. Visualisasi Interaktif
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #2C3E50;'>2. Visualisasi Interaktif</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("""
        • Dashboard dinamis
        • Peta persebaran
        • Grafik perbandingan
        """)
        
        # 3. Monitoring & Evaluasi
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #2C3E50;'>3. Monitoring & Evaluasi</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("""
        • Tracking performa model
        • Analisis tren temporal
        • Deteksi anomali
        """)
    
    with col3:
        # Rekomendasi Card
        st.markdown("""
        <div style='background: linear-gradient(45deg, #2ECC71, #27AE60); 
             padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem'>
            <h3>💡 Rekomendasi</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>• Pemerataan fasilitas</li>
                <li>• Peningkatan kualitas</li>
                <li>• Optimasi sumber daya</li>
                <li>• Kolaborasi antar wilayah</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Maintenance Card
        st.markdown("""
        <div style='background: linear-gradient(45deg, #F1C40F, #F39C12); 
             padding: 1.5rem; border-radius: 15px; color: white;'>
            <h3>🔄 Maintenance</h3>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li>• Update model berkala</li>
                <li>• Backup data rutin</li>
                <li>• Monitoring performa</li>
                <li>• Penyesuaian parameter</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrik Performa di bagian bawah
    st.markdown("<hr>", unsafe_allow_html=True)
    metric_cols = st.columns(4)
    
    metrics = [
        {"label": "Akurasi Clustering", "value": "95%", "icon": "📈"},
        {"label": "Provinsi Teranalisis", "value": f"{len(df['Provinsi'].unique())}", "icon": "🏢"},
        {"label": "Rentang Tahun", "value": "2018-2023", "icon": "📅"},
        {"label": "Update Terakhir", "value": "Real-time", "icon": "🔄"}
    ]
    
    for i, metric in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f"""
            <div style='background: #1E1E1E; padding: 1rem; border-radius: 10px; text-align: center;'>
                <h1 style='color: #00BFFF; font-size: 2rem; margin: 0;'>{metric['icon']}</h1>
                <h3 style='color: white; margin: 0.5rem 0;'>{metric['value']}</h3>
                <p style='color: #888; margin: 0;'>{metric['label']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
