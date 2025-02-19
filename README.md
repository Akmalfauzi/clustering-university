# 🎓 Clustering Perguruan Tinggi Indonesia

## 📋 Deskripsi
Aplikasi ini merupakan sistem analisis clustering untuk perguruan tinggi di Indonesia yang menggunakan data dari API BPS. Aplikasi ini membantu dalam:
- Pengelompokan perguruan tinggi berdasarkan karakteristik serupa
- Analisis distribusi geografis institusi pendidikan tinggi
- Visualisasi interaktif data pendidikan tinggi
- Pemantauan tren dan perkembangan sektor pendidikan tinggi

### Fitur Utama:
1. **Data Real-time**
   - Integrasi langsung dengan API BPS
   - Data series dari 2018-2023
   - Update otomatis

2. **Analisis Clustering**
   - Algoritma K-Means
   - Optimasi jumlah cluster
   - Evaluasi menggunakan Silhouette Score

3. **Visualisasi Interaktif**
   - Peta distribusi geografis
   - Grafik perbandingan antar cluster
   - Dashboard monitoring

## 🛠️ Instalasi dan Penggunaan

### Kebutuhan Sistem
- Python 3.8 atau lebih tinggi
- Git
- Koneksi internet (untuk akses API BPS)
- Web browser modern

### Langkah Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/akmalfauzi/clustering-mts.git
   cd clustering-mts
   ```

2. **Setup Environment**
   ```bash
   # Buat virtual environment
   python -m venv venv

   # Aktivasi virtual environment
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi**
   ```bash
   streamlit run main.py
   ```

5. **Akses Aplikasi**
   - Buka browser dan akses `http://localhost:8501`
   - Aplikasi akan otomatis terbuka di browser default Anda

### Dependencies
```txt
streamlit==1.31.1
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
plotly==5.18.0
folium==0.15.1
seaborn==0.13.2
matplotlib==3.8.2
requests==2.31.0
streamlit-folium==0.17.0
```

## 📊 Metodologi
1. **Business Understanding**
   - Analisis kebutuhan
   - Penentuan tujuan clustering
   - Identifikasi metrik evaluasi

2. **Data Understanding**
   - Pengumpulan data dari API BPS
   - Eksplorasi data
   - Analisis statistik deskriptif

3. **Data Preparation**
   - Pembersihan data
   - Feature scaling
   - Penanganan missing values

4. **Modeling**
   - Implementasi K-Means clustering
   - Optimasi jumlah cluster
   - Validasi model

5. **Evaluation**
   - Silhouette Score analysis
   - Interpretasi hasil clustering
   - Visualisasi performa model

6. **Deployment**
   - Integrasi dengan web interface
   - Monitoring berkelanjutan
   - Pembaruan data otomatis

## 📈 Hasil
- Clustering optimal untuk pengelompokan perguruan tinggi
- Visualisasi distribusi geografis
- Analisis karakteristik setiap cluster
- Rekomendasi pengembangan berdasarkan hasil analisis

## 🔄 Maintenance
- Update model berkala
- Backup data rutin
- Monitoring performa
- Penyesuaian parameter

## ⚠️ Batasan
- Ketergantungan pada data historis
- Asumsi distribusi normal
- Sensitif terhadap outlier

### Troubleshooting

1. **Error saat instalasi requirements**
   ```bash
   # Update pip
   python -m pip install --upgrade pip
   # Coba install ulang
   pip install -r requirements.txt
   ```

2. **ModuleNotFoundError**
   - Pastikan virtual environment aktif
   - Cek instalasi package dengan `pip list`

3. **Error koneksi API**
   - Periksa koneksi internet
   - Pastikan API key masih valid
   - Coba restart aplikasi

4. **Error visualisasi**
   - Update browser ke versi terbaru
   - Clear cache browser
   - Restart aplikasi Streamlit

## 📝 Lisensi
[MIT License](LICENSE)

## 👥 Kontributor
- [Akmal Fauzi](https://github.com/akmalfauzi)

## 📧 Kontak
Untuk pertanyaan dan saran, silakan hubungi [akmalfauziofficial@gmail.com](mailto:akmalfauzi@gmail.com)