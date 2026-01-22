# 🗑️ Smart Waste AI 2.0

**Klasifikasi Sampah Cerdas** menggunakan Deep Learning dengan ResNet50

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)

## 📋 Deskripsi

Aplikasi web interaktif untuk mengklasifikasikan sampah secara otomatis ke dalam 12 kategori menggunakan model deep learning ResNet50 yang telah dilatih dengan akurasi **95.74%**.

### ✨ Fitur Utama

- **📸 Kamera Real-time**: Ambil foto langsung dari webcam/kamera HP
- **📂 Upload File**: Upload gambar dari galeri
- **🎯 Akurasi Tinggi**: Model ResNet50 dengan akurasi 95.74%
- **📊 Visualisasi Interaktif**: Grafik probabilitas per kategori
- **🎨 UI Premium**: Desain modern dengan gradient dan animasi

### 🗂️ Kategori Sampah

1. Battery (Baterai)
2. Biological (Organik)
3. Brown Glass (Kaca Coklat)
4. Cardboard (Kardus)
5. Clothes (Pakaian)
6. Green Glass (Kaca Hijau)
7. Metal (Logam)
8. Paper (Kertas)
9. Plastic (Plastik)
10. Shoes (Sepatu)
11. Trash (Sampah Umum)
12. White Glass (Kaca Putih)

## 🚀 Instalasi

### Requirements

- Python 3.10+
- GPU (opsional, untuk inferensi lebih cepat)

### Langkah Instalasi

1. **Clone repository**:
```bash
git clone <your-repo-url>
cd streamlit_waste_classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi**:
```bash
streamlit run app.py
```

4. **Akses di browser**: `http://localhost:8501`

## 📁 Struktur File

```
streamlit_waste_classifier/
├── app.py                          # Aplikasi Streamlit
├── models/
│   └── waste_classifier_resnet50.h5  # Model ResNet50 (296 MB)
├── requirements.txt                # Dependencies
└── README.md                       # Dokumentasi
```

## 🎯 Cara Penggunaan

### Mode Kamera
1. Buka tab "📸 Ambil Foto (Kamera)"
2. Izinkan akses kamera di browser
3. Arahkan kamera ke objek sampah
4. Klik "Take Photo"
5. Lihat hasil klasifikasi

### Mode Upload
1. Buka tab "📂 Upload File (Galeri)"
2. Drag & drop atau browse gambar
3. Lihat hasil klasifikasi

## 🧠 Model Details

| Spesifikasi | Detail |
|------------|--------|
| **Arsitektur** | ResNet50 (Transfer Learning) |
| **Pre-trained** | ImageNet |
| **Parameters** | 23.5M |
| **Input Size** | 224x224 |
| **Accuracy** | 95.74% |
| **F1-Score (Macro)** | 0.94 |
| **Training Device** | NVIDIA Tesla P100 (Kaggle) |
| **Framework** | TensorFlow 2.15 + Keras |

## 📊 Performa Model

| Kategori | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Clothes | 1.00 | 0.99 | **0.99** |
| Shoes | 0.97 | 0.99 | **0.98** |
| Trash | 0.97 | 0.99 | **0.98** |
| Battery | 0.98 | 0.96 | 0.97 |
| Biological | 0.98 | 0.97 | 0.98 |
| Metal | 0.82 | 0.91 | 0.86 |
| Plastic | 0.85 | 0.87 | 0.86 |

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Custom CSS)
- **Backend**: TensorFlow/Keras
- **Model**: ResNet50 (Transfer Learning)
- **Visualization**: Plotly
- **Image Processing**: PIL, NumPy

## 📝 License

-

## 👤 Author

Andries Noermann Reynaldo Ratu  
Universitas Ciputra Surabaya

## 🙏 Acknowledgments

- Dataset: [Garbage Classification (Kaggle)](https://www.kaggle.com/datasets/...)
- Training Platform: Kaggle Notebooks (GPU P100)
- Framework: TensorFlow & Streamlit

---

**Note**: Model file (`waste_classifier_resnet50.h5`) berukuran ~296 MB. Pastikan koneksi internet stabil saat clone repository atau download model secara terpisah.
