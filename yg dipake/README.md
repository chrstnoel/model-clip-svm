# Smart Retail Dual-Workflow System (CLIP Fine-Tuned & Multimodal SVM)

Repositori ini berisi implementasi sistem ritel cerdas berbasis **Dual-Workflow Architecture** yang mengintegrasikan dua model Machine Learning mutakhir:
1. **Alur 1 (Admin Ingestion)**: Mengklasifikasikan kategori produk secara otomatis dari gambar fisik dan nama produk menggunakan **Multimodal SVM (1024-dim)**.
2. **Alur 2 (User Dual Search)**: 
   - **Mode A (Text-to-Image)**: Pengguna mencari barang dengan teks bahasa alami (didukung translasi otomatis jika berbahasa Indonesia) untuk menemukan foto fisik produk yang relevan via Cosine Similarity CLIP.
   - **Mode B (Image-to-Image / Visual Reverse Search)**: Pengguna mengunggah foto produk langsung dari kamera/komputer, model mengekstrak vektor visualnya, lalu mencocokkan produk dengan kemiripan vektor visual tertinggi di database katalog.

Proyek ini menggunakan konfigurasi optimal dari pembersihan data Amazon Retail (**17 Kategori Bersih**) dan pelatihan skala besar (**3.400 data seimbang**), menghasilkan skor evaluasi yang sangat solid (**Top-1 Retrieval: 80.15%**, **Top-3 Retrieval: 91.91%**, **Top-3 Category Match: 91.88%**).

---

## 📁 Struktur Folder

```text
yg dipake/
├── api/
│   ├── inventory_service.py                 # Engine inti (Python class) untuk integrasi backend
│   └── server.py                            # REST API Microservice (FastAPI) siap pakai
├── data/
│   ├── manifest_3400.csv                    # 3.400 data latih seimbang (200 item/kategori)
│   └── catalog_balanced_340.csv             # Sampel katalog seimbang 17 kategori untuk inferensi
├── models/
│   ├── model_clip_17cat_projection_head/    # Bobot Fine-Tuned CLIP (Projection Head)
│   └── svm_multimodal_large_17cat.joblib    # Model SVM Multimodal (1024-dim, 17 Kategori)
├── plots/                                   # Visualisasi & grafik metrik evaluasi
│   ├── 01_svm_multimodal_confusion_matrix.png
│   ├── 02_perbandingan_akurasi.png
│   └── 03_clip_cosine_similarity.png
├── reports/
│   └── evaluation_report.md                 # Ringkasan ilmiah hasil metrik evaluasi
├── retail_production_pipeline.py            # Logika pipeline inferensi multimodal
├── uji_alur1_admin.py                       # Pengujian mandiri Alur 1 (Admin Ingestion)
├── uji_alur2_user.py                        # Pengujian mandiri Alur 2 (User Search)
├── demo_system.py                           # Demo interaktif command line (Admin & User)
└── README.md
```

---

## 📥 Setup Model Weights (Cloud Storage)

Karena file bobot model Machine Learning berukuran besar (> 40MB s.d. 577MB) dan tidak dapat dimasukkan langsung ke repositori GitHub, silakan unduh file bobot model melalui link Google Drive di bawah ini:

* **Model CLIP Fine-Tuned (577 MB)**:  
  👉 [Download model.safetensors via Google Drive](LINK_GOOGLE_DRIVE_CLIP_SAFETENSORS_ANDA)  
  *Simpan ke*: `yg dipake/models/model_clip_17cat_projection_head/model.safetensors`

* **Model SVM Multimodal 1024-dim (47 MB)**:  
  👉 [Download svm_multimodal_large_17cat.joblib via Google Drive](LINK_GOOGLE_DRIVE_SVM_JOBLIB_ANDA)  
  *Simpan ke*: `yg dipake/models/svm_multimodal_large_17cat.joblib`

---

## 📊 Hasil Evaluasi Model

### 1. Klasifikasi Kategori Produk (Model SVM Multimodal - Alur Admin)
| Metrik Evaluasi | SVM Baseline (21 Kat) | SVM Gambar (17 Kat Clean) | **SVM Multimodal 3.400 Data (Final)** | Peningkatan |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 CV Accuracy** | 36.43% | 39.56% | **72.29%** | **🔥 +32.73%** |
| **Top-3 CV Accuracy** | 80.24% | 84.71% | **91.88%** | **+7.17%** |
| **Top-5 CV Accuracy** | 93.57% | 95.88% | **95.94%** | **Sangat Stabil** |

### 2. Pencarian Lintas Modalitas / Benda Spesifik (Model CLIP - Alur Pengguna)
| Metrik Evaluasi | Arah Pencarian | Skenario Penggunaan | Akurasi Top-1 | Akurasi Top-3 |
| :--- | :--- | :--- | :---: | :---: |
| **Recall@1 (Top-1 Match)** | Text-to-Image / Image-to-Text | Mencocokkan query teks dengan foto produk fisik | **80.15%** | **91.91%** |
| **Recall@5 (Top-5 Match)** | Text-to-Image / Image-to-Text | Menemukan barang spesifik di antara 5 rekomendasi teratas | **96.32%** | **Sangat Akurat** |

---

## 🚀 Panduan Menjalankan

### 1. Menjalankan Demo Terminal
Untuk melihat simulasi Alur Admin dan Alur User secara langsung di terminal:
```bash
python "yg dipake/demo_system.py"
```

Atau jalankan skrip pengujian mandiri per alur:
```bash
# Uji coba Alur 1 (Sisi Admin)
python "yg dipake/uji_alur1_admin.py"

# Uji coba Alur 2 (Sisi Pengguna)
python "yg dipake/uji_alur2_user.py"
```

### 2. Mengintegrasikan ke Backend (Python Native)
Gunakan `InventoryService` langsung ke dalam kode backend Anda:
```python
from api.inventory_service import InventoryService
from PIL import Image

# 1. Inisialisasi Engine
service = InventoryService()

# 2. Alur Admin: Ingestion Produk Baru (Foto + Nama) -> Prediksi Kategori Otomatis
img = Image.open("foto_produk.jpg")
admin_result = service.add_product(image=img, title="Fender Electric Guitar")
print("Kategori Terpilih:", admin_result["predicted_category"])
print("Keyakinan Model :", admin_result["confidence_pct"], "%")

# 3. Alur User - Mode A: Pencarian Semantik via Teks (Mendukung Bahasa Indonesia)
user_text_result = service.search_products(query="kabel charger fast charging", top_k=5)
print("Top-1 Match (Text):", user_text_result["retrieved_products"][0]["title"])

# 4. Alur User - Mode B: Pencarian Berdasarkan Foto Produk (Image-to-Image)
query_image = Image.open("foto_sampel_user.jpg")
user_img_result = service.search_by_image(image=query_image, top_k=5)
print("Top-1 Match (Visual):", user_img_result["retrieved_products"][0]["title"])
print("Kemiripan Vektor   :", user_img_result["retrieved_products"][0]["match_percentage"], "%")
```

### 3. Menjalankan REST API Microservice (FastAPI)
Nyalakan server API backend untuk melayani aplikasi web atau mobile:
```bash
pip install fastapi uvicorn python-multipart
python "yg dipake/api/server.py"
```
Akses dokumentasi interaktif Swagger API di: `http://localhost:8000/docs`
* **POST `/admin/add-product`**: Menerima file gambar dan judul produk, lalu mengembalikan kategori terprediksi dan mendaftarkannya ke katalog.
* **POST `/user/search`**: Menerima query teks pengguna, menerjemahkannya secara otomatis, mendeteksi kategori, dan mengembalikan Top-K produk paling cocok.
* **POST `/user/search-image`**: Menerima file upload foto produk dari pengguna untuk mencocokkan kemiripan visual vektor ke foto katalog.
* **GET `/image/{product_id}`**: Endpoint untuk menampilkan foto fisik produk secara langsung.
* **GET `/search-ui`**: Antarmuka web visual interaktif untuk menguji pencarian via Teks maupun Upload Foto produk secara langsung di browser.
* **POST `/user/search`**: Menerima query teks pengguna, menerjemahkannya secara otomatis, mendeteksi kategori, dan mengembalikan Top-K produk paling cocok.
