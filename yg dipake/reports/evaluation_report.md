# Laporan Hasil Evaluasi Model (Dual-Workflow Retail System)

Dokumen ini merangkum hasil evaluasi ilmiah dari integrasi dua arsitektur model utama:
1. **Multimodal Linear SVM (1024-Dimensi)** untuk klasifikasi kategori produk (Alur Admin).
2. **Fine-Tuned CLIP Projection Head (512-Dimensi)** untuk pencarian semantik teks-ke-gambar lintas modalitas (Alur User).

---

## 1. Evaluasi Model SVM (Klasifikasi Kategori Produk)

Dilatih menggunakan **3.400 data seimbang (200 sampel/kategori pada 17 kategori bersih Amazon)** dengan validasi **5-Fold Stratified Cross-Validation**.

| Metrik Evaluasi | SVM Baseline (21 Kat) | SVM Gambar (17 Kat Clean) | **SVM Multimodal 3.400 Data (Final)** | Peningkatan |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 CV Accuracy** | 36.43% | 39.56% | **72.29%** | **🔥 +32.73%** |
| **Top-3 CV Accuracy** | 80.24% | 84.71% | **91.88%** | **+7.17%** |
| **Top-5 CV Accuracy** | 93.57% | 95.88% | **95.94%** | **Sangat Stabil** |

---

## 2. Evaluasi Model CLIP (Pencarian & Pencocokan Benda Spesifik)

Dilatih dengan membekukan (*freeze*) 151M parameter backbone transformer dan hanya melatih *visual & text projection heads* (0.43% parameter) pada learning rate 5e-5.

| Metrik Evaluasi | Target Prediksi | Skenario Alur | Akurasi Top-1 | Akurasi Top-3 |
| :--- | :--- | :--- | :---: | :---: |
| **Image-to-Text Retrieval** | Menemukan teks judul produk dari foto fisik | Verifikasi metadata | **80.15%** | **91.91%** |
| **Text-to-Image Retrieval** | Menemukan foto produk fisik dari query teks user | Pencarian katalog user | **80.15%** | **91.91%** |
