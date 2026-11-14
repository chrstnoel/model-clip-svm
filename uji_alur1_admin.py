"""
uji_alur1_admin.py
==================
File Pengujian Mandiri Khusus ALUR 1 (SISI ADMIN):
- Mengunggah foto produk baru + memasukkan judul/nama produk.
- Model CLIP mengubah foto menjadi vektor visual 512 dimensi.
- Model SVM otomatis menganalisis dan memprediksi produk ini paling cocok masuk kategori mana.
"""

import os
import sys
from PIL import Image
from dormiy_production_pipeline import DormiyRetailSystem

def main():
    print("=" * 80)
    print(" 🏢 SISTEM PENGUJIAN ALUR 1: ADMIN PRODUCT INGESTION & AUTO-CATEGORIZATION")
    print("=" * 80)

    # 1. Inisialisasi Sistem
    system = DormiyRetailSystem(
        clip_weights_dir="experiment_results/sesi_3_projection_head/model_clip_dormiy_sesi3",
        svm_model_path="experiment_results/sesi_augmented_svm/svm_augmented_classifier.joblib",
        manifest_csv_path="DORMIY-ML-Dataset/data/final/cv/dormiy_cv_manifest.csv",
        device="cpu"
    )

    print("-" * 80)
    print("Daftar 5 Kategori Ritel yang Dikenali Sistem:")
    for i, cat in enumerate(system.categories, 1):
        print(f"  {i}. {cat}")
    print("-" * 80)

    # 2. Ambil contoh gambar untuk diuji sebagai produk baru
    # Kita uji dengan 3 sampel produk nyata dari katalog
    sample_tests = [
        {
            "img_path": "DORMIY-ML-Dataset/data/final/cv/images/prod_0132.jpg",
            "title": "Pisau Dapur Stainless Steel Serbaguna 13cm",
            "kategori_ekspektasi": "Home & Garden"
        },
        {
            "img_path": "DORMIY-ML-Dataset/data/final/cv/images/prod_0803.jpg",
            "title": "Mata Bor Tembok Beton Masonry Flute Heavy Duty",
            "kategori_ekspektasi": "Hardware"
        },
        {
            "img_path": "DORMIY-ML-Dataset/data/final/cv/images/prod_0866.jpg",
            "title": "Mini 3D Desktop Printer Machine High Precision",
            "kategori_ekspektasi": "Electronics"
        }
    ]

    print("\n🚀 MEMULAI UJI COBA OTOMATIS 3 PRODUK MASUK KE ADMIN:\n")

    for i, test in enumerate(sample_tests, 1):
        img_path = test["img_path"]
        title = test["title"]
        ekspektasi = test["kategori_ekspektasi"]

        print(f"[{i}] ADMIN INPUT:")
        print(f"    • Foto Barang : {img_path}")
        print(f"    • Nama Barang : \"{title}\"")

        # Jalankan Fungsi Alur 1
        result = system.admin_add_new_product(
            image_input=img_path,
            product_title=title,
            save_to_catalog=False  # Simulasi uji coba
        )

        pred_cat = result["svm_category_prediction"]["recommended_category"]
        conf = result["svm_category_prediction"]["confidence_pct"]
        sec_cat = result["svm_category_prediction"]["secondary_category"]
        sec_conf = result["svm_category_prediction"]["secondary_confidence_pct"]

        print(f"    🎯 HASIL REKOMENDASI KATEGORI SVM:")
        print(f"       1. Kategori Utama   : {pred_cat} (Keyakinan: {conf}%)")
        print(f"       2. Kategori Cadangan: {sec_cat} (Keyakinan: {sec_conf}%)")
        print(f"    ✅ Status: {'SANGAT TEPAT' if pred_cat == ekspektasi else 'MASUK KATEGORI CADANGAN'}")
        print("-" * 80)

    print("\n💡 ANDA BISA MENGUJI FOTO SENDIRI:")
    print("Cukup panggil di Python:")
    print("result = system.admin_add_new_product('path/ke/foto_anda.jpg', 'Nama Produk Anda')")
    print("=" * 80)

if __name__ == "__main__":
    main()

