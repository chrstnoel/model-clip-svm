"""
uji_alur1_admin.py
==================
Skrip Uji Coba Mandiri Sisi Admin (Ingestion & Auto-Categorization):
- Mengunggah foto produk baru + nama/judul produk.
- Menggunakan Model Terbaik: CLIP Projection Head + Multimodal SVM (17 Kategori Bersih).
- Menampilkan Top-3 kategori kandidat beserta persentase keyakinan.
"""

import os
import io
import zipfile
from PIL import Image
from retail_production_pipeline import RetailProductionSystem

def main():
    print("=" * 85)
    print(" 🏢 SISTEM PENGUJIAN ALUR 1: ADMIN INGESTION & AUTO-CATEGORIZATION")
    print("=" * 85)

    # Inisialisasi Sistem
    system = RetailProductionSystem(
        clip_weights_dir="yg dipake/models/model_clip_17cat_projection_head",
        svm_model_path="yg dipake/models/svm_multimodal_large_17cat.joblib",
        manifest_csv_path="yg dipake/data/manifest_3400.csv",
        zip_images_path="retail-products-classification.zip",
        device="cpu",
        preload_catalog_size=100
    )

    print("-" * 85)
    print(f"Total {len(system.categories)} Kategori Retail Bersih:")
    for i, cat in enumerate(system.categories, 1):
        print(f"  {i:2d}. {cat}")
    print("-" * 85)

    # Contoh produk uji coba
    print("\n🚀 MENJALANKAN DEMO PENGUJIAN PRODUK BARU OLEH ADMIN:\n")
    
    # Ambil sampel gambar dari zip untuk simulasi upload produk
    with zipfile.ZipFile("retail-products-classification.zip", 'r') as zf:
        # 1. Produk Gitar / Musik
        with zf.open("train/train/B0002E2GO2.jpg") as f:
            img_gitar = Image.open(io.BytesIO(f.read())).convert('RGB')
        
        # 2. Produk Pertukangan (Wood Lathe)
        with zf.open("train/train/B0000223A6.jpg") as f:
            img_tools = Image.open(io.BytesIO(f.read())).convert('RGB')

    test_samples = [
        {
            "title": "Fender 099-4912-000 Vintage-Style Telecaster Saddle Bridge",
            "image": img_gitar
        },
        {
            "title": "DELTA 46-700 12-Inch Variable Speed Wood Lathe Heavy Duty",
            "image": img_tools
        }
    ]

    for idx, sample in enumerate(test_samples, 1):
        print(f"[{idx}] ADMIN MENGUNGGAH:")
        print(f"    📦 Nama Produk : {sample['title']}")
        result = system.admin_add_product(sample['image'], sample['title'])
        print(f"    🎯 Kategori Terpilih : {result['predicted_category']} (Keyakinan: {result['confidence_pct']}%)")
        print("    📋 Top-3 Kandidat Kategori:")
        for cand in result['top_candidates']:
            print(f"       - Rank {cand['rank']}: {cand['category']} ({cand['confidence_pct']}%)")
        print(f"    ✅ Status : Berhasil didaftarkan ke katalog (Total item: {result['total_catalog_now']})\n")

if __name__ == "__main__":
    main()
