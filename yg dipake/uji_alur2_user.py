"""
uji_alur2_user.py
=================
Skrip Uji Coba Mandiri Sisi Pengguna (Semantic Search & Top-5 Recommendation):
- Menerima kata kunci pencarian bebas (Bahasa Indonesia / Inggris).
- Otomatis menerjemahkan query ke bahasa Inggris.
- Mendeteksi Top-3 kategori terkait dengan SVM.
- Membandingkan query teks dengan foto produk katalog menggunakan Cosine Similarity.
- Menampilkan Top-5 produk paling cocok.
"""

import os
import io
import zipfile
from PIL import Image
from retail_production_pipeline import RetailProductionSystem

def main():
    print("=" * 85)
    print(" 🛒 SISTEM PENGUJIAN ALUR 2: USER DUAL SEARCH (TEXT & IMAGE RETRIEVAL)")
    print("=" * 85)

    system = RetailProductionSystem(
        clip_weights_dir="yg dipake/models/model_clip_17cat_projection_head",
        svm_model_path="yg dipake/models/svm_multimodal_large_17cat.joblib",
        manifest_csv_path="yg dipake/data/manifest_3400.csv",
        zip_images_path="retail-products-classification.zip",
        device="cpu",
        preload_catalog_size=300
    )

    # ---------------------------------------------------------
    # UJI COBA 1: PENCARIAN DENGAN TEKS (TEXT-TO-IMAGE)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 🔍 METODE A: USER SEARCH MENGGUNAKAN TEKS (AUTO-TRANSLATE)")
    print("=" * 85)
    
    sample_queries = [
        "senar gitar elektrik fender",
        "charger kabel usb fast charging",
        "kunci pas obeng perkakas bengkel"
    ]

    for i, query in enumerate(sample_queries, 1):
        print(f"\n[{i}] User mencari via teks: \"{query}\"")
        res = system.user_search_by_text(query, top_k=3, auto_translate=True)
        print(f"    🌐 Terjemahan Query  : \"{res['user_query_translated']}\"")
        print(f"    🎯 Kategori SVM      : {res['detected_category']['category']} ({res['detected_category']['confidence_pct']}%)")
        print("    📦 Top Rekomendasi Foto Produk Ditemukan:")
        for prod in res['retrieved_products']:
            print(f"       - Rank {prod['rank']} [{prod['match_percentage']}%]: {prod['title'][:60]}... (Kat: {prod['category']})")

    # ---------------------------------------------------------
    # UJI COBA 2: PENCARIAN DENGAN GAMBAR (IMAGE-TO-IMAGE)
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 📷 METODE B: USER SEARCH MENGGUNAKAN FOTO / GAMBAR PRODUK")
    print("=" * 85)

    with zipfile.ZipFile("retail-products-classification.zip", 'r') as zf:
        # Ambil sampel foto produk untuk pengujian query gambar
        with zf.open("train/train/B0002E2GO2.jpg") as f:
            query_img_gitar = Image.open(io.BytesIO(f.read())).convert('RGB')

    print("[*] User mengunggah foto produk gitar/alat musik ke pencarian...")
    img_res = system.user_search_by_image(query_img_gitar, top_k=3)
    print(f"    🎯 Kategori Terdeteksi via SVM : {img_res['detected_category']['category']} ({img_res['detected_category']['confidence_pct']}%)")
    print("    📦 Top Rekomendasi Produk dengan Kemiripan Vektor Visual Tertinggi:")
    for prod in img_res['retrieved_products']:
        print(f"       - Rank {prod['rank']} [{prod['match_percentage']}% Match]: {prod['title'][:60]}... (ID: {prod['product_id']})")
    print("=" * 85)

if __name__ == "__main__":
    main()
