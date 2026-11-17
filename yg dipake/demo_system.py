"""
demo_system.py
==============
Demo Interaktif Command Line untuk Sistem Ritel Terpadu (Dual-Workflow):
1. Alur Admin: Ingestion produk baru (Gambar + Judul) -> Auto-Categorization SVM.
2. Alur Pengguna: Pencarian Semantik Produk via Teks (Bahasa Indonesia/Inggris) -> Top-K Visual Retrieval via CLIP.
"""

import os
import io
import zipfile
from PIL import Image
from retail_production_pipeline import RetailProductionSystem

def main():
    print("=" * 85)
    print(" 🌟 DEMO INTERAKTIF SISTEM RITEL DUAL-WORKFLOW (ADMIN & USER)")
    print("=" * 85)

    system = RetailProductionSystem(
        clip_weights_dir="yg dipake/models/model_clip_17cat_projection_head",
        svm_model_path="yg dipake/models/svm_multimodal_large_17cat.joblib",
        manifest_csv_path="yg dipake/data/manifest_3400.csv",
        zip_images_path="retail-products-classification.zip",
        device="cpu",
        preload_catalog_size=340
    )

    print("\n" + "=" * 85)
    print("--- 🏢 ALUR 1: DEMO SISI ADMIN (INGESTION & AUTO-CATEGORIZATION) ---")
    print("=" * 85)

    with zipfile.ZipFile("retail-products-classification.zip", 'r') as zf:
        with zf.open("train/train/B0002E2GO2.jpg") as f:
            sample_img = Image.open(io.BytesIO(f.read())).convert('RGB')

    admin_title = "Fender 099-4912-000 Vintage-Style Telecaster Saddle Bridge"
    print(f"[*] Admin mengunggah gambar produk fisik...")
    print(f"[*] Admin memasukkan nama produk: \"{admin_title}\"")

    res_admin = system.admin_add_product(sample_img, admin_title)
    print(f"\n[HASIL ALUR ADMIN]")
    print(f"  • Kategori Terpilih : {res_admin['predicted_category']} (Keyakinan: {res_admin['confidence_pct']}%)")
    print("  • Kandidat Kategori Teratas (Top-3):")
    for cand in res_admin['top_candidates']:
        print(f"     - Rank {cand['rank']}: {cand['category']} ({cand['confidence_pct']}%)")
    print(f"  • Status Katalog    : Produk berhasil didaftarkan (Total sekarang: {res_admin['total_catalog_now']} item)")

    print("\n" + "=" * 85)
    print("--- 🛒 ALUR 2A: DEMO PENGGUNA MENCARI VIA TEKS (TEXT-TO-IMAGE) ---")
    print("=" * 85)

    user_queries = [
        "senar gitar elektrik akustik fender",
        "charger hp kabel usb type c fast charging"
    ]

    for q in user_queries:
        print(f"\n[*] Pengguna mengetik query: \"{q}\"")
        res_user = system.user_search_by_text(q, top_k=3, auto_translate=True)
        print(f"  • Terjemahan Otomatis : \"{res_user['user_query_translated']}\"")
        print(f"  • Kategori Terdeteksi : {res_user['detected_category']['category']} ({res_user['detected_category']['confidence_pct']}%)")
        print("  • Rekomendasi Foto Produk Ditemukan (Cosine Similarity CLIP):")
        for prod in res_user['retrieved_products']:
            print(f"     [{prod['match_percentage']}% Match] {prod['title'][:65]}... (ID: {prod['product_id']})")

    print("\n" + "=" * 85)
    print("--- 📷 ALUR 2B: DEMO PENGGUNA MENCARI VIA FOTO PRODUK (IMAGE-TO-IMAGE) ---")
    print("=" * 85)

    print("[*] Pengguna mengunggah foto produk gitar listrik dari galeri/kamera...")
    res_img = system.user_search_by_image(sample_img, top_k=3)
    print(f"  • Kategori Visual Terdeteksi (SVM) : {res_img['detected_category']['category']} ({res_img['detected_category']['confidence_pct']}%)")
    print("  • Produk dengan Kemiripan Vektor Visual Tertinggi di Database:")
    for prod in res_img['retrieved_products']:
        print(f"     [{prod['match_percentage']}% Match] {prod['title'][:65]}... (ID: {prod['product_id']})")
    print("=" * 85)

if __name__ == "__main__":
    main()
