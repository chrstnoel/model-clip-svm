"""
uji_alur2_user.py
=================
File Pengujian Mandiri Khusus ALUR 2 (SISI USER / PENCARIAN TEKS):
- Menerima kata kunci pencarian bebas dari pengguna.
- Model CLIP mengubah teks menjadi vektor 512 dimensi.
- Model SVM mendeteksi kategori yang dicari pengguna.
- Cosine Similarity Engine membandingkan dengan 50 foto produk katalog dan mengeluarkan Top-5 produk paling cocok.
"""

import os
import sys
from dormiy_production_pipeline import DormiyRetailSystem

def main():
    print("=" * 85)
    print(" 🛒 SISTEM PENGUJIAN ALUR 2: USER SEMANTIC TEXT SEARCH & TOP-5 RETRIEVAL")
    print("=" * 85)

    # 1. Inisialisasi Sistem
    system = DormiyRetailSystem(
        clip_weights_dir="experiment_results/sesi_3_projection_head/model_clip_dormiy_sesi3",
        svm_model_path="experiment_results/sesi_augmented_svm/svm_augmented_classifier.joblib",
        manifest_csv_path="DORMIY-ML-Dataset/data/final/cv/dormiy_cv_manifest.csv",
        device="cpu"
    )

    # 2. Daftar Query Pencarian Teks Bebas untuk Diuji
    sample_queries = [
        "pisau dapur pemotong makanan stainless steel",
        "alat pertukangan bor tembok beton obeng",
        "stationery office desk bell stapler",
        "3d printer machine for electronics project",
        "meja lemari pakaian kamar asrama"
    ]

    print("\n🚀 MEMULAI UJI COBA 5 PENCARIAN TEKS PENGGUNA:\n")

    for i, query in enumerate(sample_queries, 1):
        print(f"[{i}] USER MENGETIK DI PENCARIAN:")
        print(f"    🔍 Query: \"{query}\"")

        # Jalankan Fungsi Alur 2
        result = system.user_search_by_text(text_query=query, top_k=5)

        detected_cat = result["detected_category"]["category"]
        conf = result["detected_category"]["confidence_pct"]

        print(f"    🎯 Kategori Terdeteksi (SVM): {detected_cat} (Keyakinan: {conf}%)")
        print("    📦 Top 5 Rekomendasi Produk Paling Cocok (Cosine Similarity):")

        for prod in result["top_5_recommended_products"]:
            sim_score = prod["similarity_score"]
            prod_id = prod["product_id"]
            title = prod["title"][:45]
            cat = prod["category"]
            print(f"       #{prod['rank']} [Skor: {sim_score:.4f}] [{prod_id}] {title:<45} ({cat})")

        print("-" * 85)

    # 3. Mode Interaktif (Bisa langsung ketik sendiri di terminal)
    print("\n💡 MODE INTERAKTIF:")
    print("Anda bisa mencoba mengetik kalimat apa saja sesuka hati!")
    print("Ketik kata pencarian Anda di bawah ini (atau tekan Enter/ketik 'exit' untuk selesai):")
    
    try:
        while True:
            user_input = input("\nKetik kata pencarian: ").strip()
            if not user_input or user_input.lower() in ['exit', 'quit', 'q']:
                print("Pengujian selesai. Sampai jumpa!")
                break

            res = system.user_search_by_text(text_query=user_input, top_k=5)
            print(f"-> Kategori Terdeteksi : {res['detected_category']['category']} ({res['detected_category']['confidence_pct']}%)")
            print("-> 5 Produk Teratas:")
            for p in res['top_5_recommended_products']:
                print(f"   #{p['rank']} [Skor: {p['similarity_score']:.4f}] [{p['product_id']}] {p['title'][:45]} ({p['category']})")
    except (EOFError, KeyboardInterrupt):
        print("\nKeluar dari mode interaktif.")

if __name__ == "__main__":
    main()

