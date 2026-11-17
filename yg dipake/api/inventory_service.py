"""
inventory_service.py
====================
Service Engine Backend untuk Sistem Ritel Terpadu (Dual-Workflow):
Dirancang khusus untuk diintegrasikan ke Backend (FastAPI, Flask, Express, Django)
atau dipanggil secara langsung (Python Native).

Menyediakan interface siap pakai untuk:
1. Alur Admin: Ingestion Produk Baru (Foto + Judul) & Auto-Categorization (Multimodal SVM).
2. Alur Pengguna: Pencarian Semantik Produk via Teks (Bahasa Indonesia/Inggris) ke Foto Produk Fisik (CLIP Cross-Modal).
"""

import os
import sys
from typing import List, Dict, Any, Optional
from PIL import Image

# Hubungkan dengan pipeline produksi inti
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from retail_production_pipeline import RetailProductionSystem

class InventoryService:
    def __init__(
        self,
        clip_weights_dir: Optional[str] = None,
        svm_model_path: Optional[str] = None,
        manifest_csv_path: Optional[str] = None,
        zip_images_path: Optional[str] = None,
        device: str = "cpu",
        preload_catalog_size: int = 340
    ):
        base_models = os.path.join(parent_dir, "models")
        base_data = os.path.join(parent_dir, "data")

        self.clip_path = clip_weights_dir or os.path.join(base_models, "model_clip_17cat_projection_head")
        self.svm_path = svm_model_path or os.path.join(base_models, "svm_multimodal_large_17cat.joblib")
        self.manifest_path = manifest_csv_path or os.path.join(base_data, "manifest_3400.csv")
        self.zip_path = zip_images_path or os.path.join(parent_dir, "..", "retail-products-classification.zip")
        if not os.path.exists(self.zip_path):
            self.zip_path = "retail-products-classification.zip"

        # Inisialisasi Engine Inti
        self.system = RetailProductionSystem(
            clip_weights_dir=self.clip_path,
            svm_model_path=self.svm_path,
            manifest_csv_path=self.manifest_path,
            zip_images_path=self.zip_path,
            device=device,
            preload_catalog_size=preload_catalog_size
        )

    # -------------------------------------------------------------------------
    # 1. ALUR ADMIN: INGESTION & AUTO-CATEGORIZATION
    # -------------------------------------------------------------------------
    def add_product(self, image: Image.Image, title: str, product_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Mendaftarkan produk baru ke katalog:
        - Mengekstrak fitur visual & teks (1024-dim multimodal).
        - Menentukan kategori teratas menggunakan model SVM.
        - Menyimpan produk ke feature store katalog aktif.
        """
        return self.system.admin_add_product(pil_image=image, product_title=title, product_id=product_id)

    # -------------------------------------------------------------------------
    # 2. ALUR PENGGUNA: PENCARIAN SEMANTIK PRODUK (TEXT & IMAGE)
    # -------------------------------------------------------------------------
    def search_products(self, query: str, top_k: int = 5, auto_translate: bool = True) -> Dict[str, Any]:
        """
        Mencari produk berdasarkan teks bahasa alami pengguna (mendukung Bahasa Indonesia & Inggris):
        - Translasi otomatis query jika non-Inggris.
        - Deteksi kategori query via SVM.
        - Perangkingan foto produk relevan via Cosine Similarity CLIP (Text-to-Image).
        """
        return self.system.user_search_by_text(text_query=query, top_k=top_k, auto_translate=auto_translate)

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Mencari produk berdasarkan foto / gambar produk yang diunggah pengguna:
        - Ekstraksi vektor visual gambar query (512-dimensi CLIP).
        - Deteksi estimasi kategori via SVM.
        - Pencocokan gambar paling mirip di katalog via Cosine Similarity (Image-to-Image).
        """
        return self.system.user_search_by_image(query_image=image, top_k=top_k)

    def get_image_bytes(self, product_id: str) -> Optional[bytes]:
        """Mengambil data byte gambar produk dari katalog/zip untuk di-stream ke browser."""
        return self.system.get_product_image_bytes(product_id)

    # -------------------------------------------------------------------------
    # 3. HELPER FUNCTIONS
    # -------------------------------------------------------------------------
    def get_categories(self) -> List[str]:
        """Mendapatkan daftar 17 kategori ritel resmi."""
        return self.system.categories

    def get_catalog_count(self) -> int:
        """Mendapatkan jumlah produk aktif di katalog."""
        return len(self.system.df_catalog)
