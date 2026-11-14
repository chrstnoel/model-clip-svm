"""
inventory_service.py
====================
Service perantara untuk mengelola katalog produk inventaris dan pencarian teks/gambar.
Digunakan untuk integrasi antara database produk toko dengan model CLIP.
"""

import os
import pandas as pd
from typing import List, Dict

class InventoryService:
    def __init__(self, manifest_csv: str = "DORMIY-ML-Dataset/data/final/cv/dormiy_cv_manifest.csv"):
        self.manifest_csv = manifest_csv
        self.catalog = None
        self.load_catalog()

    def load_catalog(self):
        if os.path.exists(self.manifest_csv):
            self.catalog = pd.read_csv(self.manifest_csv)
            print(f"[InventoryService] Berhasil memuat {len(self.catalog)} produk inventaris.")
        else:
            print(f"[InventoryService] Warning: {self.manifest_csv} tidak ditemukan.")
            self.catalog = pd.DataFrame()

    def get_product_by_id(self, product_id: str) -> Optional_Dict:
        if self.catalog.empty:
            return None
        matched = self.catalog[self.catalog['product_id'] == product_id]
        if not matched.empty:
            return matched.iloc[0].to_dict()
        return None

    def list_categories(self) -> List[str]:
        if self.catalog.empty or 'main_category' not in self.catalog.columns:
            return []
        return sorted(self.catalog['main_category'].unique().tolist())

if __name__ == "__main__":
    service = InventoryService()
    print("Daftar kategori inventaris:", service.list_categories())

