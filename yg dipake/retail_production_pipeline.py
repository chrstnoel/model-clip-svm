"""
retail_production_pipeline.py
==============================
Pipeline Model Produksi Ritel (17 Kategori Bersih & Multimodal Architecture):
1. MODEL:
   - CLIP: Fine-Tuned Projection Head (17 Kategori Amazon Bersih).
   - SVM: Calibrated Classifier (1024-dim Multimodal: Visual 512 + Text 512).
2. ALUR 1 (ADMIN INGESTION):
   - Menerima Gambar + Nama/Judul Produk baru.
   - Ekstrak representasi visual dan representasi semantik judul.
   - SVM memprediksi Top-1 & Top-3 kategori dengan persentase keyakinan.
   - Mendaftarkan produk ke katalog dan feature store.
3. ALUR 2 (USER SEARCH & RETRIEVAL):
   - Menerima query teks pengguna (otomatis deteksi dan terjemahkan jika bahasa non-Inggris/Indonesia).
   - Ekstrak feature teks query dengan CLIP.
   - Prediksi Top-3 kategori terkait dengan SVM.
   - Cosine Similarity Engine membandingkan query teks dengan foto produk katalog (Top-K visual retrieval).
"""

import os
import io
import re
import urllib.parse
import urllib.request
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
import joblib
from transformers import CLIPProcessor, CLIPModel

def translate_to_english(text: str) -> str:
    """Menerjemahkan teks query bahasa Indonesia/non-Inggris ke bahasa Inggris via Google Translate API."""
    if not text or not text.strip():
        return text
    clean_text = text.strip()
    encoded_text = urllib.parse.quote(clean_text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            translated_parts = [segment[0] for segment in res_data[0] if segment and segment[0]]
            translated_text = "".join(translated_parts).strip()
            return translated_text if translated_text else clean_text
    except Exception:
        # Fallback ke MyMemory API
        try:
            mm_url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair=id|en"
            req_mm = urllib.request.Request(mm_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_mm, timeout=4) as resp_mm:
                data_mm = json.loads(resp_mm.read().decode('utf-8'))
                mm_trans = data_mm.get("responseData", {}).get("translatedText", "").strip()
                return mm_trans if mm_trans else clean_text
        except Exception:
            return clean_text

class RetailProductionSystem:
    def __init__(
        self,
        clip_weights_dir="yg dipake/models/model_clip_17cat_projection_head",
        svm_model_path="yg dipake/models/svm_multimodal_large_17cat.joblib",
        manifest_csv_path="yg dipake/data/manifest_3400.csv",
        zip_images_path="retail-products-classification.zip",
        device="cpu",
        preload_catalog_size=500  # Memuat 500 produk awal ke feature store untuk simulasi katalog aktif
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"[*] Menginisialisasi Retail Production System pada device: {str(self.device).upper()}...")

        # 1. Load Model Fine-Tuned CLIP
        print(f"[*] Memuat CLIP Fine-Tuned dari: {clip_weights_dir}")
        self.clip_model = CLIPModel.from_pretrained(clip_weights_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_weights_dir)
        self.clip_model.eval()

        # 2. Load Model SVM Multimodal (1024-dim)
        print(f"[*] Memuat Multimodal SVM dari: {svm_model_path}")
        svm_bundle = joblib.load(svm_model_path)
        self.svm_model = svm_bundle['model']
        self.categories = svm_bundle['categories']
        self.cat2idx = svm_bundle['cat2idx']
        self.idx2cat = svm_bundle['idx2cat']

        # 3. Load Katalog Data & Setup ZIP Reader
        self.zip_path = zip_images_path
        self.manifest_path = manifest_csv_path
        print(f"[*] Membaca manifest katalog dari: {manifest_csv_path}")
        self.df_manifest = pd.read_csv(manifest_csv_path)

        # Aktifkan katalog produk seimbang (mengambil perwakilan dari setiap 17 kategori)
        if preload_catalog_size and preload_catalog_size < len(self.df_manifest):
            per_cat = max(1, preload_catalog_size // len(self.categories))
            self.df_catalog = self.df_manifest.groupby('clean_category').head(per_cat).reset_index(drop=True)
        else:
            self.df_catalog = self.df_manifest.copy().reset_index(drop=True)

        self.zip_ref = zipfile.ZipFile(self.zip_path, 'r') if os.path.exists(self.zip_path) else None

        # 4. Bangun Feature Store Visual Embeddings untuk Katalog Aktif
        print(f"[*] Membangun Feature Store Visual Embeddings ({len(self.df_catalog)} item katalog aktif)...")
        self.catalog_embeddings = self._precompute_catalog_embeddings()
        print(f"[✓] Sistem Produksi Siap! Total {len(self.categories)} Kategori Bersih Terdaftar.\n")

    def _extract_image_feature(self, pil_image: Image.Image) -> np.ndarray:
        """Ekstraksi 512-dim visual normalized embedding menggunakan CLIP."""
        inputs = self.processor(images=[pil_image.convert("RGB")], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.clip_model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, 'pooler_output') else (out if torch.is_tensor(out) else out[0])
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy().flatten()

    def _extract_text_feature(self, text: str) -> np.ndarray:
        """Ekstraksi 512-dim text normalized embedding menggunakan CLIP."""
        inputs = self.processor(
            text=[str(text)[:70]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            out = self.clip_model.get_text_features(**inputs)
            feat = out.pooler_output if hasattr(out, 'pooler_output') else (out if torch.is_tensor(out) else out[0])
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy().flatten()

    def _precompute_catalog_embeddings(self) -> np.ndarray:
        """Menghitung visual embeddings untuk produk di katalog."""
        embeddings = []
        if self.zip_ref is None:
            return np.zeros((len(self.df_catalog), 512), dtype=np.float32)

        for _, row in self.df_catalog.iterrows():
            try:
                with self.zip_ref.open(row['img_path']) as f:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
                    feat = self._extract_image_feature(img)
                    embeddings.append(feat)
            except Exception:
                embeddings.append(np.zeros(512, dtype=np.float32))

        return np.vstack(embeddings)

    def _convert_label(self, val):
        if isinstance(val, (int, np.integer)):
            return self.idx2cat.get(int(val), str(val))
        return str(val)

    def get_product_image_bytes(self, product_id: str) -> bytes:
        """Mengambil data raw bytes gambar produk dari arsip ZIP atau custom upload."""
        matched = self.df_catalog[self.df_catalog['ImgId'] == product_id]
        if matched.empty:
            matched = self.df_manifest[self.df_manifest['ImgId'] == product_id]
        if matched.empty:
            return None
        
        img_rel_path = matched.iloc[0]['img_path']
        if self.zip_ref is not None:
            try:
                with self.zip_ref.open(img_rel_path) as f:
                    return f.read()
            except Exception:
                pass
        return None

    # =========================================================================
    # ALUR 1: ADMIN INGESTION (GAMBAR + JUDUL -> DETEKSI KATEGORI SVM -> KATALOG)
    # =========================================================================
    def admin_add_product(self, pil_image: Image.Image, product_title: str, product_id: str = None) -> dict:
        """
        Alur Sisi Admin:
        1. Ekstrak visual embedding (512-dim) dan text embedding (512-dim).
        2. Gabungkan menjadi vektor 1024-dim multimodal.
        3. Model SVM memprediksi probabilitas untuk seluruh 17 kategori.
        4. Masukkan produk ke database katalog aktif beserta vektor visualnya.
        """
        # Ekstraksi fitur visual dan teks
        v_img = self._extract_image_feature(pil_image)
        v_txt = self._extract_text_feature(product_title)
        v_multi = np.concatenate([v_img, v_txt]).reshape(1, -1)

        # Prediksi SVM
        probs = self.svm_model.predict_proba(v_multi)[0]
        top_indices = np.argsort(probs)[::-1]

        top_predictions = []
        for rank, idx in enumerate(top_indices[:3], 1):
            cat_name = self._convert_label(self.svm_model.classes_[idx])
            top_predictions.append({
                "rank": rank,
                "category": cat_name,
                "confidence_pct": round(float(probs[idx]) * 100, 2)
            })

        pred_category = top_predictions[0]["category"]
        confidence = top_predictions[0]["confidence_pct"]

        # Registrasi ke katalog aktif
        if not product_id:
            product_id = f"NEW_{len(self.df_catalog)+1:04d}"

        new_item = {
            "ImgId": product_id,
            "title": product_title,
            "clean_category": pred_category,
            "img_path": "custom_upload"
        }
        self.df_catalog = pd.concat([self.df_catalog, pd.DataFrame([new_item])], ignore_index=True)
        self.catalog_embeddings = np.vstack([self.catalog_embeddings, v_img.reshape(1, -1)])

        return {
            "status": "success",
            "action": "Product Ingestion & Categorization",
            "product_id": product_id,
            "title": product_title,
            "predicted_category": pred_category,
            "confidence_pct": confidence,
            "top_candidates": top_predictions,
            "total_catalog_now": len(self.df_catalog)
        }

    # =========================================================================
    # ALUR 2: USER SEARCH (TEKS QUERY -> TRANSLASI -> TOP-3 KAT -> TOP-K RETRIEVAL)
    # =========================================================================
    def user_search_by_text(self, text_query: str, top_k: int = 5, auto_translate: bool = True) -> dict:
        """
        Alur Sisi Pengguna:
        1. Query teks diterjemahkan otomatis ke bahasa Inggris (jika bahasa Indonesia/asing).
        2. Ekstraksi text embedding dengan CLIP (512-dim).
        3. Estimasi kategori query dengan SVM (Top-3 kandidat).
        4. Cosine Similarity Engine membandingkan vektor teks dengan seluruh foto produk katalog.
        5. Mengembalikan Top-K produk terbaik beserta skor kesamaan visual.
        """
        query_en = translate_to_english(text_query) if auto_translate else text_query

        # Ekstraksi fitur teks query
        q_feat = self._extract_text_feature(query_en)

        # Prediksi kategori query dengan SVM (Simulated Multimodal text prior: v_txt + v_txt)
        q_pseudo_multi = np.concatenate([q_feat, q_feat]).reshape(1, -1)
        probs = self.svm_model.predict_proba(q_pseudo_multi)[0]
        top_indices = np.argsort(probs)[::-1]

        detected_cats = []
        for rank, idx in enumerate(top_indices[:3], 1):
            cat_name = self._convert_label(self.svm_model.classes_[idx])
            detected_cats.append({
                "rank": rank,
                "category": cat_name,
                "confidence_pct": round(float(probs[idx]) * 100, 2)
            })

        # Cosine Similarity Retrieval
        # self.catalog_embeddings: (N, 512), q_feat: (512,)
        scores = np.dot(self.catalog_embeddings, q_feat)
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_products = []
        for rank, idx in enumerate(top_k_indices, 1):
            row = self.df_catalog.iloc[idx]
            sim_score = float(scores[idx])
            prod_id = str(row.get("ImgId", f"ITEM_{idx}"))
            img_path = str(row.get("img_path", ""))
            
            # Siapkan URL endpoint gambar lokal untuk dilihat langsung di browser
            image_url = f"/image/{prod_id}"

            retrieved_products.append({
                "rank": rank,
                "product_id": prod_id,
                "title": str(row.get("title", "")),
                "category": str(row.get("clean_category", "")),
                "similarity_score": round(sim_score, 4),
                "match_percentage": round(sim_score * 100, 2),
                "image_path": img_path,
                "image_url": image_url
            })

        return {
            "status": "success",
            "action": "Semantic Text Search & Product Retrieval",
            "user_query_raw": text_query,
            "user_query_translated": query_en,
            "detected_category": detected_cats[0],
            "top_category_candidates": detected_cats,
            "retrieved_products": retrieved_products
        }

    # =========================================================================
    # ALUR 2B: USER VISUAL SEARCH (FOTO QUERY -> VEKTOR VISUAL -> TOP-K RETRIEVAL)
    # =========================================================================
    def user_search_by_image(self, query_image: Image.Image, top_k: int = 5) -> dict:
        """
        Alur Sisi Pengguna (Pencarian Gambar ke Gambar / Visual Similarity Search):
        1. Menerima gambar/foto produk dari pengguna.
        2. Ekstraksi visual feature vector 512-dimensi menggunakan Vision Backbone CLIP.
        3. Prediksi estimasi kategori gambar menggunakan SVM (pseudo multimodal: v_img + v_img).
        4. Cosine Similarity Engine membandingkan vektor gambar query dengan seluruh embedding
           foto produk di database katalog.
        5. Mengembalikan Top-K produk katalog dengan nilai kemiripan vektor visual tertinggi.
        """
        # 1. Ekstraksi vektor visual gambar query (512-dim normalized)
        q_img_feat = self._extract_image_feature(query_image)

        # 2. Prediksi kategori visual query menggunakan SVM
        q_pseudo_multi = np.concatenate([q_img_feat, q_img_feat]).reshape(1, -1)
        probs = self.svm_model.predict_proba(q_pseudo_multi)[0]
        top_indices = np.argsort(probs)[::-1]

        detected_cats = []
        for rank, idx in enumerate(top_indices[:3], 1):
            cat_name = self._convert_label(self.svm_model.classes_[idx])
            detected_cats.append({
                "rank": rank,
                "category": cat_name,
                "confidence_pct": round(float(probs[idx]) * 100, 2)
            })

        # 3. Hitung Cosine Similarity antara gambar query dengan seluruh gambar katalog
        # self.catalog_embeddings: (N, 512), q_img_feat: (512,)
        scores = np.dot(self.catalog_embeddings, q_img_feat)
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        retrieved_products = []
        for rank, idx in enumerate(top_k_indices, 1):
            row = self.df_catalog.iloc[idx]
            sim_score = float(scores[idx])
            prod_id = str(row.get("ImgId", f"ITEM_{idx}"))
            img_path = str(row.get("img_path", ""))
            image_url = f"/image/{prod_id}"

            retrieved_products.append({
                "rank": rank,
                "product_id": prod_id,
                "title": str(row.get("title", "")),
                "category": str(row.get("clean_category", "")),
                "similarity_score": round(sim_score, 4),
                "match_percentage": round(sim_score * 100, 2),
                "image_path": img_path,
                "image_url": image_url
            })

        return {
            "status": "success",
            "action": "Visual Image Search & Product Retrieval (Image-to-Image)",
            "search_type": "image_to_image",
            "detected_category": detected_cats[0],
            "top_category_candidates": detected_cats,
            "retrieved_products": retrieved_products
        }
