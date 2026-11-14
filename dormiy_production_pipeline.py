"""
dormiy_production_pipeline.py
==============================
Pipeline Model Produksi DORMIY yang menyatukan 2 alur kerja utama:
1. ALUR 1 (ADMIN INGESTION): 
   Menerima gambar baru + nama produk, mengekstrak embedding visual CLIP,
   memprediksi kategori yang cocok secara otomatis menggunakan SVM, 
   dan mendaftarkannya ke database katalog.

2. ALUR 2 (USER SEARCH & RETRIEVAL):
   Menerima input query teks dari pengguna, mendeteksi kategori yang dicari menggunakan SVM,
   dan menemukan Top-5 produk paling cocok menggunakan Cosine Similarity.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from transformers import CLIPProcessor, CLIPModel

class DormiyRetailSystem:
    def __init__(
        self,
        clip_weights_dir="experiment_results/sesi_3_projection_head/model_clip_dormiy_sesi3",
        svm_model_path="experiment_results/sesi_augmented_svm/svm_augmented_classifier.joblib",
        manifest_csv_path="DORMIY-ML-Dataset/data/final/cv/dormiy_cv_manifest.csv",
        device="cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Menginisialisasi DORMIY Retail System pada device: {str(self.device).upper()}...")

        # 1. Load CLIP Model (Fine-Tuned Sesi 3)
        print(f"Loading CLIP weights from: {clip_weights_dir}")
        self.clip_model = CLIPModel.from_pretrained(clip_weights_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_weights_dir)
        self.clip_model.eval()

        # 2. Load Katalog Produk
        print(f"Loading catalog manifest from: {manifest_csv_path}")
        self.manifest_path = manifest_csv_path
        self.df_catalog = pd.read_csv(manifest_csv_path)
        self.df_catalog['full_image_path'] = self.df_catalog['final_image_path'].apply(
            lambda p: os.path.join("DORMIY-ML-Dataset", p.replace("\\", "/"))
        )
        self.categories = sorted(self.df_catalog['main_category'].unique().tolist())
        self.cat2idx = {c: i for i, c in enumerate(self.categories)}
        self.idx2cat = {i: c for i, c in enumerate(self.categories)}

        # 3. Load SVM Classifier (Augmented 250 Samples)
        print(f"Loading SVM Category Classifier from: {svm_model_path}")
        self.svm_model = joblib.load(svm_model_path)

        # 4. Bangun Feature Store (Katalog Embeddings 50 Produk)
        print("Membangun Feature Store Embeddings...")
        self.catalog_embeddings = self._precompute_catalog_embeddings()
        print(f"✅ Sistem Siap! Total Produk di Katalog: {len(self.df_catalog)} item.\n")

    def _precompute_catalog_embeddings(self):
        embeddings = []
        for path in self.df_catalog['full_image_path']:
            img = Image.open(path).convert("RGB")
            inp = self.processor(images=[img], return_tensors="pt").to(self.device)
            with torch.no_grad():
                im_out = self.clip_model.get_image_features(**inp)
                im_f = im_out.pooler_output if hasattr(im_out, 'pooler_output') else (im_out if torch.is_tensor(im_out) else im_out[0])
                im_f = im_f / im_f.norm(dim=-1, keepdim=True)
                embeddings.append(im_f.cpu().numpy())
        return np.vstack(embeddings)

    def _convert_label(self, val):
        if isinstance(val, (int, np.integer)):
            return self.idx2cat.get(int(val), str(val))
        return str(val)

    # =========================================================================
    # ALUR 1: ADMIN INGESTION (Tambah Gambar + Nama Produk -> Prediksi Kategori)
    # =========================================================================
    def admin_add_new_product(self, image_input, product_title, save_to_catalog=False):
        """
        Alur 1 untuk Admin:
        - Menerima gambar baru dan nama produk
        - CLIP mengubah gambar menjadi vektor 512 dimensi
        - SVM otomatis menentukan produk ini paling cocok masuk ke kategori mana
        - (Opsional) Langsung mendaftarkannya ke database katalog aktif
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            img_path = image_input
        else:
            img = image_input.convert("RGB")
            img_path = "uploaded_image"

        # Ekstraksi Vektor Gambar
        inp = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            im_out = self.clip_model.get_image_features(**inp)
            im_f = im_out.pooler_output if hasattr(im_out, 'pooler_output') else (im_out if torch.is_tensor(im_out) else im_out[0])
            im_f = im_f / im_f.norm(dim=-1, keepdim=True)
            img_vec = im_f.cpu().numpy()

        # Prediksi Kategori dengan SVM
        raw_pred = self.svm_model.predict(img_vec)[0]
        category_probs = self.svm_model.predict_proba(img_vec)[0]

        # Ranking Top-2 Kategori SVM
        ranked_cats = np.argsort(category_probs)[::-1]
        top1_raw = self.svm_model.classes_[ranked_cats[0]]
        top2_raw = self.svm_model.classes_[ranked_cats[1]]

        top1_cat = self._convert_label(top1_raw)
        top1_conf = round(float(category_probs[ranked_cats[0]]) * 100, 2)
        top2_cat = self._convert_label(top2_raw)
        top2_conf = round(float(category_probs[ranked_cats[1]]) * 100, 2)

        # Opsi simpan ke in-memory catalog
        new_product_id = f"prod_new_{len(self.df_catalog) + 1:04d}"
        if save_to_catalog:
            self.catalog_embeddings = np.vstack([self.catalog_embeddings, img_vec])
            new_row = pd.DataFrame([{
                'product_id': new_product_id,
                'title': product_title,
                'main_category': top1_cat,
                'final_image_path': img_path,
                'full_image_path': img_path
            }])
            self.df_catalog = pd.concat([self.df_catalog, new_row], ignore_index=True)

        return {
            "status": "success",
            "workflow": "ALUR 1: Admin Product Ingestion",
            "product_title": product_title,
            "assigned_product_id": new_product_id,
            "svm_category_prediction": {
                "recommended_category": top1_cat,
                "confidence_pct": top1_conf,
                "secondary_category": top2_cat,
                "secondary_confidence_pct": top2_conf
            },
            "embedding_vector_shape": list(img_vec.shape),
            "added_to_active_catalog": save_to_catalog
        }

    # =========================================================================
    # ALUR 2: USER TEXT SEARCH (Input Teks -> Deteksi Kategori -> Top-5 Produk)
    # =========================================================================
    def user_search_by_text(self, text_query, top_k=5):
        """
        Alur 2 untuk Pengguna:
        - Menerima kata kunci pencarian teks bebas dari user
        - CLIP mengubah teks menjadi vektor 512 dimensi
        - SVM mendeteksi kategori yang dicari user
        - Cosine Similarity mencari Top-5 produk katalog yang paling cocok
        """
        # Ekstraksi Vektor Teks
        txt_inp = self.processor(text=[text_query], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            txt_out = self.clip_model.get_text_features(**txt_inp)
            txt_f = txt_out.pooler_output if hasattr(txt_out, 'pooler_output') else (txt_out if torch.is_tensor(txt_out) else txt_out[0])
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            txt_vec = txt_f.cpu().numpy()

        # 1. Deteksi Kategori Teks via SVM
        raw_pred = self.svm_model.predict(txt_vec)[0]
        category_pred = self._convert_label(raw_pred)
        category_probs = self.svm_model.predict_proba(txt_vec)[0]
        top_cat_conf = round(float(np.max(category_probs)) * 100, 2)

        # 2. Cari Top-K Produk via Cosine Similarity
        cosine_sims = np.dot(self.catalog_embeddings, txt_vec.T).squeeze()
        top_indices = np.argsort(cosine_sims)[::-1][:top_k]

        recommended_products = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.df_catalog.iloc[idx]
            recommended_products.append({
                "rank": rank,
                "product_id": row["product_id"],
                "title": row["title"],
                "category": row["main_category"],
                "similarity_score": round(float(cosine_sims[idx]), 4),
                "image_path": row["full_image_path"]
            })

        return {
            "status": "success",
            "workflow": "ALUR 2: User Semantic Text Search",
            "user_query": text_query,
            "detected_category": {
                "category": category_pred,
                "confidence_pct": top_cat_conf
            },
            "top_5_recommended_products": recommended_products
        }
