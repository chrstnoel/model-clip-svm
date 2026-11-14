"""
server.py
=========
API Server Backend FastAPI untuk Production Retail DORMIY.
Mengimplementasikan 2 Alur Model Terpadu:
1. POST /admin/add-product : Alur Admin (Gambar + Nama -> Deteksi Kategori SVM -> Simpan ke Katalog)
2. POST /user/search       : Alur User (Teks Query -> Deteksi Kategori SVM -> Top-5 Produk Cosine)
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from dormiy_production_pipeline import DormiyRetailSystem

app = FastAPI(
    title="DORMIY Retail Dual-Workflow API",
    description="Backend Service Produksi: Alur Admin (Ingestion) & Alur Pengguna (Semantic Search)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Model Pipeline Tunggal
print("Memuat DormiyRetailSystem ke memori server...")
retail_system = DormiyRetailSystem(
    clip_weights_dir="experiment_results/sesi_3_projection_head/model_clip_dormiy_sesi3",
    svm_model_path="experiment_results/sesi_augmented_svm/svm_augmented_classifier.joblib",
    manifest_csv_path="DORMIY-ML-Dataset/data/final/cv/dormiy_cv_manifest.csv",
    device="cpu"
)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "DORMIY Retail System API",
        "total_catalog_products": len(retail_system.df_catalog),
        "available_categories": retail_system.categories,
        "supported_workflows": [
            "POST /admin/add-product (Alur 1: Ingestion & Auto-Categorization)",
            "POST /user/search (Alur 2: Text Search & Top-5 Recommendation)"
        ]
    }

# =============================================================================
# ALUR 1: ADMIN (Input Gambar + Nama Produk -> Deteksi Kategori SVM)
# =============================================================================
@app.post("/admin/add-product")
async def admin_add_product(
    file: UploadFile = File(...),
    product_title: str = Form(...),
    auto_save_to_catalog: bool = Form(False)
):
    """
    ALUR 1: Admin memasukkan foto produk baru dan nama produk.
    CLIP mengekstrak vektor visual, dan SVM otomatis mengelompokkan kategori yang cocok.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {str(e)}")

    result = retail_system.admin_add_new_product(
        image_input=image,
        product_title=product_title,
        save_to_catalog=auto_save_to_catalog
    )
    return result

# =============================================================================
# ALUR 2: USER (Input Teks -> Deteksi Kategori SVM -> Top-5 Produk Cosine)
# =============================================================================
@app.post("/user/search")
async def user_search(
    query: str = Form(...),
    top_k: int = Form(5)
):
    """
    ALUR 2: Pengguna mencari barang menggunakan teks bebas.
    Sistem mendeteksi kategori lewat SVM dan mencari Top-5 produk yang paling cocok via Cosine Similarity.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query teks pencarian tidak boleh kosong.")

    result = retail_system.user_search_by_text(
        text_query=query,
        top_k=top_k
    )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
