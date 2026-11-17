"""
server.py
=========
Production FastAPI Server untuk Sistem Ritel Modern:
- Endpoint Alur 1: POST /admin/add-product (Upload Gambar + Nama Produk -> Deteksi Kategori SVM -> Tambah ke Katalog)
- Endpoint Alur 2: POST /user/search (Query Teks -> Auto-Translate -> Deteksi Kategori -> Top-K Visual Retrieval)
- Model: Fine-Tuned CLIP Projection Head + Multimodal SVM (17 Kategori Bersih).
"""

import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

# Import service engine
from inventory_service import InventoryService

app = FastAPI(
    title="Production Retail Semantic Search & Categorization API",
    description="Backend API Service: Alur Admin Ingestion & Alur User Semantic Search (Best Architecture)",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Service Engine Tunggal
print("Memuat InventoryService ke memori server...")
service = InventoryService()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Retail Multi-Modal API (Production)",
        "total_categories": len(service.get_categories()),
        "catalog_size": service.get_catalog_count(),
        "endpoints": {
            "admin_ingestion": "POST /admin/add-product",
            "user_search": "POST /user/search"
        }
    }

@app.get("/categories")
def get_categories():
    return {
        "total": len(service.get_categories()),
        "categories": service.get_categories()
    }

@app.post("/admin/add-product")
async def admin_add_product(
    title: str = Form(..., description="Nama atau judul lengkap produk"),
    product_id: str = Form(None, description="Kode ID produk opsional"),
    image: UploadFile = File(..., description="Foto produk fisik")
):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {str(e)}")

    res = service.add_product(image=pil_img, title=title, product_id=product_id)
    return res

@app.post("/user/search")
async def user_search(
    query: str = Form(..., description="Kata kunci pencarian produk bebas (Bahasa Indonesia / Inggris)"),
    top_k: int = Form(5, description="Jumlah rekomendasi produk teratas"),
    auto_translate: bool = Form(True, description="Terjemahkan otomatis query jika non-Inggris")
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query pencarian tidak boleh kosong")

    res = service.search_products(query=query, top_k=top_k, auto_translate=auto_translate)
    return res

@app.post("/user/search-image")
async def user_search_image(
    image: UploadFile = File(..., description="Foto produk dari pengguna untuk dicocokkan dengan gambar di database"),
    top_k: int = Form(5, description="Jumlah rekomendasi produk teratas")
):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {str(e)}")

    res = service.search_by_image(image=pil_img, top_k=top_k)
    return res

@app.get("/image/{product_id}")
def get_product_image(product_id: str):
    """Endpoint untuk menampilkan foto fisik produk secara langsung."""
    img_bytes = service.get_image_bytes(product_id)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="Gambar produk tidak ditemukan")
    return Response(content=img_bytes, media_type="image/jpeg")

@app.get("/search-ui", response_class=HTMLResponse)
def visual_search_ui():
    """Tampilan web visual interaktif untuk menguji pencarian: Mode Teks ATAU Mode Gambar."""
    html_content = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Dual Search: Text & Image Similarity (CLIP + SVM)</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f1f5f9; margin: 0; padding: 25px; }
            .container { max-width: 980px; margin: auto; background: white; padding: 25px 35px; border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }
            h2 { color: #0f172a; margin-top: 0; }
            .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
            .tab-btn { background: none; border: none; padding: 10px 20px; font-size: 15px; font-weight: bold; color: #64748b; cursor: pointer; border-radius: 8px; transition: 0.2s; }
            .tab-btn.active { background: #2563eb; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .search-box { display: flex; gap: 10px; margin-bottom: 25px; }
            input[type="text"] { flex: 1; padding: 12px 16px; font-size: 15px; border: 2px solid #cbd5e1; border-radius: 8px; outline: none; }
            input[type="text"]:focus { border-color: #3b82f6; }
            input[type="file"] { padding: 10px; border: 2px dashed #cbd5e1; border-radius: 8px; flex: 1; }
            button.action-btn { background: #2563eb; color: white; border: none; padding: 12px 24px; font-size: 15px; font-weight: bold; border-radius: 8px; cursor: pointer; transition: 0.2s; }
            button.action-btn:hover { background: #1d4ed8; }
            .meta-info { background: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px 18px; border-radius: 0 8px 8px 0; margin-bottom: 25px; display: none; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
            .card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s, box-shadow 0.2s; }
            .card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.08); }
            .img-container { width: 100%; height: 210px; background: #f8fafc; display: flex; align-items: center; justify-content: center; overflow: hidden; }
            .img-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
            .card-body { padding: 15px; display: flex; flex-direction: column; flex: 1; justify-content: space-between; }
            .badge { display: inline-block; padding: 4px 8px; font-size: 11px; font-weight: bold; border-radius: 4px; background: #e0f2fe; color: #0369a1; margin-bottom: 8px; }
            .title { font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 10px; line-height: 1.4; }
            .match-score { font-size: 12px; font-weight: bold; color: #059669; }
            .preview-box { margin-bottom: 15px; display: none; }
            .preview-box img { max-height: 140px; border-radius: 8px; border: 1px solid #cbd5e1; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🛒 Dual Semantic Product Search (CLIP + SVM)</h2>
            <p style="color: #64748b; margin-bottom: 20px;">Temukan produk di katalog menggunakan <strong>Teks Bahasa Alami</strong> ATAU <strong>Upload Foto/Gambar Produk</strong>.</p>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('textTab', this)">🔍 Cari via Teks</button>
                <button class="tab-btn" onclick="switchTab('imageTab', this)">📷 Cari via Foto Produk</button>
            </div>

            <!-- TAB 1: SEARCH BY TEXT -->
            <div id="textTab" class="tab-content active">
                <div class="search-box">
                    <input type="text" id="queryInput" placeholder="Contoh: gitar akustik fender, charger kabel type c, obeng perkakas bengkel..." value="senar gitar akustik fender">
                    <button class="action-btn" onclick="doTextSearch()">Cari Teks</button>
                </div>
            </div>

            <!-- TAB 2: SEARCH BY IMAGE -->
            <div id="imageTab" class="tab-content">
                <div class="search-box">
                    <input type="file" id="fileInput" accept="image/*" onchange="previewImage(this)">
                    <button class="action-btn" onclick="doImageSearch()">Cocokkan Gambar</button>
                </div>
                <div id="previewBox" class="preview-box">
                    <div style="font-size: 12px; font-weight: bold; margin-bottom: 5px; color: #64748b;">Preview Foto yang Dicari:</div>
                    <img id="previewImg" src="">
                </div>
            </div>

            <!-- META INFO -->
            <div id="metaInfo" class="meta-info">
                <div id="transRow"><strong>🌐 Terjemahan Query:</strong> <span id="transText"></span></div>
                <div style="margin-top: 5px;"><strong>🎯 Kategori Terdeteksi (SVM):</strong> <span id="catText" style="color: #2563eb; font-weight: bold;"></span></div>
                <div id="searchTypeRow" style="margin-top: 5px; font-size: 13px; color: #64748b;"><strong>⚡ Metode:</strong> <span id="searchTypeText"></span></div>
            </div>

            <!-- RESULTS -->
            <div id="resultGrid" class="grid"></div>
        </div>

        <script>
            function switchTab(tabId, btn) {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }

            function previewImage(input) {
                if (input.files && input.files[0]) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('previewImg').src = e.target.result;
                        document.getElementById('previewBox').style.display = 'block';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            }

            function renderResults(data, searchMode) {
                document.getElementById('metaInfo').style.display = 'block';
                if(data.user_query_translated) {
                    document.getElementById('transRow').style.display = 'block';
                    document.getElementById('transText').innerText = data.user_query_translated;
                } else {
                    document.getElementById('transRow').style.display = 'none';
                }

                document.getElementById('catText').innerText = `${data.detected_category.category} (${data.detected_category.confidence_pct}%)`;
                document.getElementById('searchTypeText').innerText = searchMode;

                const grid = document.getElementById('resultGrid');
                grid.innerHTML = '';
                data.retrieved_products.forEach(p => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <div class="img-container">
                            <img src="${p.image_url}" alt="${p.title}" onerror="this.src='https://via.placeholder.com/200?text=No+Image'">
                        </div>
                        <div class="card-body">
                            <div>
                                <span class="badge">${p.category}</span>
                                <div class="title">${p.title}</div>
                            </div>
                            <div class="match-score">⚡ Vector Similarity: ${p.match_percentage}%</div>
                        </div>
                    `;
                    grid.appendChild(card);
                });
            }

            async function doTextSearch() {
                const q = document.getElementById('queryInput').value;
                if(!q.trim()) return;
                const grid = document.getElementById('resultGrid');
                grid.innerHTML = '<div style="color: #64748b; font-style: italic;">Sedang mencocokkan vektor teks ke gambar...</div>';
                
                const formData = new FormData();
                formData.append('query', q);
                formData.append('top_k', 6);
                formData.append('auto_translate', 'true');
                
                try {
                    const res = await fetch('/user/search', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResults(data, "Text-to-Image (Teks diterjemahkan -> Cosine Sim CLIP)");
                } catch(e) {
                    grid.innerHTML = '<div style="color: red;">Error: ' + e + '</div>';
                }
            }

            async function doImageSearch() {
                const fileInput = document.getElementById('fileInput');
                if(!fileInput.files || !fileInput.files[0]) {
                    alert("Pilih file foto/gambar produk terlebih dahulu!");
                    return;
                }
                const grid = document.getElementById('resultGrid');
                grid.innerHTML = '<div style="color: #64748b; font-style: italic;">Sedang mengekstrak vektor visual dan mencocokkan gambar di database...</div>';
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                formData.append('top_k', 6);
                
                try {
                    const res = await fetch('/user/search-image', { method: 'POST', body: formData });
                    const data = await res.json();
                    renderResults(data, "Image-to-Image (Visual Embedding CLIP -> Cosine Similarity)");
                } catch(e) {
                    grid.innerHTML = '<div style="color: red;">Error: ' + e + '</div>';
                }
            }

            window.onload = doTextSearch;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
