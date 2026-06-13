# =============================================================
# RAG Chatbot Ekonomi Indonesia
# Author : Muhammad Ikmal Riza
# Stack  : IndoBERT · ChromaDB · LLaMA 3.3-70B (Groq) · NewsAPI · Gradio
# Deploy : Hugging Face Spaces
# =============================================================

import os
import re
import glob
import time
import hashlib
import logging
import requests
import numpy as np
import torch
import chromadb
import gradio as gr
import threading

from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from chromadb.config import Settings
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================
# KONFIGURASI PATH
# Di HF Spaces tidak ada Google Drive — pakai persistent storage
# /data adalah direktori persistent di HF Spaces (tidak reset saat restart)
# =============================================================
load_dotenv()

BASE_DIR        = "/data/rag-ekonomi-indonesia"
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", f"{BASE_DIR}/vectorstore")
PDF_PATH         = os.getenv("PDF_PATH",          f"{BASE_DIR}/pdfs")
MODEL_CACHE_PATH = os.getenv("MODEL_CACHE_PATH",  f"{BASE_DIR}/models")
LOG_PATH         = f"{BASE_DIR}/logs"
REGISTRY_FILE    = f"{LOG_PATH}/downloaded_pdfs.txt"

for d in [VECTORSTORE_PATH, PDF_PATH, MODEL_CACHE_PATH, LOG_PATH]:
    os.makedirs(d, exist_ok=True)

# API Keys — diset via HF Spaces Secrets (Settings → Variables and secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BPS_API_KEY  = os.getenv("BPS_API_KEY")

logger.info("✅ Konfigurasi path selesai")

# =============================================================
# INDOBERT EMBEDDING
# =============================================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
device     = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"⏳ Memuat IndoBERT ({device.upper()})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
bert_model = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_PATH,
    use_safetensors=True,
).to(device)
bert_model.eval()

logger.info("✅ IndoBERT berhasil dimuat")


class IndoBERTEmbeddings(Embeddings):
    """Custom LangChain-compatible embedding menggunakan IndoBERT."""

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings    = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * input_mask_expanded).sum(1) / \
               input_mask_expanded.sum(1).clamp(min=1e-9)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = bert_model(**encoded)
        emb = self._mean_pooling(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)


embeddings = IndoBERTEmbeddings()

# =============================================================
# CHROMADB — PERSISTENT DI /data
# =============================================================
chroma_client = chromadb.PersistentClient(
    path=VECTORSTORE_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    name="ekonomi_indonesia",
    metadata={"hnsw:space": "cosine"},
)
logger.info(f"✅ ChromaDB siap — {collection.count()} dokumen")

# =============================================================
# PDF INGESTION PIPELINE
# =============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)


def load_and_chunk_pdf(pdf_path: str) -> list:
    loader   = PyPDFLoader(pdf_path)
    pages    = loader.load()
    chunks   = text_splitter.split_documents(pages)
    filename = os.path.basename(pdf_path)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "filename":     filename,
            "chunk_index":  i,
            "total_chunks": len(chunks),
        })
    return chunks


def ingest_pdfs(pdf_dir: str) -> dict:
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        logger.info("⚠️  Tidak ada PDF ditemukan untuk diingest")
        return {"total_files": 0, "total_chunks": 0}

    all_chunks = []
    stats      = {"total_files": len(pdf_files), "total_chunks": 0}

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            chunks = load_and_chunk_pdf(pdf_path)
            all_chunks.extend(chunks)
            logger.info(f"  ✅ {filename}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"  ❌ {filename}: {e}")

    if not all_chunks:
        return stats

    BATCH_SIZE = 32
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch     = all_chunks[i:i + BATCH_SIZE]
        texts     = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]
        vectors   = embeddings.embed_documents(texts)
        ids = [
            hashlib.md5(
                f"{meta.get('filename','')}-{meta.get('chunk_index','')}-{text[:100]}"
                .encode()
            ).hexdigest()[:16]
            for text, meta in zip(texts, metadatas)
        ]
        collection.upsert(
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    stats["total_chunks"] = len(all_chunks)
    logger.info(f"✅ Ingestion selesai — {collection.count()} dokumen di ChromaDB")
    return stats

# =============================================================
# AUTO-SCRAPING PDF
# =============================================================
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
    "Referer": "https://www.google.com/",
}

EKONOMI_SUBJ_BPS = [
    "inflasi", "harga", "ekonomi", "pdb", "gdp", "ekspor", "impor",
    "neraca", "perdagangan", "kemiskinan", "upah", "tenaga kerja",
    "indeks", "keuangan", "fiskal", "moneter", "produksi",
]


def load_registry() -> set:
    if not os.path.exists(REGISTRY_FILE):
        return set()
    with open(REGISTRY_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def save_to_registry(filename: str):
    with open(REGISTRY_FILE, "a") as f:
        f.write(filename + "\n")


def download_pdf(url: str, filename: str, registry: set) -> bool:
    if filename in registry:
        return False
    filepath = os.path.join(PDF_PATH, filename)
    if os.path.exists(filepath):
        save_to_registry(filename)
        return False
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=30)
        resp.raise_for_status()
        if not (resp.content[:4] == b"%PDF" or
                "pdf" in resp.headers.get("Content-Type", "")):
            return False
        with open(filepath, "wb") as f:
            f.write(resp.content)
        save_to_registry(filename)
        size_kb = len(resp.content) / 1024
        logger.info(f"    ✅ {filename} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        logger.error(f"    ❌ Gagal: {filename} — {e}")
        return False


def scrape_kemenkeu_fiskal(registry: set) -> int:
    logger.info("  💰 Scraping Kemenkeu Fiskal...")
    base_url  = "https://fiskal.kemenkeu.go.id"
    new_count = 0
    try:
        resp = requests.get(f"{base_url}/publikasi", headers=SCRAPE_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            if href.startswith("/"):
                href = base_url + href
            elif not href.startswith("http"):
                href = base_url + "/" + href
            url_hash  = hashlib.md5(href.encode()).hexdigest()[:8]
            orig_name = href.split("/")[-1].replace("%20", "_")[:50]
            filename  = f"kemenkeu_{url_hash}_{orig_name}"
            if download_pdf(href, filename, registry):
                new_count += 1
                registry.add(filename)
                time.sleep(1.5)
            if new_count >= 8:
                break
    except Exception as e:
        logger.error(f"  ❌ Kemenkeu error: {e}")
    logger.info(f"  → {new_count} PDF baru dari Kemenkeu Fiskal")
    return new_count


def scrape_bps_api(registry: set) -> int:
    logger.info("  📊 Scraping BPS via API...")
    if not BPS_API_KEY:
        logger.warning("  ⚠️  BPS_API_KEY tidak tersedia — skip")
        return 0
    new_count = 0
    try:
        url  = (f"https://webapi.bps.go.id/v1/api/list/model/pressrelease"
                f"/domain/0000/lang/ind/key/{BPS_API_KEY}")
        resp = requests.get(url, timeout=15)
        data = resp.json()
        releases = data["data"][1]
        for release in releases:
            title_lower = release.get("title", "").lower()
            if not any(kw in title_lower for kw in EKONOMI_SUBJ_BPS):
                continue
            pdf_url = release.get("pdf", "")
            if not pdf_url:
                continue
            if not pdf_url.startswith("http"):
                pdf_url = "https://www.bps.go.id" + pdf_url
            brs_id   = release.get("brs_id", "unknown")
            url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
            filename = f"bps_{brs_id}_{url_hash}.pdf"
            if download_pdf(pdf_url, filename, registry):
                new_count += 1
                registry.add(filename)
            time.sleep(1)
            if new_count >= 8:
                break
    except Exception as e:
        logger.error(f"  ❌ BPS API error: {e}")
    logger.info(f"  → {new_count} PDF baru dari BPS")
    return new_count


def run_scraping_pipeline():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"🔄 Auto-Scraping dimulai: {timestamp}")
    registry  = load_registry()
    total_new = scrape_kemenkeu_fiskal(registry) + scrape_bps_api(registry)
    logger.info(f"📥 Total PDF baru: {total_new}")
    if total_new > 0:
        ingest_pdfs(PDF_PATH)
    with open(f"{LOG_PATH}/scraping_log.txt", "a") as f:
        f.write(f"{timestamp} | Baru: {total_new} | ChromaDB: {collection.count()}\n")


def startup_tasks():
    """Jalankan ingestion + scraping di background — tidak blokir Gradio."""
    if collection.count() == 0:
        logger.info("🔄 ChromaDB kosong — ingestion awal di background...")
        ingest_pdfs(PDF_PATH)
    run_scraping_pipeline()

# Background thread — Gradio langsung launch tanpa nunggu ini selesai
thread = threading.Thread(target=startup_tasks, daemon=True)
thread.start()

scheduler = BackgroundScheduler(timezone="Asia/Jakarta")
scheduler.add_job(
    func=run_scraping_pipeline,
    trigger=IntervalTrigger(hours=24),
    id="pdf_scraping",
    replace_existing=True,
)
scheduler.start()
logger.info("✅ Scheduler aktif — update setiap 24 jam")

# =============================================================
# NEWSAPI RETRIEVAL
# =============================================================
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

EKONOMI_KEYWORDS = [
    "ekonomi", "inflasi", "rupiah", "bank indonesia", "bi rate",
    "gdp", "pdb", "ekspor", "impor", "investasi", "saham", "ihsg",
    "suku bunga", "devisa", "apbn", "fiskal", "moneter", "bps",
    "pertumbuhan", "kemenkeu", "ojk", "neraca", "perdagangan",
    "kurs", "dolar", "valas", "obligasi", "sbsn", "sri", "yield",
]

STOPWORDS = [
    "apa", "itu", "dan", "atau", "yang", "untuk", "dari", "ke",
    "di", "dengan", "adalah", "bagaimana", "mengapa", "kenapa",
    "berapa", "kapan", "siapa", "jelaskan", "ceritakan", "tolong",
    "penting", "bagi", "tentang", "mengenai", "terkait", "mohon",
    "apakah", "sebutkan", "kondisi", "situasi", "update", "berita",
    "kabar", "terbaru", "terkini", "saat", "ini", "sekarang",
]


def extract_keywords(query: str) -> str:
    words    = query.lower().split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 3]
    core     = " ".join(keywords[:3])
    if "indonesia" not in core.lower():
        core += " Indonesia"
    return core


def is_ekonomi_relevant(article: dict) -> bool:
    text = (article["title"] + " " + article["content"]).lower()
    return any(kw in text for kw in EKONOMI_KEYWORDS)


def fetch_economic_news(query: str, max_articles: int = 5) -> list:
    if not newsapi:
        return []
    date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    keywords  = extract_keywords(query)
    try:
        resp     = newsapi.get_everything(
            q=f"{keywords} ekonomi",
            from_param=date_from,
            sort_by="relevancy",
            page_size=max_articles * 2,
        )
        articles = resp.get("articles", []) if resp.get("status") == "ok" else []
    except Exception:
        articles = []

    if not articles:
        try:
            resp     = newsapi.get_everything(
                q="ekonomi Indonesia",
                from_param=date_from,
                sort_by="publishedAt",
                page_size=max_articles * 2,
            )
            articles = resp.get("articles", []) if resp.get("status") == "ok" else []
        except Exception:
            return []

    results = []
    for article in articles:
        title       = article.get("title", "")       or ""
        description = article.get("description", "") or ""
        content     = article.get("content", "")     or ""
        full_text   = " ".join(f"{title}. {description} {content}".split())
        if len(full_text) < 50:
            continue
        d = {
            "title":        title,
            "content":      full_text,
            "url":          article.get("url", ""),
            "published_at": article.get("publishedAt", ""),
            "source":       article.get("source", {}).get("name", "Unknown"),
        }
        if is_ekonomi_relevant(d):
            results.append(d)
        if len(results) >= max_articles:
            break
    return results


def format_news_context(articles: list) -> str:
    if not articles:
        return "Tidak ada berita ekonomi terkini yang ditemukan."
    parts = []
    for i, a in enumerate(articles, 1):
        try:
            dt      = datetime.fromisoformat(a["published_at"].replace("Z", "+00:00"))
            tanggal = dt.strftime("%d %B %Y")
        except Exception:
            tanggal = a["published_at"][:10] if a["published_at"] else "Tanggal tidak diketahui"
        parts.append(
            f"[Berita {i}] {a['source']} — {tanggal}\n"
            f"Judul: {a['title']}\n"
            f"Isi: {a['content']}\n"
            f"URL: {a['url']}"
        )
    return "\n\n".join(parts)

# =============================================================
# QUERY ROUTER
# =============================================================
class QueryRoute(Enum):
    CHROMADB = "chromadb"
    NEWSAPI  = "newsapi"


TEMPORAL_KEYWORDS = [
    "hari ini", "kemarin", "minggu ini", "bulan ini", "tahun ini",
    "pekan ini", "tadi", "baru saja", "terkini", "terbaru",
    "sekarang", "saat ini", "kini",
    "naik", "turun", "menguat", "melemah", "anjlok", "meroket",
    "reaksi", "respon", "dampak", "efek",
    "2025", "2026", "q1", "q2", "q3", "q4",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
    "berita", "kabar", "update", "perkembangan", "kondisi",
    "prediksi", "proyeksi", "forecast",
]

STATIC_KEYWORDS = [
    "apa itu", "jelaskan", "pengertian", "definisi",
    "bagaimana cara", "mekanisme", "kebijakan", "regulasi",
    "undang-undang", "peraturan", "sejarah", "historis",
]


def route_query(query: str) -> QueryRoute:
    q = query.lower()
    t = sum(1 for kw in TEMPORAL_KEYWORDS if kw in q)
    s = sum(1 for kw in STATIC_KEYWORDS   if kw in q)
    return QueryRoute.NEWSAPI if t > s else QueryRoute.CHROMADB


def retrieve_context(query: str, n_results: int = 3) -> tuple:
    route = route_query(query)

    if route == QueryRoute.NEWSAPI:
        articles = fetch_economic_news(query, max_articles=n_results)
        return format_news_context(articles), route

    if collection.count() == 0:
        articles = fetch_economic_news(query, max_articles=n_results)
        return format_news_context(articles), QueryRoute.NEWSAPI

    query_vector = embeddings.embed_query(query)
    results      = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    parts = []
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ), 1):
        sim    = 1 - (dist / 2)
        source = meta.get("filename", "Dokumen")
        page   = meta.get("page", "?")
        parts.append(f"[Dokumen {i}] {source} — Hal. {page} (similarity: {sim:.3f})\n{doc}")
    return "\n\n".join(parts), route

# =============================================================
# LLM CHAIN — GROQ + FALLBACK
# =============================================================
MODELS = [
    {"name": "llama-3.3-70b-versatile", "label": "LLaMA 3.3-70B"},
    {"name": "qwen-qwq-32b",            "label": "Qwen3-32B"},
    {"name": "llama-3.1-8b-instant",    "label": "LLaMA 3.1-8B"},
    {"name": "gemma2-9b-it",            "label": "Gemma2-9B"},
]

SYSTEM_PROMPT = """Kamu adalah asisten ekonomi Indonesia yang berpengetahuan luas.
Tugasmu adalah menjawab pertanyaan tentang ekonomi Indonesia berdasarkan konteks yang diberikan.

Aturan:
1. Jawab HANYA berdasarkan konteks yang diberikan — jangan mengarang fakta
2. Jika konteks tidak cukup untuk menjawab, katakan dengan jelas
3. Gunakan Bahasa Indonesia yang formal namun mudah dipahami
4. Sertakan angka/data spesifik jika tersedia di konteks
5. Sebutkan sumber informasi (nama dokumen atau media) di akhir jawaban
6. Jawaban maksimal 3-4 paragraf — padat dan informatif"""


def ask(query: str) -> dict:
    context, route = retrieve_context(query)
    source_label   = "Berita Terkini" if route == QueryRoute.NEWSAPI else "Dokumen Referensi"
    prompt         = (
        f"Berdasarkan {source_label} berikut, jawab pertanyaan pengguna.\n\n"
        f"=== {source_label} ===\n{context}\n\n"
        f"=== Pertanyaan ===\n{query}\n\n=== Jawaban ==="
    )
    answer     = None
    model_used = None
    for m in MODELS:
        try:
            llm      = ChatGroq(api_key=GROQ_API_KEY, model=m["name"],
                                temperature=0.2, max_tokens=1024)
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            answer     = response.content
            model_used = m["label"]
            break
        except Exception as e:
            logger.warning(f"  {m['label']} gagal: {e}")
            continue

    if answer is None:
        answer     = "Maaf, semua model tidak tersedia saat ini. Coba beberapa saat lagi."
        model_used = "none"

    return {"answer": answer, "route": route.value, "model_used": model_used}

# =============================================================
# GRADIO UI
# =============================================================
EXAMPLE_QUESTIONS = [
    "Apa itu inflasi dan bagaimana cara mengukurnya?",
    "Bagaimana kondisi nilai tukar rupiah terkini?",
    "Jelaskan kebijakan moneter Bank Indonesia",
    "Apa berita ekonomi Indonesia terbaru?",
    "Berapa cadangan devisa Indonesia saat ini?",
    "Bagaimana pertumbuhan ekonomi Indonesia tahun ini?",
]


def chat(message: str, history: list) -> str:
    if not message.strip():
        return "Silakan masukkan pertanyaan tentang ekonomi Indonesia."
    result      = ask(message)
    route_label = "📰 Berita Real-time" if result["route"] == "newsapi" \
                  else "📚 Dokumen Referensi"
    timestamp   = datetime.now().strftime("%H:%M:%S")
    return (
        f"{result['answer']}\n\n"
        f"---\n"
        f"*Sumber: {route_label} · Model: {result['model_used']} · {timestamp}*"
    )


with gr.Blocks(title="RAG Ekonomi Indonesia", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🇮🇩 RAG Chatbot Ekonomi Indonesia
    Chatbot berbasis **Retrieval-Augmented Generation (RAG)** untuk menjawab
    pertanyaan seputar ekonomi Indonesia.

    **Sumber data:**
    - 📚 Dokumen resmi BPS & Kemenkeu (diperbarui otomatis tiap 24 jam)
    - 📰 Berita ekonomi real-time (via NewsAPI)

    **Model:** LLaMA 3.3-70B via Groq · **Embedding:** IndoBERT
    """)

    gr.ChatInterface(
        fn=chat,
        type="messages",
        chatbot=gr.Chatbot(
            height=450,
            placeholder="Tanyakan seputar ekonomi Indonesia...",
            show_label=False,
            type="messages",
        ),
        textbox=gr.Textbox(
            placeholder="Contoh: Apa itu inflasi dan bagaimana dampaknya?",
            container=False,
            scale=7,
        ),
        examples=EXAMPLE_QUESTIONS,
        cache_examples=False,
    )

    gr.Markdown("""
    ---
    💡 **Tips:** Tanya tentang kebijakan, data makroekonomi, atau berita terkini.
    Untuk data real-time (hari ini, minggu ini), chatbot otomatis menggunakan sumber berita.

    *Muhammad Ikmal Riza · 2026*
    """)

if __name__ == "__main__":
    demo.launch()
