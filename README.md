# 🇮🇩 Indonesian Economy RAG Chatbot

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rizaikmal7/rag-ekonomi-indonesia)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> 🚀 **Live Demo:** [huggingface.co/spaces/rizaikmal7/rag-ekonomi-indonesia](https://huggingface.co/spaces/rizaikmal7/rag-ekonomi-indonesia)

A **Retrieval-Augmented Generation (RAG)** chatbot for answering questions about the Indonesian economy with accuracy and up-to-date information. It combines official government documents with real-time news to generate factual, source-verified answers — entirely in Bahasa Indonesia.

---

## 📋 Table of Contents

- [Demo](#-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [API Keys](#-api-keys-required)
- [Data Sources](#-data-sources)
- [Author](#-author)

---

## 🎬 Demo

Example questions the chatbot can answer:
- *"What was Indonesia's inflation rate in May 2026?"*
- *"Explain Bank Indonesia's current monetary policy"*
- *"How is the rupiah exchange rate this week?"*
- *"What are foreign exchange reserves and why are they important?"*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Query Router                         │
│         (keyword-based temporal detection)              │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
       Conceptual / Historical       Real-time / Current
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────┐
│      ChromaDB        │    │          NewsAPI             │
│  (BPS / Kemenkeu PDF)│    │   (News from last 7 days)   │
│   1,400+ documents   │    │   Economy relevance filter   │
└──────────┬───────────┘    └──────────────┬──────────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              IndoBERT Embedding                         │
│        (indobenchmark/indobert-base-p1)                 │
│     Semantic search optimized for Bahasa Indonesia      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  LLM Chain (Groq)                       │
│  Primary  : LLaMA 3.3-70B                               │
│  Fallback : Qwen3-32B → LLaMA 3.1-8B → Gemma2-9B       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Final Answer                         │
│          + Data source + Model used                     │
└─────────────────────────────────────────────────────────┘
```

**Auto-Scraping Pipeline (runs every 24 hours):**
```
APScheduler
     │
     ├── BPS API (webapi.bps.go.id)
     │     └── Latest economic press releases → PDF
     │
     └── Kemenkeu Fiskal (fiskal.kemenkeu.go.id)
           └── Fiscal publications → PDF
                    │
                    ▼
             PDF Ingestion
        (chunk → embed → upsert ChromaDB)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Automatically chooses between static PDF documents or real-time news based on query type |
| **IndoBERT Embedding** | Embedding model trained specifically on Bahasa Indonesia — more accurate than generic multilingual models |
| **Auto-Scraping** | PDFs from BPS and Kemenkeu Fiskal are automatically updated every 24 hours without manual intervention |
| **Fallback Chain** | If LLaMA 3.3-70B hits rate limits, automatically switches to Qwen3-32B → LLaMA 3.1-8B → Gemma2-9B |
| **Persistent Storage** | ChromaDB stored in HF Spaces `/data` — survives Space restarts |
| **Query Router** | Automatically detects whether a question needs historical data or current news |
| **Source Attribution** | Every answer cites its document or news source |

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| **Embedding Model** | IndoBERT (`indobenchmark/indobert-base-p1`) | - |
| **Vector Store** | ChromaDB | 0.6.3 |
| **LLM** | LLaMA 3.3-70B via Groq | - |
| **LLM Fallback** | Qwen3-32B, LLaMA 3.1-8B, Gemma2-9B | - |
| **RAG Framework** | LangChain | 0.3.25 |
| **Real-time News** | NewsAPI | - |
| **Auto-Scraping** | BPS API + BeautifulSoup4 | - |
| **Scheduler** | APScheduler | 3.10.4 |
| **UI** | Gradio | 5.33.0 |
| **PDF Processing** | PyPDF | 5.6.0 |
| **Deep Learning** | PyTorch + HuggingFace Transformers | 2.5.1 / 4.52.4 |
| **Deployment** | Hugging Face Spaces | - |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/rizaikmal7/rag-ekonomi-indonesia.git
cd rag-ekonomi-indonesia
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_xxxxxxxxxx        # from console.groq.com (free)
NEWS_API_KEY=xxxxxxxxxx            # from newsapi.org (free, 100 req/day)
BPS_API_KEY=xxxxxxxxxx             # from webapi.bps.go.id (free with registration)

# Local paths (adjust as needed)
VECTORSTORE_PATH=./data/vectorstore
PDF_PATH=./data/pdfs
MODEL_CACHE_PATH=./data/models
```

### 4. Run the application
```bash
python app.py
```

Open your browser at `http://localhost:7860`

> **Note:** On first run, IndoBERT (~440MB) will be downloaded automatically and cached for future sessions.

---

## 📁 Project Structure

```
rag-ekonomi-indonesia/
├── app.py                  # Main application (embedding, retrieval, LLM, UI)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Excludes .env, data/, models/
│
└── notebooks/
    └── RAG_Ekonomi_Indonesia.ipynb   # Step-by-step development notebook (Google Colab)
```

**Data & model directories** (not committed to Git, generated at runtime):
```
data/
├── pdfs/           # Auto-scraped PDF files
├── vectorstore/    # ChromaDB persistent storage
├── models/         # IndoBERT cache (~440MB)
└── logs/
    ├── scraping_log.txt       # Scraping session logs
    └── downloaded_pdfs.txt    # Registry of downloaded PDFs
```

---

## 🔍 How It Works

### 1. Query Router
Every user query is analyzed for keyword signals:
```python
# Conceptual question → ChromaDB (PDF documents)
"What is inflation?"  →  static_score > temporal_score  →  ChromaDB

# Real-time question → NewsAPI (latest news)
"What is this month's inflation?"  →  temporal_score > static_score  →  NewsAPI
```

### 2. IndoBERT Embedding
Text is converted into 768-dimensional vectors using mean pooling:
```
"Bank Indonesia raises interest rates"  →  [0.23, -0.87, 0.41, ...]  (768 numbers)
"BI hikes benchmark rate"               →  [0.21, -0.83, 0.44, ...]  (similar vectors!)
"fried rice recipe"                     →  [-0.91, 0.12, -0.67, ...]  (very different)
```
Cosine similarity between two semantically equivalent Indonesian sentences: **~0.87**

### 3. PDF Chunking Strategy
```
PDF (hundreds of pages)
    ↓
Split: chunk_size=800 chars, overlap=100 chars
    ↓
1 large PDF → ~200–300 chunks
    ↓
Unique ID: MD5(filename + chunk_index + text[:100])
    ↓
ChromaDB upsert (safe to run multiple times)
```

### 4. LLM Fallback Chain
```
Request → LLaMA 3.3-70B
              ↓ (rate limit / error)
         Qwen3-32B
              ↓ (rate limit / error)
         LLaMA 3.1-8B
              ↓ (rate limit / error)
         Gemma2-9B
              ↓ (all failed)
         Informative error message to user
```

---

## 🔑 API Keys Required

| Service | Register | Free Tier |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | ✅ LLaMA 3.3-70B free |
| **NewsAPI** | [newsapi.org/register](https://newsapi.org/register) | ✅ 100 requests/day |
| **BPS API** | [webapi.bps.go.id](https://webapi.bps.go.id) | ✅ Free with registration |

---

## 📊 Data Sources

| Source | Type | Update Frequency |
|---|---|---|
| **BPS** (Statistics Indonesia) | Press releases — inflation, exports/imports, price indices | Automatic via official API |
| **Kemenkeu Fiskal** (Ministry of Finance) | Fiscal publications, APBN reports | Automatic via scraping |
| **NewsAPI** | Indonesian economy news from multiple outlets | Real-time (last 7 days) |

---

## 👤 Author

**Muhammad Ikmal Riza**

- 💼 Specialization: NLP / LLM Engineer
- 🐙 GitHub: [@rizaikmal7](https://github.com/rizaikmal7)
- 📍 Tasikmalaya, West Java, Indonesia

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use and modify with attribution.
