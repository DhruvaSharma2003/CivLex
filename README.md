# ⚖️ CivLex — Plain-Law Assistant

Demystify legal PDFs: ask grounded questions with citations, get plain-language rewrites (English/Hindi), compare versions, extract dates, and export a 2-page brief.  
**Note:** This is an educational prototype and **not legal advice**.

---

## ✨ Key Features (USPs)
- **Grounded Q&A with citations** — answers reference numbered context blocks `[1] [2]` from the uploaded PDF.
- **Clause-level plain-language rewrite** — grade-controlled, with optional **bilingual (English + Hindi)** output.
- **Version diff** — sentence-level highlights for additions/removals between two PDFs.
- **Dates → quick validity hints** — extracts and lists relevant date mentions.
- **One-click brief export** — generates a concise 2-page PDF summary with selected Q&A and dates.

---

## 🧱 Tech Stack
- **Frontend/App:** Streamlit (Python)
- **Parsing:** PyMuPDF (fitz)
- **Retrieval:** Sentence-Transformers embeddings + FAISS 
- **Generation (optional):** OpenAI Chat Completions
- **Export:** ReportLab

---

## 🚀 Quickstart

### 1) Clone & install
```bash
git clone https://github.com/<your-username>/CivLex.git
cd CivLex
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
