# app.py — CivLex (Hackathon Prototype, Gemini Edition)
# Demystify legal PDFs with retrieval-augmented answers, plain-language rewrites (EN/HI),
# version diff, date timeline, and 2-page brief export.
# NOTE: Educational prototype; NOT legal advice.

from __future__ import annotations

import os, io, re, time, difflib, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from datetime import datetime
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Optional vector index; auto-fallback to NumPy if unavailable
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# Optional PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------------------- Config ---------------------------------

APP_NAME = "CivLex — Plain-Law Assistant"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "gemini-2.5-flash"  # fast & capable for hackathons

# Provider: Gemini via google-genai
GEMINI_KEY = None
try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    pass
GEMINI_KEY = GEMINI_KEY or os.getenv("GEMINI_API_KEY")

# ------------------------------- Styling --------------------------------

st.set_page_config(page_title=APP_NAME, page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
/* overall polish */
[data-testid="stSidebar"] {width: 360px;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
h1,h2,h3,h4 {font-weight: 700;}
.small-note {opacity: 0.8; font-size: 0.9rem;}
.kpi-card {border:1px solid rgba(120,120,120,0.2); border-radius:16px; padding:14px 16px; background:rgba(240,240,240,0.25);}
.codechip {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
           background:#f6f6f6; border:1px solid #eee; padding:2px 8px; border-radius:8px;}
.diff-box {border:1px solid rgba(120,120,120,0.2); border-radius:12px; padding:10px; background: #fff;}
.confidence {font-weight:600;}
ul.compact>li{margin-bottom:0.25rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("## ⚖️ CivLex — Demystify Legal Documents")
st.caption("Ask grounded questions with citations, simplify clauses (English/Hindi), compare versions, and export a brief. **Not legal advice.**")

# ------------------------------- Utilities -------------------------------

def timer() -> Tuple[callable, callable]:
    """Simple context-free timer."""
    start = time.perf_counter()
    def elapsed() -> float:
        return (time.perf_counter() - start) * 1000.0
    def reset():
        nonlocal start
        start = time.perf_counter()
    return elapsed, reset

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T

DATE_REGEXPS = [
    r"\b\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{2,4}\b",
    r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b"
]

def highlight(text: str, term: str) -> str:
    if not term: return text
    pat = re.compile(re.escape(term), re.IGNORECASE)
    return pat.sub(lambda m: f"**{m.group(0)}**", text)

# ------------------------------- Embeddings ------------------------------

@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL_NAME)

def chunk_text(text: str, max_chars=900, overlap=150) -> List[str]:
    # Paragraph-aware chunking with overlap
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf: chunks.append(buf)
            tail = buf[-overlap:] if overlap < len(buf) else ""
            buf = (tail + "\n\n" + p).strip()
            while len(buf) > max_chars:
                chunks.append(buf[:max_chars])
                buf = buf[max_chars - overlap:]
    if buf: chunks.append(buf)
    return chunks

def extract_text_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append((i + 1, text))
    return pages

def build_corpus(pages: List[Tuple[int, str]]) -> List[Dict]:
    corpus, cid = [], 0
    for pg, text in pages:
        for ch in chunk_text(text):
            corpus.append({"id": cid, "page": pg, "text": ch})
            cid += 1
    return corpus

def embed_corpus(corpus: List[Dict]) -> Tuple[np.ndarray, Optional[object]]:
    model = load_embedder()
    texts = [c["text"] for c in corpus]
    X = model.encode(texts, normalize_embeddings=True)
    index = None
    if HAVE_FAISS and len(X) > 0:
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X.astype(np.float32))
    return X, index

def search(corpus: List[Dict], X: np.ndarray, index: Optional[object], query: str, k=5) -> List[Tuple[Dict, float]]:
    model = load_embedder()
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    if index is not None:
        scores, ids = index.search(q, min(k, len(corpus)))
        ids = ids[0].tolist()
        scores = scores[0].tolist()
    else:
        sims = (cosine_sim(X, q)).reshape(-1)
        ids = np.argsort(-sims)[:k].tolist()
        scores = [float(sims[i]) for i in ids]
    hits = [({**corpus[i]}, scores[j]) for j, i in enumerate(ids)]
    return hits

def similarity_confidence(scores: List[float]) -> float:
    """Map cosine sims (approx -1..1) to 0..1 confidence."""
    if not scores: return 0.0
    # ignore negatives, cap to [0,1]
    vals = [max(0.0, min(1.0, (s + 1) / 2)) for s in scores]
    return float(np.mean(vals))

# ------------------------------- Gemini LLM ------------------------------

def call_gemini(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Calls Gemini via google-genai; raises RuntimeError if no key or API error."""
    if not GEMINI_KEY:
        raise RuntimeError("no_gemini_key")
    try:
        # Lightweight import to avoid hard dependency when running in fallback
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        resp = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={"system_instruction": system_prompt, "temperature": 0.2}
        )
        # google-genai returns .text for concatenated parts
        return (resp.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"gemini_error: {e}")

def grounded_answer(question: str, hits: List[Tuple[Dict, float]], lang: str) -> str:
    # Build context with bracketed citations
    ctx_parts = []
    for i, (chunk, score) in enumerate(hits, start=1):
        ctx_parts.append(f"[{i}] (page {chunk['page']})\n{chunk['text']}\n")
    context = "\n\n".join(ctx_parts)

    system = (
        "You are CivLex, a legal document explainer. "
        "Answer ONLY using the provided context. "
        "Cite brackets like [1], [2] that map to the context sections. "
        "If the answer isn't in context, say you don't know and suggest where it might be in the document. "
        "Be concise and accurate; do not hallucinate."
    )
    if lang.startswith("Hindi"):
        system += " Respond in simple Hindi."

    user = f"Question: {question}\n\nContext:\n{context}"

    # Guardrail: if retrieval is weak, prefer safe fallback
    scores = [s for _, s in hits]
    conf = similarity_confidence(scores)
    if conf < 0.18:  # conservative threshold for legal text
        best = "\n\n".join([f"[{i}] p.{h[0]['page']}: {h[0]['text']}"
                            for i, h in enumerate(hits, start=1)])
        return ("I'm not confident the answer is supported by this document. "
                "Here are the most relevant excerpts. Please verify:\n\n" + best)

    # Try Gemini; if unavailable, extractive fallback
    try:
        return call_gemini(system, user)
    except RuntimeError:
        best = "\n\n".join([f"[{i}] p.{h[0]['page']}: {h[0]['text']}"
                            for i, h in enumerate(hits, start=1)])
        return ("(Generator unavailable) Most relevant excerpts:\n\n" + best)

def eli_rewrite(text: str, grade: int, lang: str, bilingual: bool) -> str:
    # Fallback simplifier if no key
    if not GEMINI_KEY:
        short = " ".join(s.strip() for s in re.split(r"[.;]\s+", text)[:3])
        return f"(Fallback) Simple version: {short}"

    lanes = "Hindi" if lang.startswith("Hindi") else "English"
    sys = (
        f"Rewrite the passage for a reader at approximately grade {grade}. "
        "Use plain language, short sentences, and bullet points if appropriate. "
        "Preserve legal meaning; do not invent facts."
    )
    if bilingual:
        sys += " Provide both English and Hindi versions."

    prompt = f"Rewrite this ({lanes} preferred). Text:\n\n{text}"
    try:
        return call_gemini(sys, prompt)
    except RuntimeError as e:
        return "(Model error during rewrite.)"

# ------------------------------- Dates / Diff ----------------------------

def find_dates(all_text: str) -> List[str]:
    found = set()
    for rx in DATE_REGEXPS:
        for m in re.findall(rx, all_text, flags=re.IGNORECASE):
            found.add(m if isinstance(m, str) else m[0])

    def key_fn(s: str) -> datetime:
        for fmt in ("%d/%m/%Y","%d-%m-%Y","%Y-%m-%d","%d %b %Y","%d %B %Y"):
            try:
                return datetime.strptime(s.replace('.', ''), fmt)
            except Exception:
                pass
        return datetime(1900,1,1)

    return [d for d in sorted(found, key=key_fn)]

def diff_text(a: str, b: str) -> str:
    a_sents = re.split(r"(?<=[.;:!?])\s+", a)
    b_sents = re.split(r"(?<=[.;:!?])\s+", b)
    sm = difflib.SequenceMatcher(None, a_sents, b_sents)
    html = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for s in a_sents[i1:i2]: html.append(f"<span>{s} </span>")
        elif tag == 'delete':
            for s in a_sents[i1:i2]: html.append(f"<span style='background:#ffecec;'>− {s} </span>")
        elif tag == 'insert':
            for s in b_sents[j1:j2]: html.append(f"<span style='background:#eaffea;'>+ {s} </span>")
        elif tag == 'replace':
            for s in a_sents[i1:i2]: html.append(f"<span style='background:#ffecec;'>~ {s} </span>")
            for s in b_sents[j1:j2]: html.append(f"<span style='background:#eaffea;'>→ {s} </span>")
    return "<div style='line-height:1.6'>" + " ".join(html) + "</div>"

# ------------------------------- PDF Brief -------------------------------

def wrap_text(txt: str, width: int) -> List[str]:
    words, lines, cur = txt.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def make_brief_pdf(path_or_buf, title: str, summary: str,
                   qas: List[Tuple[str,str]], dates: List[str]) -> None:
    c = canvas.Canvas(path_or_buf, pagesize=A4)
    W, H = A4
    margin, y = 40, H - 60

    c.setFont("Helvetica-Bold", 16); c.drawString(margin, y, title); y -= 22
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Auto-generated brief — Not legal advice. Verify against original document."); y -= 18

    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Plain-language Summary:"); y -= 14
    c.setFont("Helvetica", 10)
    for line in wrap_text(summary, 100):
        c.drawString(margin, y, line); y -= 12
        if y < 80: c.showPage(); y = H - 60

    if dates:
        y -= 8
        c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Dates found:"); y -= 14
        c.setFont("Helvetica", 10)
        for d in dates[:20]:
            c.drawString(margin, y, f"• {d}"); y -= 12
            if y < 80: c.showPage(); y = H - 60

    if qas:
        y -= 8
        c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Q&A with citations:"); y -= 16
        c.setFont("Helvetica", 10)
        for q, a in qas[:4]:
            for line in wrap_text("Q: " + q, 100): c.drawString(margin, y, line); y -= 12
            for line in wrap_text("A: " + a, 100): c.drawString(margin, y, line); y -= 12
            y -= 6
            if y < 80: c.showPage(); y = H - 60

    c.showPage(); c.save()

# ------------------------------- Sidebar --------------------------------

with st.sidebar:
    st.markdown("### Settings")
    out_lang = st.selectbox("Answer language", ["English", "Hindi (हिंदी)"])
    grade = st.slider("Rewrite grade level", 5, 14, 8)
    topk = st.slider("Retriever top-k", 2, 10, 5)
    st.divider()
    if GEMINI_KEY:
        st.success("Model: **Gemini** (`google-genai`) — API key detected")
    else:
        st.warning("Model: Gemini **disabled** (no `GEMINI_API_KEY`) — using extractive fallbacks")
    st.markdown(
        "<span class='small-note'>Tip: Smaller PDFs index faster. For scanned PDFs, text extraction may be limited.</span>",
        unsafe_allow_html=True
    )
    st.divider()
    if st.button("↺ Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ------------------------------- Uploads & Index -------------------------

colA, colB = st.columns(2)
with colA:
    st.subheader("📄 Document A (current / main)")
    file_a = st.file_uploader("Upload PDF", type=["pdf"], key="A")
with colB:
    st.subheader("📄 Document B (optional — for 'What changed?')")
    file_b = st.file_uploader("Upload PDF", type=["pdf"], key="B")

if "corpusA" not in st.session_state:
    st.session_state.update({
        "corpusA": None, "X_A": None, "indexA": None,
        "fulltextA": "", "metrics": {}
    })

if file_a is not None and st.button("🔎 Build index for A"):
    t, _ = timer()
    pagesA = extract_text_pages(file_a.read())
    build_ms_1 = t()
    st.session_state.fulltextA = "\n\n".join([t for _, t in pagesA])
    corpusA = build_corpus(pagesA)
    build_ms_2 = t()
    X_A, idxA = embed_corpus(corpusA)
    build_ms_3 = t()
    st.session_state.corpusA, st.session_state.X_A, st.session_state.indexA = corpusA, X_A, idxA
    st.session_state.metrics = {
        "pages": len(pagesA),
        "chunks": len(corpusA),
        "parse_ms": round(build_ms_1, 1),
        "chunk_ms": round(build_ms_2 - build_ms_1, 1),
        "embed_ms": round(build_ms_3 - build_ms_2, 1)
    }
    st.success(f"Indexed {len(corpusA)} chunks from {len(pagesA)} pages.")
    m = st.session_state.metrics
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f"<div class='kpi-card'><b>Pages</b><br>{m['pages']}</div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='kpi-card'><b>Chunks</b><br>{m['chunks']}</div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='kpi-card'><b>Index time</b><br>{m['parse_ms']+m['chunk_ms']+m['embed_ms']} ms</div>", unsafe_allow_html=True)

# ------------------------------- Tabs -----------------------------------

tab_qna, tab_rewrite, tab_diff, tab_timeline, tab_brief = st.tabs(
    ["Q&A with citations", "ELI-style Rewrite", "What changed? (diff)", "Validity timeline", "Export 2-page brief"]
)

# --- Q&A ---
with tab_qna:
    st.markdown("#### Ask a question about Document A")
    q = st.text_input("Your question", placeholder="What fees do I need to pay? Deadline? Penalty if I miss it?")
    if st.button("Answer"):
        if not st.session_state.get("corpusA"):
            st.warning("Upload & build index for Document A first.")
        elif not q.strip():
            st.warning("Enter a question.")
        else:
            t, _ = timer()
            hits = search(st.session_state.corpusA, st.session_state.X_A, st.session_state.indexA, q, k=topk)
            search_ms = t()
            ans = grounded_answer(q, hits, out_lang)
            gen_ms = t() - search_ms
            # Confidence meter
            conf = similarity_confidence([s for _, s in hits])
            st.markdown("##### Answer")
            st.write(ans)
            st.markdown(
                f"<span class='confidence'>Confidence (retrieval): {int(conf*100)}%</span>  "
                f"<span class='small-note'>| retrieve {int(search_ms)} ms • generate {int(max(gen_ms,0))} ms</span>",
                unsafe_allow_html=True
            )
            with st.expander("Cited context"):
                for i, (chunk, score) in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] Page {chunk['page']}** — similarity: {float(score):.3f}")
                    st.write(chunk["text"])

# --- Rewrite ---
with tab_rewrite:
    st.markdown("#### Simplify a clause from Document A")
    if st.session_state.get("corpusA") is None:
        st.info("Upload & build index for Document A first.")
    else:
        kw = st.text_input("Find a clause by keyword (optional)")
        subset = st.session_state.corpusA
        if kw.strip():
            subset = [c for c in subset if kw.lower() in c["text"].lower()]
            st.caption(f"Matched {len(subset)} chunk(s).")
        if subset:
            picked = st.selectbox("Choose a clause", options=range(len(subset)),
                                  format_func=lambda i: f"p.{subset[i]['page']} — {subset[i]['text'][:120]}...")
            bilingual = st.checkbox("Also output Hindi + English")
            if st.button("Rewrite simply"):
                txt = subset[picked]["text"]
                out = eli_rewrite(txt, grade, out_lang, bilingual)
                st.markdown("##### Plain-language rewrite")
                if kw:
                    st.write(highlight(out, kw))
                else:
                    st.write(out)
        else:
            st.info("No matching clause found. Try a different keyword or build index first.")

# --- Diff ---
with tab_diff:
    st.markdown("#### Compare two versions (A vs B)")
    if not file_b:
        st.info("Upload Document B to enable diff.")
    else:
        if st.button("Run diff"):
            text_a = st.session_state.get("fulltextA","")
            text_b = "\n\n".join([t for _, t in extract_text_pages(file_b.read())])
            html = diff_text(text_a, text_b)
            st.markdown("<div class='diff-box'>", unsafe_allow_html=True)
            st.components.v1.html(html, height=600, scrolling=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- Timeline ---
with tab_timeline:
    st.markdown("#### Dates & validity hints (from Document A)")
    if not st.session_state.get("fulltextA"):
        st.info("Upload & build index for Document A first.")
    else:
        dates = find_dates(st.session_state.fulltextA)
        if dates:
            st.write("Found dates (sorted):")
            st.markdown("<ul class='compact'>" + "".join(f"<li>{d}</li>" for d in dates) + "</ul>", unsafe_allow_html=True)
            st.caption("These are extracted mentions; verify the *effective* date in the original.")
        else:
            st.write("No clear dates found.")

# --- Brief ---
with tab_brief:
    st.markdown("#### Export a 2-page brief PDF")
    title = st.text_input("Brief title", value="CivLex Plain-Law Brief")
    q1 = st.text_input("Optional Q1", value="What is my obligation and deadline?")
    q2 = st.text_input("Optional Q2", value="What fees/penalties are defined?")
    if st.button("Generate brief"):
        if not st.session_state.get("corpusA"):
            st.warning("Upload & build index for Document A first.")
        else:
            hits1 = search(st.session_state.corpusA, st.session_state.X_A, st.session_state.indexA, q1, k=topk) if q1 else []
            hits2 = search(st.session_state.corpusA, st.session_state.X_A, st.session_state.indexA, q2, k=topk) if q2 else []
            ans1 = grounded_answer(q1, hits1, out_lang) if q1 else ""
            ans2 = grounded_answer(q2, hits2, out_lang) if q2 else ""
            # concise summary from the most relevant chunks (or first 2k chars)
            top_join = " ".join([h[0]["text"] for h in (hits1+hits2)])[:2000]
            summary = eli_rewrite(top_join or st.session_state.fulltextA[:2000], grade=8, lang=out_lang, bilingual=False)
            dates = find_dates(st.session_state.fulltextA)
            buf = io.BytesIO()
            make_brief_pdf(buf, title, summary, [(q1, ans1),(q2, ans2)], dates)
            st.download_button("⬇️ Download brief.pdf", data=buf.getvalue(),
                               file_name="brief.pdf", mime="application/pdf")

# ------------------------------- Footer ----------------------------------

st.markdown(
    "<div class='small-note'>Built with Streamlit • PyMuPDF • Sentence-Transformers • FAISS/NumPy • google-genai • ReportLab</div>",
    unsafe_allow_html=True
)
