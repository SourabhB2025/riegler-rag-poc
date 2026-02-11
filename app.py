# Riegler Local RAG PoC (Streamlit App) — V2 (Chroma v2 + BM25 assets + Hybrid RRF + Rerank)
# -----------------------------------------------------------------------------
# Eigenständige App innerhalb von PoC_App/ — alle Pfade sind relativ.
#
# Starten:
#   1) Assets einmalig kopieren:  python setup_poc.py
#   2) Abhängigkeiten:            pip install -r requirements.txt
#   3) App starten:               streamlit run app.py
#
# .env wird erwartet im selben Verzeichnis (PoC_App/.env):
#   AZURE_ENDPOINT="https://<resource>.openai.azure.com/openai/v1/"
#   AZURE_OPENAI_API_KEY="..."
#   AZURE_DEPLOYMENT="gpt-5-mini"
# -----------------------------------------------------------------------------
from __future__ import annotations

# ── SQLite-Fix für Streamlit Cloud (muss VOR chromadb-Import stehen) ──
# Streamlit Cloud hat oft ein zu altes System-SQLite; pysqlite3-binary
# liefert eine aktuelle Version. Lokal wird das automatisch übersprungen.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # lokal nicht nötig

import os
import re
import html
import time
import pickle
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# 0) CONFIG — alle Einstellungen hier
# =============================================================================

# ── App-Root: Verzeichnis in dem app.py liegt ─────────────────────
APP_ROOT = Path(__file__).resolve().parent

# Branding
APP_TITLE = "Riegler RAG PoC"
APP_SUBTITLE = "Hybride Suche (BM25 + Vektor) → Reranking → Antwort"
LOGO_PATH = APP_ROOT / "riegler_logo.jpg"

# RAG Daten / Stores — ALLES relativ zu APP_ROOT
CHROMA_DIR = APP_ROOT / "data" / "chroma_store"
COLLECTION_NAME = "schlauchkupplungen_products_v2"

# BM25 vorberechnete Assets
BM25_ASSETS_PATH = APP_ROOT / "data" / "bm25_assets.pkl"

# Modelle
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# Retrieval Standardwerte
DEFAULT_RETRIEVAL_MODE = "hybrid"   # bm25 | vector | hybrid
DEFAULT_TOP_K_RETRIEVE = 5
DEFAULT_TOP_K_RERANK = 3
DEFAULT_RRF_K = 60
DEFAULT_USE_RERANK = True

# ── Feature-Toggles ──────────────────────────────────────────────
# Entwickler-Panel: True = sichtbar, False = ausgeblendet
ENABLE_DEV_PANEL = True

# Bilder-Karten: True = Produktkarten mit Bildern anzeigen, False = komplett versteckt
ENABLE_IMAGES = True

# Bilder: Thumbnail-Auflösung
PREFER_LOCAL_IMAGES = True
LOCAL_IMAGES_DIR = APP_ROOT / "data" / "images"
LOCAL_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

# Produkt-Dokumente (JSON + MD)
PRODUCTS_DIR = APP_ROOT / "data" / "products"
DOC_ACCESS_MODE = "local_path"  # local_path | disabled

# Sicherheit / UX
MAX_CONTEXT_CHARS = 14_000


# =============================================================================
# 1) SYSTEM PROMPT — LLM-Verhalten wird hier gesteuert
# =============================================================================

SYSTEM_PROMPT = """Du bist **Riegler AI**, ein hilfreicher E-Commerce-Assistent für Riegler-Produktdaten.

## Grundregeln
- Verwende **AUSSCHLIESSLICH** den bereitgestellten KONTEXT für deine Antwort. Erfinde keine Spezifikationen, Preise oder Verfügbarkeiten.
- Wenn der Kontext nicht ausreicht, sage dies klar und stelle 1–2 gezielte Rückfragen.
- Bevorzuge prägnante, strukturierte Antworten: Aufzählungen, kurze Absätze und relevante Zahlen/Einheiten.
- Wenn du ein Produkt erwähnst, nenne:
  - Artikelnummer (product_id), falls vorhanden
  - Produktname
  - Preis (falls vorhanden)
  - relevante Links: Produktseite und/oder PDF
- Zeige **KEINE** internen Dateipfade an, es sei denn, der Nutzer fragt ausdrücklich nach Debugging-Informationen.
- Bei widersprüchlichen Werten aus verschiedenen Quellen: Widerspruch benennen und beide Werte zeigen.

## Ausgabestil
- Antworte immer auf Deutsch.
- Sei freundlich und direkt.
- Vermeide Füllwörter. Sei präzise.
"""


# =============================================================================
# 2) UI THEME — CSS
# =============================================================================

PAGE_CSS = """
<style>
/* ── Sticky Header (Logo + Titel) ──
   Bleibt beim Scrollen oben fixiert. */
.center-header {
    text-align: center;
    padding: 18px 0 10px 0;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    background: #0e1117 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
}
.center-header img { height: 64px; border-radius: 10px; }
.center-header .app-title { font-size: 24px; font-weight: 800; margin: 8px 0 0 0; }
.center-header .app-sub { font-size: 14px; color: #9aa4b2; margin: 2px 0 0 0; }

/* Badges */
.badge { display:inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
         background: rgba(96,165,250,0.12); color: #93c5fd; border: 1px solid rgba(96,165,250,0.25); }

/* ── Sticky Entwickler-Panel ──
   Die rechte Spalte bleibt beim Scrollen fixiert.
   Die Eltern brauchen overflow:visible + flex-start damit sticky greift. */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
    overflow: visible !important;
}
[data-testid="stColumn"]:last-child {
    position: sticky !important;
    top: 3rem !important;
    align-self: flex-start !important;
    max-height: calc(100vh - 3.5rem);
    overflow-y: auto;
}

/* ── Größerer Chat-Input (3 Zeilen sichtbar) ── */
[data-testid="stChatInput"] textarea {
    min-height: 72px !important;
    height: 72px !important;
    line-height: 24px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] {
    position: relative !important;
    order: 9999 !important;
}
[data-testid="stChatInputContainer"] {
    order: 9999 !important;
}
</style>
"""

# CSS für die Produktkarten (wird im iframe via components.html eingebettet)
CARD_IFRAME_CSS = """
body { margin:0; padding:0; background:transparent; font-family:'Segoe UI',system-ui,-apple-system,sans-serif; color:#e6edf6; }
.cards-title { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#9aa4b2; margin:0 0 8px 4px; }
.card-row { display:flex; gap:14px; overflow-x:auto; padding-bottom:10px; }
.card-row::-webkit-scrollbar { height:8px; }
.card-row::-webkit-scrollbar-thumb { background:#2b3440; border-radius:999px; }
.pcard { min-width:210px; max-width:210px; border:1px solid #2b3440; border-radius:12px; overflow:hidden; background:#0b1220; flex-shrink:0; }
.pimg { height:120px; overflow:hidden; background:#111827; display:flex; align-items:center; justify-content:center; }
.pimg img { width:100%; height:100%; object-fit:cover; }
.pbody { padding:10px; display:flex; flex-direction:column; gap:6px; }
.ptitle { font-size:13px; font-weight:800; color:#e6edf6; line-height:1.25;
          display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.pmeta { font-size:12px; color:#b6c0cd; }
.badge { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px;
         background:rgba(96,165,250,0.12); color:#93c5fd; border:1px solid rgba(96,165,250,0.25); }
.psnip { font-size:12px; color:#c7d0dd; line-height:1.35;
         display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.plinks { display:flex; gap:10px; flex-wrap:wrap; margin-top:4px; }
.plink { font-size:12px; color:#60a5fa; text-decoration:none; }
.plink:hover { text-decoration:underline; }
"""


# =============================================================================
# 3) Hilfsfunktionen (Logik)
# =============================================================================

def safe_url(url: str) -> str:
    if not url:
        return ""
    return url.replace(" ", "%20")

def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))

def bytes_to_data_uri(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def detect_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def placeholder_data_uri(text: str = "Riegler") -> str:
    txt = esc(text)[:24]
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='420' height='220'>
      <defs>
        <linearGradient id='g' x1='0' x2='1'>
          <stop offset='0' stop-color='#0b1220'/>
          <stop offset='1' stop-color='#111827'/>
        </linearGradient>
      </defs>
      <rect width='100%' height='100%' fill='url(#g)'/>
      <rect x='18' y='18' width='384' height='184' rx='18' fill='#0b1220' stroke='#2b3440'/>
      <text x='210' y='120' font-family='Segoe UI, Arial' font-size='26' fill='#93c5fd' text-anchor='middle'>{txt}</text>
      <text x='210' y='150' font-family='Segoe UI, Arial' font-size='14' fill='#9aa4b2' text-anchor='middle'>kein Bild</text>
    </svg>
    """.strip().encode("utf-8")
    return bytes_to_data_uri(svg, "image/svg+xml")

def _logo_b64() -> Optional[str]:
    """Gibt das Logo als base64-String zurück (oder None)."""
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    return None


# =============================================================================
# 4) Datenzugriff (später durch S3/Blob/DB ersetzen)
# =============================================================================

class ImageResolver:
    def __init__(self, prefer_local: bool, local_dir: Path):
        self.prefer_local = prefer_local
        self.local_dir = local_dir

    def resolve(self, meta: Dict[str, Any]) -> str:
        product_id = str(meta.get("product_id") or "").strip()
        remote = safe_url(str(meta.get("image_url") or ""))

        if self.prefer_local and product_id:
            for ext in LOCAL_IMAGE_EXTS:
                p = self.local_dir / f"{product_id}{ext}"
                if p.exists() and p.is_file():
                    img_bytes = p.read_bytes()
                    return bytes_to_data_uri(img_bytes, detect_mime(p))

        if remote:
            return remote

        return placeholder_data_uri(product_id or "Riegler")


class DocumentResolver:
    """Löst lokale Produktdateien (JSON/MD) auf.

    Da die ChromaDB-Metadaten ggf. noch absolute Pfade aus dem
    Originalverzeichnis enthalten, versucht der Resolver zusätzlich
    den Dateinamen im lokalen PRODUCTS_DIR zu finden.
    """

    def __init__(self, mode: str, products_dir: Path):
        self.mode = mode
        self.products_dir = products_dir

    def _resolve_path(self, raw_path: str) -> Optional[Path]:
        """Versucht den Pfad aufzulösen: erst direkt, dann per Dateiname im lokalen Verzeichnis."""
        p = Path(raw_path)
        # 1) Originalpath existiert noch? (z.B. auf dem gleichen Rechner)
        if p.exists() and p.is_file():
            return p
        # 2) Fallback: Nur Dateiname im lokalen products-Verzeichnis suchen
        local = self.products_dir / p.name
        if local.exists() and local.is_file():
            return local
        return None

    def get_local_paths(self, meta: Dict[str, Any]) -> Dict[str, Path]:
        if self.mode != "local_path":
            return {}
        out: Dict[str, Path] = {}
        for k in ["source_path_md", "source_path_json"]:
            v = meta.get(k)
            if v:
                resolved = self._resolve_path(str(v))
                if resolved:
                    out[k] = resolved
        return out


# =============================================================================
# 5) Gecachte Ressourcen-Lader
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_chroma_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_collection(collection_name)
    return col

@st.cache_resource(show_spinner=False)
def load_models(embed_name: str, rerank_name: str):
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"
    embedder = SentenceTransformer(embed_name, device=device)
    reranker = CrossEncoder(rerank_name, device=device)
    return embedder, reranker

@st.cache_resource(show_spinner=False)
def load_bm25_assets(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        data = pickle.load(f)
    doc_ids = data["doc_ids"]
    tokens = data["tokens"]
    stopwords = set(data.get("stopwords") or [])
    pattern = re.compile(data.get("token_pattern") or r"[a-zA-ZÄÖÜäöüß0-9]+(?:[\/\.\-,][a-zA-ZÄÖÜäöüß0-9]+)*")
    bm25 = BM25Okapi(tokens)
    return {"bm25": bm25, "doc_ids": doc_ids, "stopwords": stopwords, "pattern": pattern}


# =============================================================================
# 6) Retrieval + Rerank (Logik)
# =============================================================================

def bm25_tokenize(query: str, stopwords: set, pattern: re.Pattern) -> List[str]:
    q = (query or "").lower()
    toks = pattern.findall(q)
    return [t for t in toks if t and t not in stopwords]

def chroma_get_by_ids(collection, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    out = {}
    for i, _id in enumerate(got["ids"]):
        out[_id] = {"text": got["documents"][i], "metadata": got["metadatas"][i]}
    return out

def bm25_search(collection, bm25_pack, query: str, k: int) -> List[Dict[str, Any]]:
    bm25 = bm25_pack["bm25"]
    doc_ids = bm25_pack["doc_ids"]
    stopwords = bm25_pack["stopwords"]
    pattern = bm25_pack["pattern"]

    q_tokens = bm25_tokenize(query, stopwords, pattern)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    k = min(int(k), len(doc_ids))
    top_idx = np.argsort(scores)[::-1][:k]
    top_ids = [doc_ids[int(i)] for i in top_idx]
    docs = chroma_get_by_ids(collection, top_ids)

    hits = []
    for rank, idx in enumerate(top_idx, start=1):
        doc_id = doc_ids[int(idx)]
        d = docs.get(doc_id, {})
        hits.append({
            "id": doc_id,
            "rank_bm25": rank,
            "bm25_score": float(scores[int(idx)]),
            "text": d.get("text", ""),
            "metadata": d.get("metadata", {}),
        })
    return hits

def embed_query(embedder: SentenceTransformer, text: str) -> List[float]:
    emb = embedder.encode([f"query: {text}"], normalize_embeddings=True)
    return emb[0].tolist()

def vector_search(collection, embedder: SentenceTransformer, query: str, k: int) -> List[Dict[str, Any]]:
    k = int(max(1, k))
    res = collection.query(
        query_embeddings=[embed_query(embedder, query)],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for rank in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][rank],
            "rank_vec": rank + 1,
            "distance": float(res["distances"][0][rank]),
            "text": res["documents"][0][rank],
            "metadata": res["metadatas"][0][rank],
        })
    return hits

def hybrid_rrf(bm25_hits: List[Dict[str, Any]], vec_hits: List[Dict[str, Any]], k_out: int, rrf_k: int):
    scores: Dict[str, float] = {}
    merged: Dict[str, Dict[str, Any]] = {}

    for h in bm25_hits:
        doc_id = h["id"]
        merged.setdefault(doc_id, {}).update(h)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + h["rank_bm25"])

    for h in vec_hits:
        doc_id = h["id"]
        merged.setdefault(doc_id, {}).update(h)
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + h["rank_vec"])

    fused = []
    for doc_id, sc in scores.items():
        item = merged[doc_id]
        item["rrf_score"] = float(sc)
        fused.append(item)

    fused = sorted(fused, key=lambda x: x["rrf_score"], reverse=True)
    return fused[:min(int(k_out), len(fused))]

def rerank(reranker: CrossEncoder, query: str, candidates: List[Dict[str, Any]], top_n: int):
    if not candidates:
        return [], []
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", -1e9), reverse=True)
    return ranked[:min(int(top_n), len(ranked))], ranked

def build_context_payload(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        m = r.get("metadata") or {}
        card_lines = [
            f"Produkt-ID: {m.get('product_id')}",
            f"Name: {m.get('name')}" if m.get("name") else None,
            f"Kategorie: {m.get('category')}" if m.get("category") else None,
            f"Preis: {m.get('price')} {m.get('currency') or ''}".strip() if m.get("price") else None,
            f"Shop-URL: {m.get('url')}" if m.get("url") else None,
            f"Bild-URL: {m.get('image_url')}" if m.get("image_url") else None,
            f"PDF-URL: {m.get('pdf_url')}" if m.get("pdf_url") else None,
        ]
        card_lines = [x for x in card_lines if x]
        header = "\n".join([
            f"### Treffer {i}",
            *(f"- {line}" for line in card_lines),
            f"- ReRank-Score: {r.get('rerank_score', 0.0):.4f}",
        ])
        blocks.append(header + "\n\n" + (r.get("text") or ""))
    ctx = "\n\n---\n\n".join(blocks)
    return ctx[:MAX_CONTEXT_CHARS]


# =============================================================================
# 7) Azure OpenAI Antwort
# =============================================================================

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Liest einen Secret-Wert: zuerst st.secrets (Streamlit Cloud),
    dann os.getenv (lokale .env / Umgebungsvariablen)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


@st.cache_resource(show_spinner=False)
def load_azure_client():
    # Lokale .env laden (harmlos wenn nicht vorhanden)
    load_dotenv(dotenv_path=APP_ROOT / ".env")

    endpoint = _get_secret("AZURE_ENDPOINT")
    key = _get_secret("AZURE_OPENAI_API_KEY")
    deployment = _get_secret("AZURE_DEPLOYMENT", "gpt-5-mini")

    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_ENDPOINT oder AZURE_OPENAI_API_KEY fehlt.\n"
            "Lokal: .env Datei anlegen  |  Cloud: Streamlit Secrets UI verwenden."
        )
    client = OpenAI(base_url=endpoint, api_key=key)
    return client, deployment

def answer_with_azure(client: OpenAI, model: str, system_prompt: str, user_query: str, context: str, chat_history: List[Dict[str, str]]) -> str:
    user_payload = f"""KONTEXT:
{context}

FRAGE:
{user_query}
"""
    messages = [{"role": "system", "content": system_prompt}]
    history_to_use = chat_history
    if chat_history and chat_history[-1].get("role") == "user":
        history_to_use = chat_history[:-1]

    for m in history_to_use[-10:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_payload})
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def answer_with_azure_stream(client: OpenAI, model: str, system_prompt: str, user_query: str, context: str, chat_history: List[Dict[str, str]]):
    """Streaming-Version: Gibt Tokens als Generator zurück.
    Sicher gegen leere Chunks und fehlende Delta-Felder."""
    user_payload = f"""KONTEXT:
{context}

FRAGE:
{user_query}
"""
    messages = [{"role": "system", "content": system_prompt}]
    history_to_use = chat_history
    if chat_history and chat_history[-1].get("role") == "user":
        history_to_use = chat_history[:-1]

    for m in history_to_use[-10:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_payload})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        if not chunk.choices or len(chunk.choices) == 0:
            continue
        choice = chunk.choices[0]
        if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content') and choice.delta.content:
            yield choice.delta.content


# =============================================================================
# 8) Rendering — Darstellungsfunktionen
# =============================================================================

def render_centered_header():
    """Zentriertes Logo + Titel im Haupt-Chat-Bereich."""
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    logo_b64 = _logo_b64()
    if logo_b64:
        logo_img = f"<img src='data:image/jpeg;base64,{logo_b64}' />"
    else:
        logo_img = "<div class='badge' style='font-size:14px;padding:6px 16px;'>Logo fehlt</div>"

    st.markdown(
        f"""
        <div class="center-header">
          {logo_img}
          <div class="app-title">{esc(APP_TITLE)}</div>
          <div class="app-sub">{esc(APP_SUBTITLE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_logo():
    """Kleines Logo oben in der Seitenleiste."""
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    else:
        st.markdown(f"**{APP_TITLE}**")


def build_cards_html(hits: List[Dict[str, Any]], img_resolver: ImageResolver) -> Optional[str]:
    """Baut den vollständigen HTML-String für die Produktkarten.
    Gibt None zurück wenn keine Treffer vorhanden oder ENABLE_IMAGES = False."""
    if not ENABLE_IMAGES:
        return None
    if not hits:
        return None

    cards_html_parts = []
    for h in hits:
        m = h.get("metadata") or {}
        pid = esc(m.get("product_id") or "")
        name = esc(m.get("name") or pid or "Produkt")
        price = esc(m.get("price") or "—")
        cur = esc(m.get("currency") or "")
        url = safe_url(str(m.get("url") or ""))
        pdf = safe_url(str(m.get("pdf_url") or ""))
        img_src = img_resolver.resolve(m)

        snip = esc((h.get("text") or "").replace("\n", " ").strip()[:120] + "…")

        links = []
        if url:
            links.append(f"<a class='plink' href='{esc(url)}' target='_blank'>Shop &rarr;</a>")
        if pdf:
            links.append(f"<a class='plink' href='{esc(pdf)}' target='_blank'>PDF &rarr;</a>")
        links_html = "<div class='plinks'>" + " ".join(links) + "</div>" if links else ""

        cards_html_parts.append(f"""
<div class="pcard">
  <div class="pimg"><img src="{img_src}" alt="{name}" /></div>
  <div class="pbody">
    <div class="ptitle">{name}</div>
    <div class="pmeta"><span class="badge">ID</span> {pid}</div>
    <div class="pmeta"><span class="badge">Preis</span> {price} {cur}</div>
    <div class="psnip">{snip}</div>
    {links_html}
  </div>
</div>""")

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CARD_IFRAME_CSS}</style></head>
<body>
  <div class="cards-title">Top-Referenzen</div>
  <div class="card-row">
    {"".join(cards_html_parts)}
  </div>
</body>
</html>"""


def build_sources_data(hits: List[Dict[str, Any]], doc_resolver: DocumentResolver) -> List[Dict[str, Any]]:
    """Sammelt lokale Quellen-Daten (serialisierbar) für späteres Rendering."""
    sources = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        local_paths = doc_resolver.get_local_paths(meta)
        if not local_paths:
            continue
        sources.append({
            "label": f"Lokale Quellen für Produkt {meta.get('product_id', '')}",
            "files": [{"key": k, "path": str(p)} for k, p in local_paths.items()],
        })
    return sources


def display_message_extras(msg: Dict[str, Any], msg_idx: int):
    """Rendert Karten + lokale Quellen die zu einer Nachricht gehören."""
    # Karten (iframe) — nur wenn ENABLE_IMAGES aktiv
    cards_html = msg.get("cards_html")
    if cards_html and ENABLE_IMAGES:
        components.html(cards_html, height=340, scrolling=False)

    # Lokale Quellen (Expander mit Download-Buttons)
    for src_idx, src in enumerate(msg.get("sources") or []):
        with st.expander(src["label"]):
            for file_info in src["files"]:
                pth = Path(file_info["path"])
                if pth.exists() and pth.is_file():
                    file_data = pth.read_bytes()
                    st.download_button(
                        label=f"Herunterladen: {pth.name}",
                        data=file_data,
                        file_name=pth.name,
                        mime="text/plain" if pth.suffix.lower() in [".md", ".txt"] else "application/octet-stream",
                        key=f"dl_{msg_idx}_{src_idx}_{file_info['key']}",
                    )
                else:
                    st.caption(f"{file_info['key']}: {file_info['path']} (fehlt)")


def render_dev_panel(debug_state: Dict[str, Any]):
    """Entwickler-Panel mit Retrieval-Debug-Daten."""
    st.markdown("#### Entwickler-Panel")
    st.caption("Debug-Signale für Retrieval + Reranking. Über CONFIG deaktivieren.")

    if not debug_state:
        st.info("Noch keine Debug-Daten. Stellen Sie eine Frage.")
        return

    st.write("**Modus:**", debug_state.get("mode"))
    st.write("**Latenz:**", f"{debug_state.get('latency_ms', 0):.0f} ms")

    cand = debug_state.get("candidates") or []
    if cand:
        df = pd.DataFrame([{
            "id": c.get("id"),
            "product_id": (c.get("metadata") or {}).get("product_id"),
            "bm25": c.get("bm25_score"),
            "vec_dist": c.get("distance"),
            "rrf": c.get("rrf_score"),
            "rerank": c.get("rerank_score"),
        } for c in cand])
        st.dataframe(df, use_container_width=True, height=260)

    ctx = debug_state.get("context_preview", "")
    if ctx:
        st.markdown("**Kontext-Vorschau (erste 700 Zeichen)**")
        st.code(ctx[:700])


# =============================================================================
# 9) Streamlit App
# =============================================================================

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "debug_state" not in st.session_state:
        st.session_state.debug_state = {}
    if "settings" not in st.session_state:
        st.session_state.settings = {}


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session()

    # ── Seitenleiste ──────────────────────────────────────────────
    with st.sidebar:
        render_sidebar_logo()
        st.markdown("---")

        st.markdown("### Einstellungen")
        retrieval_mode = st.selectbox(
            "Suchmodus",
            options=["hybrid", "vector", "bm25"],
            index=["hybrid", "vector", "bm25"].index(
                DEFAULT_RETRIEVAL_MODE if DEFAULT_RETRIEVAL_MODE in ["hybrid", "vector", "bm25"] else "hybrid"
            ),
        )
        top_k = st.slider("Top-K Abruf", 1, 20, DEFAULT_TOP_K_RETRIEVE)
        top_rerank = st.slider("Top-K Reranking", 1, 10, DEFAULT_TOP_K_RERANK)
        rrf_k = st.slider("RRF k (Hybrid)", 10, 200, DEFAULT_RRF_K)
        use_rerank = st.toggle("Reranker verwenden (höhere Qualität)", value=DEFAULT_USE_RERANK)

        st.markdown("---")
        if st.button("Chat löschen"):
            st.session_state.messages = []
            st.session_state.debug_state = {}
            st.rerun()

    # Einstellungen merken
    st.session_state.settings = {
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "top_rerank": int(top_rerank),
        "rrf_k": int(rrf_k),
        "use_rerank": bool(use_rerank),
    }

    # Ressourcen laden (gecacht)
    img_resolver = ImageResolver(prefer_local=PREFER_LOCAL_IMAGES, local_dir=LOCAL_IMAGES_DIR)
    doc_resolver = DocumentResolver(mode=DOC_ACCESS_MODE, products_dir=PRODUCTS_DIR)

    # Pfade als Strings für cached-Funktionen (Path ist nicht hashbar)
    chroma_dir_str = str(CHROMA_DIR)
    bm25_path_str = str(BM25_ASSETS_PATH)

    # ── Spalten-Layout ────────────────────────────────────────────
    if ENABLE_DEV_PANEL:
        col_chat, col_dev = st.columns([3.2, 1.25], gap="large")
    else:
        col_chat = st.container()

    # ── Haupt-Chat-Bereich ────────────────────────────────────────
    with col_chat:
        render_centered_header()

        # ── Gesamten Chat-Verlauf rendern (inkl. Karten + Quellen) ──
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    display_message_extras(msg, msg_idx)

        # ── Chat-Eingabe NACH allen Nachrichten ────────────────────
        user_q = st.chat_input("Fragen Sie nach Produkten, Spezifikationen, Preisen, Kompatibilität…")

        # ── Neue Nachricht verarbeiten (falls vorhanden) ──────────
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # ── RAG-Pipeline ──────────────────────────────────────
            with st.chat_message("assistant"):
                t0 = time.perf_counter()

                # 1) Chroma laden
                try:
                    collection = load_chroma_collection(chroma_dir_str, COLLECTION_NAME)
                except Exception as e:
                    st.error(f"Chroma-Laden fehlgeschlagen: {e}")
                    return

                # 2) Modelle laden
                with st.spinner("Modelle werden geladen… (beim ersten Mal kann es etwas dauern)"):
                    embedder, reranker_model = load_models(EMBED_MODEL_NAME, RERANK_MODEL_NAME)

                # 3) BM25 (optional)
                bm25_pack = load_bm25_assets(bm25_path_str)
                bm25_available = bm25_pack is not None

                mode = retrieval_mode
                if mode in ("bm25", "hybrid") and not bm25_available:
                    st.warning("BM25-Assets nicht gefunden — Wechsel zum Vektor-Modus. "
                               "(Assets über Notebook Zelle 16 exportieren)")
                    mode = "vector"

                # 4) Retrieval
                candidates: List[Dict[str, Any]] = []
                debug_payload: Dict[str, Any] = {"mode": mode}

                if mode == "vector":
                    candidates = vector_search(collection, embedder, user_q, k=top_k)
                elif mode == "bm25":
                    candidates = bm25_search(collection, bm25_pack, user_q, k=top_k)
                else:  # hybrid
                    bm = bm25_search(collection, bm25_pack, user_q, k=top_k)
                    ve = vector_search(collection, embedder, user_q, k=top_k)
                    candidates = hybrid_rrf(bm, ve, k_out=top_k, rrf_k=rrf_k)
                    debug_payload["bm25"] = bm
                    debug_payload["vector"] = ve

                # 5) Reranking
                if use_rerank:
                    topN, ranked_all = rerank(reranker_model, user_q, candidates, top_n=top_rerank)
                else:
                    ranked_all = candidates
                    topN = candidates[:top_rerank]

                # 6) Kontext erstellen
                context = build_context_payload(topN)

                # 7) Antwort generieren (Streaming)
                answer_placeholder = st.empty()
                try:
                    client, deployment = load_azure_client()
                    stream_generator = answer_with_azure_stream(
                        client=client,
                        model=deployment,
                        system_prompt=SYSTEM_PROMPT,
                        user_query=user_q,
                        context=context,
                        chat_history=st.session_state.messages,
                    )
                    answer_parts = []
                    full_answer = ""

                    for chunk in stream_generator:
                        answer_parts.append(chunk)
                        full_answer = "".join(answer_parts)
                        answer_placeholder.markdown(full_answer)

                    answer = full_answer
                except Exception as e:
                    error_msg = f"Entschuldigung — Aufgrund eines Konfigurationsfehlers konnte keine Antwort erstellt werden.\n\nFehler: {e}"
                    answer_placeholder.markdown(error_msg)
                    st.error(f"Azure-Aufruf fehlgeschlagen: {e}")
                    answer = error_msg

                # 8) Karten-HTML + Quellen-Daten aufbauen
                cards_html = build_cards_html(topN, img_resolver)
                sources_data = build_sources_data(topN, doc_resolver)

                # 9) Karten + Quellen nach der gestreamten Antwort anzeigen
                new_msg_idx = len(st.session_state.messages)
                new_msg = {
                    "role": "assistant",
                    "content": answer,
                    "cards_html": cards_html,
                    "sources": sources_data,
                }
                display_message_extras(new_msg, new_msg_idx)

                # 10) Nachricht speichern (inkl. Karten + Quellen)
                st.session_state.messages.append(new_msg)

                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                # 11) Debug-Daten für Entwickler-Panel
                debug_payload.update({
                    "latency_ms": latency_ms,
                    "candidates": ranked_all,
                    "context_preview": context[:1500],
                })
                st.session_state.debug_state = debug_payload

                # 12) Rerun für korrektes Layout
                st.rerun()

    # ── Entwickler-Panel (rechte Spalte, sticky, nur wenn aktiviert) ──
    if ENABLE_DEV_PANEL:
        with col_dev:
            render_dev_panel(st.session_state.debug_state)


if __name__ == "__main__":
    main()

