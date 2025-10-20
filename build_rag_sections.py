#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import time
import json
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit("openai package with new client required. pip install openai") from e

try:
    import tiktoken
except Exception as e:
    raise SystemExit("tiktoken required. pip install tiktoken") from e

try:
    import faiss
except Exception as e:
    raise SystemExit("faiss-cpu required. pip install faiss-cpu") from e

try:
    from bs4 import BeautifulSoup, Comment
except Exception as e:
    raise SystemExit("beautifulsoup4 required. pip install beautifulsoup4") from e

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

DEFAULT_INPUT_DIR = "blogs_html"
DEFAULT_INDEX_PATH = "rag_sections.faiss"
DEFAULT_META_PATH = "rag_sections_meta.jsonl"

EMBEDDING_MODEL = "text-embedding-3-large"  # change to -small to reduce cost
_EMBEDDING_DIM_MAP = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}
EMBEDDING_DIM = _EMBEDDING_DIM_MAP.get(EMBEDDING_MODEL)
if EMBEDDING_DIM is None:
    raise SystemExit(f"Unknown embedding model: {EMBEDDING_MODEL}")

ENCODING_NAME = "cl100k_base"
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100

EMBED_BATCH = 16
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_rag_sections")

def load_env_file(env_path: Path | None):
    if env_path and env_path.exists() and DOTENV_AVAILABLE:
        load_dotenv(str(env_path))
        logger.debug(f"Loaded .env from {env_path}")
    elif env_path and env_path.exists() and not DOTENV_AVAILABLE:
        # minimal parser
        with env_path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set it in environment or .env file.")
    return OpenAI(api_key=key)

_enc = tiktoken.get_encoding(ENCODING_NAME)

def tokens_of(text: str) -> List[int]:
    return _enc.encode(text)

def detokenize(tokens: List[int]) -> str:
    return _enc.decode(tokens)

def chunk_tokens_to_texts(tokens: List[int], target_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int,int,str]]:
    """
    Return list of (start_token, end_token, text) chunks for given tokens.
    """
    n = len(tokens)
    if n == 0:
        return []
    chunks = []
    i = 0
    while i < n:
        j = min(i + target_tokens, n)
        chunk_toks = tokens[i:j]
        chunk_text = detokenize(chunk_toks)
        chunks.append((i, j, chunk_text))
        if j >= n:
            break
        i = j - overlap
    return chunks

def html_file_to_sections(html_path: Path) -> List[Dict[str, Any]]:
    """
    Parse HTML and split into logical sections.
    Strategy:
      - Remove <script>, <style>, comments, and <img> tags.
      - Find all H2 elements; each H2 + following siblings until next H2 is a section.
      - If no H2: use H1 headings similarly.
      - If no H1/H2: take full body text and split into two token-balanced sections.
    Returns list of dicts: {"title": str, "html_fragment": str, "text": str}
    """
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # remove scripts/styles/comments/images
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for img in soup.find_all("img"):
        img.decompose()  # remove images completely from RAG content

    body = soup.body or soup
    # Collect H2 sections
    sections: List[Dict[str, Any]] = []
    h2s = body.find_all(["h2"])
    if h2s:
        for h2 in h2s:
            title = h2.get_text(separator=" ", strip=True)
            # collect siblings until next H2
            content_parts = []
            sibling = h2.next_sibling
            while sibling:
                # if sibling is a tag and is H2, stop
                if getattr(sibling, "name", None) == "h2":
                    break
                # include the sibling HTML/text
                content_parts.append(str(sibling))
                sibling = sibling.next_sibling
            fragment_html = f"<h2>{title}</h2>\n" + "\n".join(content_parts)
            # extract readable text
            frag_soup = BeautifulSoup(fragment_html, "html.parser")
            text = frag_soup.get_text(separator="\n", strip=True)
            sections.append({"title": title, "html_fragment": fragment_html, "text": text})
        return sections

    # fallback: H1 sections
    h1s = body.find_all(["h1"])
    if h1s:
        for h1 in h1s:
            title = h1.get_text(separator=" ", strip=True)
            content_parts = []
            sibling = h1.next_sibling
            while sibling:
                if getattr(sibling, "name", None) == "h1":
                    break
                content_parts.append(str(sibling))
                sibling = sibling.next_sibling
            fragment_html = f"<h1>{title}</h1>\n" + "\n".join(content_parts)
            frag_soup = BeautifulSoup(fragment_html, "html.parser")
            text = frag_soup.get_text(separator="\n", strip=True)
            sections.append({"title": title, "html_fragment": fragment_html, "text": text})
        return sections

    # no headings -> split body text into two (token balanced)
    full_text = body.get_text(separator="\n", strip=True)
    toks = tokens_of(full_text)
    if len(toks) == 0:
        return []
    mid = len(toks) // 2
    # ensure we split at nearest sentence boundary by decoding small windows
    # find split token index by moving right until we find a punctuation/linebreak
    split_idx = mid
    # attempt to move forward up to 200 tokens to find a linebreak char
    for offset in range(0, min(200, len(toks)-mid)):
        candidate = detokenize(toks[mid+offset:mid+offset+10])
        if "\n" in candidate or "。" in candidate or "." in candidate:
            split_idx = mid + offset
            break
    sec1_toks = toks[:split_idx]
    sec2_toks = toks[split_idx:]
    sec1_text = detokenize(sec1_toks)
    sec2_text = detokenize(sec2_toks)
    return [
        {"title": "بخش اول", "html_fragment": "", "text": sec1_text},
        {"title": "بخش دوم", "html_fragment": "", "text": sec2_text},
    ]

def create_embeddings(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = EMBED_BATCH) -> List[List[float]]:
    embeddings: List[List[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start+batch_size]
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # resp.data is list corresponding to inputs
                for item in resp.data:
                    embeddings.append(list(item.embedding))
                break
            except Exception as e:
                last_exc = e
                wait = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Embedding API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait:.1f}s.")
                time.sleep(wait)
                continue
        else:
            raise RuntimeError(f"Failed creating embeddings after {MAX_RETRIES} attempts: {last_exc}")
    if len(embeddings) != total:
        raise RuntimeError(f"Embedding count mismatch: expected {total}, got {len(embeddings)}")
    return embeddings

def build_faiss_index(vectors: np.ndarray, dim: int) -> faiss.Index:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def build_rag_from_html_folder(input_dir: Path, out_index_path: Path, out_meta_path: Path, env_file: Path | None = None):
    # load .env if provided
    if env_file:
        load_env_file(env_file)

    client = get_openai_client()

    # collect html files
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".html", ".htm")])
    if not files:
        raise RuntimeError(f"No HTML files found in {input_dir}")

    logger.info(f"Found {len(files)} HTML files. Parsing and creating sections...")
    all_chunks = []  # will hold metadata dicts
    chunk_texts = []  # texts to embed

    unique_id = 0
    for file_idx, path in enumerate(files):
        try:
            sections = html_file_to_sections(path)
            if not sections:
                logger.warning(f"No text extracted from {path}; skipping.")
                continue

            for sec_idx, sec in enumerate(sections):
                text = sec.get("text", "").strip()
                if not text:
                    continue
                # normalize whitespace
                text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip() != ""])
                # chunk this section token-wise
                toks = tokens_of(text)
                chunks = chunk_tokens_to_texts(toks, target_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
                # If section yields no chunks (very short), still create one chunk
                if not chunks:
                    chunks = [(0, len(toks), text)]
                for chunk_idx, (start_t, end_t, chunk_text) in enumerate(chunks):
                    meta = {
                        "id": unique_id,
                        "source_file": path.name,
                        "file_index": file_idx,
                        "section_index": sec_idx,
                        "section_title": sec.get("title", "") or "",
                        "chunk_index": chunk_idx,
                        "start_token": int(start_t),
                        "end_token": int(end_t),
                        "text": chunk_text,
                    }
                    all_chunks.append(meta)
                    chunk_texts.append(chunk_text)
                    unique_id += 1
        except Exception as e:
            logger.exception(f"Error processing file {path}: {e}")

    if not all_chunks:
        raise RuntimeError("No chunks produced from input files.")

    logger.info(f"Created {len(all_chunks)} chunks total. Creating embeddings (model={EMBEDDING_MODEL})...")
    embeddings = create_embeddings(client, chunk_texts, model=EMBEDDING_MODEL)
    vecs = np.array(embeddings, dtype="float32")
    if vecs.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vecs.shape[1]}")

    logger.info("Building FAISS index...")
    index = build_faiss_index(vecs, EMBEDDING_DIM)
    logger.info(f"Saving FAISS index to {out_index_path} ...")
    faiss.write_index(index, str(out_index_path))

    # write metadata as JSONL
    logger.info(f"Writing metadata JSONL to {out_meta_path} ...")
    with out_meta_path.open("w", encoding="utf-8") as f:
        for meta in all_chunks:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    logger.info("RAG build complete.")
    logger.info(f"Index: {out_index_path}  Metadata: {out_meta_path}  Chunks: {len(all_chunks)}")

def parse_args():
    p = argparse.ArgumentParser(description="Build section-aware RAG from HTML blog files.")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT_DIR, help="Input folder containing HTML files.")
    p.add_argument("--out-index", default=DEFAULT_INDEX_PATH, help="Output FAISS index path.")
    p.add_argument("--out-meta", default=DEFAULT_META_PATH, help="Output metadata JSONL path.")
    p.add_argument("--env", default=".env", help="Optional .env path to load OPENAI_API_KEY (or set env var).")
    return p.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)
    out_index = Path(args.out_index)
    out_meta = Path(args.out_meta)
    env_path = Path(args.env) if args.env else None

    try:
        if env_path and env_path.exists():
            load_env_file(env_path)
        build_rag_from_html_folder(input_dir=input_dir, out_index_path=out_index, out_meta_path=out_meta, env_file=env_path)
    except Exception as e:
        logger.exception("Failed to build RAG: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
