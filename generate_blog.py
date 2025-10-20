#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import json
import time
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# third-party
try:
    from openai import OpenAI
except Exception:
    raise SystemExit("Install openai (new v1+ client): pip install openai")

try:
    import tiktoken
except Exception:
    raise SystemExit("Install tiktoken: pip install tiktoken")

try:
    import faiss
except Exception:
    raise SystemExit("Install faiss-cpu: pip install faiss-cpu")

try:
    from dotenv import load_dotenv
    DOTENV = True
except Exception:
    DOTENV = False

import numpy as np
from tqdm import tqdm

LOG = logging.getLogger("generate_blog")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_INDEX_PATH = "rag_sections.faiss"
DEFAULT_META_PATH = "rag_sections_meta.jsonl"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM_MAP = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}
EMBEDDING_DIM = EMBEDDING_DIM_MAP.get(EMBEDDING_MODEL)
ENCODING_NAME = "cl100k_base"

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.30
DEFAULT_MAX_TOKENS = 1500  # Increased for better content
RETRIEVE_TOP_K = 12  # Increased for better context
RETRIEVE_MULT = 8  # multiply to get more candidates
SUMMARIZE_MAX_TOKENS = 500  # Increased for better summaries
API_RETRY = 5  # Increased retry attempts
API_BACKOFF_BASE = 1.0
MIN_WORD_COUNT = 1500  # Increased minimum word count
ENHANCED_CONTENT_GENERATION = True  # Enable enhanced content strategies
CONTENT_QUALITY_THRESHOLD = 0.8  # Quality threshold for content validation
DIVERSITY_WEIGHT = 0.3  # Weight for content diversity
RELEVANCE_WEIGHT = 0.7  # Weight for content relevance

RULES_BLOCK = r"""
🔻 قوانین تولید محتوا (الزامی — حتما رعایت شود):

📌 قوانین عنوان H1:
- حتما شامل کلمه کلیدی اصلی باشد (از مشتقات آن نباشد) و دقیقا در ابتدای عنوان آورده شود
- در ادامه یک جمله ترغیب کننده شامل اعداد، ترین، صفر تا صد، گام به گام، نمونه کار و ... آورده شود
- مثال: "جراحی بینی چیست؟ + صفر تا صد عمل بینی به همراه نمونه کار"
- طول عنوان بیشتر از 40 حرف نباشد

📌 قوانین محتوای اصلی:
- ابتدای متن اصلی یک پاراگراف 3 الی 4 خطی باشد که دقیقا با خود کلمه کلیدی شروع شود
- تمام عنوان ها با تگ H2 نوشته شوند مگر در شرایطی که به لحاظ معنایی یک تیتر می تواند زیر مجموعه تیتر دیگری باشد
- برای مثال: (تیتر h2: درد عمل بینی) اگر تیتر بعد "بررسی روش های کاهش درد عمل بینی" باشد می توان از H3 استفاده کرد
- اگر ارتباط معنایی و حالت زیر مجموعه نباشد، تمام تیتر ها H2 باشند

📌 قوانین پاراگراف‌ها:
- نباید پاراگراف بیشتر از 3 الی 4 خط داشته باشد
- باید به پاراگراف بعد رفت چون طولانی شدن باعث بی توجهی کاربر به متن می شود
- هر پاراگراف باید یک ایده اصلی داشته باشد

📌 قوانین جدول:
- حتما در مقاله از یک جدول متناسب با تم رنگی وب سایت استفاده شود
- فونت مناسب عموما در سایت های فارسی IRANSansWeb استفاده می شود
- جدول باید اطلاعات مفید و مقایسه‌ای ارائه دهد

📌 قوانین توزیع کلمه کلیدی (بسیار مهم):
- باید متن با یک پراکندگی بسیار طبیعی و نرمال از کلمه کلیدی همراه باشد
- کلمه کلیدی در هر پاراگراف یک بار بیاید (نه خیلی پشت سر هم و مصنوعی)
- نه خیلی دور و بدون تکرار باشد
- از کلمات مرتبط و هم‌معنی نیز استفاده کن

📌 قوانین محتوا و لحن:
- کاربر نیاز به محتوای انسانی، با احساس، همراه با مثال دارد
- محتوا را طوری تهیه کن که کاربر احساس کند نیاز او برآورده می شود
- ابهام و سوالات او با بیان روان و ساده و بدون پیچیدگی پاسخ داده می شود
- مخاطب: پزشکان، وکلا، کلینیک های زیبایی، کارخانه ها و همه افرادی که نیاز به توسعه کسب و کار دارند
- هدف: ترغیب کاربر به انجام این خدمات (نه بیان علمی و سنگین)
- محتوا باید کاملا انسانی و کمی دوستانه باشد
- حتما توضیحات همراه مثال باشد و متن روان باشد

        📌 قوانین نگارشی (الزامی):
        - (من ، تو) بین کاما و کلمه قبل و بعد باید فاصله وجود داشته باشد
        - فعل ها فاصله مناسب زبان فارسی را داشته باشند: می شود ، می تواند ، خوانده ام ، می توانند ، نداشته است
        - فاصله حروف فارسی به صورت استاندارد: جا به جا ، طراحی سایت ، سئو سایت ، مد نظر ، پاره وقت ، قابل قبول ، جست و جو
        - فاصله بین کلمات باید به این شکل رعایت شوند: راه ها (درست) ، راهکار های (درست) ، وبسایت هایی (درست) - نه راهها ، راهکارهای
        - استفاده مداوم و پر تکرار از "تر" را کم و در حد طبیعی انجام بده
        - نیازی نیست پشت سر نوشته شود: سریع تر ، مهم تر ، کاربردی تر (متن را مصنوعی می کند)
        - متن باید انسانی و در بعضی قسمت های متن هیجانی و در بعضی قسمت ها دوستانه باشد

📌 قوانین عنوان سئو:
- حتما شامل کلمه کلیدی اصلی + یک متن ترغیب کننده و جذاب باشد

📌 قوانین طول محتوا:
- کل مقاله نهایی حداقل 1000 کلمه باشد
- محتوا باید جامع و کامل باشد
"""

EXAMPLES_BLOCK = r"""
مثال‌های عملی (نمونه ورودی -> خروجی):

        📝 مثال‌های نگارشی:
- کاما: نادرست -> "من،تو"; صحیح -> "من ، تو"
- فعل پیوسته: نادرست -> "میشود" یا "می‌شود"; صحیح -> "می شود"
- فاصله کلمات: نادرست -> "طراحیسایت" یا "جابهجا"; صحیح -> "طراحی سایت" ، "جا به جا"
        - فاصله کلمات مرکب: نادرست -> "راهها" ، "راهکارهای"; صحیح -> "راه ها" ، "راهکار های"
        - فاصله کلمات مرکب: نادرست -> "وبسایتهایی"; صحیح -> "وبسایت هایی"
        - استفاده از "تر": نادرست -> "سریع تر ، مهم تر ، کاربردی تر"; صحیح -> "سریع ، مهم ، کاربردی"

📝 مثال‌های عنوان H1:
- کلیدواژه "طراحی سایت پزشکی" -> "طراحی سایت پزشکی + صفر تا صد راهنمای کامل با نمونه کارها"
- کلیدواژه "سئو سایت" -> "سئو سایت + 10 روش طلایی افزایش رتبه در گوگل"
- کلیدواژه "جراحی بینی" -> "جراحی بینی + صفر تا صد عمل بینی به همراه نمونه کار"

📝 مثال‌های پاراگراف آغازین:
- شروع با کلمه کلیدی: "طراحی سایت پزشکی یکی از مهم‌ترین ابزارهای بازاریابی دیجیتال برای کلینیک‌ها و مطب‌های پزشکی است. در دنیای امروز که بیماران قبل از مراجعه به پزشک، ابتدا در اینترنت جست‌وجو می‌کنند، داشتن یک وب‌سایت حرفه‌ای و بهینه‌شده برای موتورهای جست‌وجو، می‌تواند تفاوت بزرگی در جذب بیماران جدید ایجاد کند."

📝 مثال‌های جدول:
<table style="font-family: IRANSansWeb; border-collapse: collapse; width: 100%; margin: 20px 0;">
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 12px; text-align: right;">مزایا</th>
<th style="border: 1px solid #ddd; padding: 12px; text-align: right;">قبل از طراحی سایت</th>
<th style="border: 1px solid #ddd; padding: 12px; text-align: right;">بعد از طراحی سایت</th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 12px;">تعداد بیماران</td>
<td style="border: 1px solid #ddd; padding: 12px;">10 نفر در ماه</td>
<td style="border: 1px solid #ddd; padding: 12px;">50 نفر در ماه</td>
</tr>
</table>

📝 مثال‌های توزیع کلمه کلیدی:
- پاراگراف 1: "طراحی سایت پزشکی امروزه ضروری است..."
- پاراگراف 2: "برای شروع طراحی سایت پزشکی، ابتدا باید..."
- پاراگراف 3: "هزینه طراحی سایت پزشکی بستگی به..."

📝 مثال‌های عنوان سئو:
- "طراحی سایت پزشکی: افزایش بیماران با وب‌سایت حرفه‌ای"
- "سئو سایت: راهنمای کامل افزایش رتبه در گوگل"
- "جراحی بینی: صفر تا صد عمل بینی با بهترین جراحان"

📝 مثال‌های لحن:
- هیجانی: "تصور کنید کسب‌وکارتان رونق بگیرد! با طراحی سایت پزشکی حرفه‌ای، می‌توانید..."
- دوستانه: "بیایید با هم بررسی کنیم چطور این کار را انجام دهید. در ادامه، مراحل طراحی سایت پزشکی را..."
- ترغیب‌کننده: "اگر می‌خواهید بیماران بیشتری جذب کنید، طراحی سایت پزشکی اولین قدم است..."
"""

def load_env(env_path: Optional[Path]):
    if env_path:
        if DOTENV:
            load_dotenv(dotenv_path=str(env_path))
            LOG.info("Loaded .env from %s", env_path)
        else:
            with env_path.open(encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            LOG.info("Loaded .env (simple parser) from %s", env_path)

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Provide via .env or env var.")
    return OpenAI(api_key=key)

_enc = tiktoken.get_encoding(ENCODING_NAME)

def embed_text(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    out = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        last_exc = None
        for attempt in range(API_RETRY):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                for item in resp.data:
                    out.append(list(item.embedding))
                break
            except Exception as e:
                last_exc = e
                wait = API_BACKOFF_BASE * (2 ** attempt)
                LOG.warning("Embedding attempt %d failed: %s — retrying in %.1fs", attempt + 1, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed embedding batch after retries: {last_exc}")
    return out

def embed_query(client: OpenAI, query: str, model: str = EMBEDDING_MODEL) -> List[float]:
    return embed_text(client, [query], model=model)[0]

def load_meta_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    meta = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            meta.append(json.loads(ln))
    return meta

def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {path}")
    return faiss.read_index(str(path))

def retrieve_top_k(client: OpenAI, index: faiss.Index, meta: List[Dict[str, Any]],
                   query: str, top_k: int = RETRIEVE_TOP_K) -> List[Tuple[Dict[str,Any], float]]:
    """
    Retrieve top_k chunks globally for the query.
    Searches for N = top_k * RETRIEVE_MULT, keeps unique by id, ordered by score desc.
    """
    vec = np.array([embed_query(client, query)], dtype="float32")
    faiss.normalize_L2(vec)
    N = top_k * RETRIEVE_MULT
    max_N = min(len(meta), top_k * 20)
    attempts = 0
    while attempts < 3:
        D, I = index.search(vec, N)
        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(meta):
                continue
            m = meta[idx]
            candidates.append((m, float(score)))
        # keep unique by chunk id and preserve order by score
        seen = set()
        filtered = []
        for m, s in candidates:
            cid = m.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            filtered.append((m, s))
        results = filtered[:top_k]
        if len(results) >= top_k:
            return results
        # increase N
        attempts += 1
        N = min(max_N, N * 2)
    LOG.warning("Insufficient retrieval results for query '%s'; returning %d", query, len(results))
    return results

def summarize_section_for_continuation(client: OpenAI, text: str) -> Dict[str, Any]:
    """
    Ask the model to produce a compact JSON brief.
    Improved prompt for strict JSON output.
    """
    user_prompt = (
        "خلاصه‌ای کوتاه و ساختارمند برای ادامهٔ متن تولید کن. خروجی فقط یک JSON معتبر بده با فیلدهای زیر:\n"
        "title (عنوان بخش)، last_sentence (آخرین جمله بخش)، main_points (لیستی از حداکثر 6 نکته کلیدی)، "
        "tone (یک کلمه: مثل 'ترغیب‌کننده' یا 'اطلاعی')، suggested_next_headings (آرایه‌ای از 3 تیتر پیشنهادی برای بخش بعدی).\n\n"
        f"متن:\n{text}\n\n"
        "فرمت خروجی: JSON خالص. هیچ متن اضافی ننویس. مطمئن شو تمام رشته‌ها به درستی بسته شوند و JSON معتبر باشد."
    )
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[{"role": "system", "content": "شما یک خلاصه‌ساز هستید که فقط JSON معتبر تولید می‌کند. هیچ متن دیگری ننویس."},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=SUMMARIZE_MAX_TOKENS,
            )
            out = resp.choices[0].message.content.strip()
            # remove code fences if present
            out = re.sub(r"^```json\s*|\s*```$", "", out, flags=re.I)
            # attempt to fix unterminated string by adding " at end if needed
            if out[-1] != '}':
                out += '" }'
            m = re.search(r"(\{.*\})", out, flags=re.S)
            json_text = m.group(1) if m else out
            parsed = json.loads(json_text)
            return parsed
        except Exception as e:
            last_exc = e
            LOG.warning("Summarization attempt %d failed: %s", attempt + 1, e)
            time.sleep(API_BACKOFF_BASE * (2 ** attempt))
    LOG.warning("Summarization failed: %s. Falling back.", last_exc)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_sentence = lines[-1] if lines else ""
    brief = {
        "title": "",
        "last_sentence": last_sentence[:200],
        "main_points": lines[:6],  # Increased to 6
        "tone": "ترغیب‌کننده",
        "suggested_next_headings": []
    }
    return brief

def generate_structure(client: OpenAI, index: faiss.Index, meta: List[Dict[str,Any]],
                       keyword: str, perfect_html_reference: Optional[str], model: str = DEFAULT_CHAT_MODEL,
                       temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    LOG.info("Generating blog structure for keyword=%s", keyword)
    
    # Enhanced query with keyword variations for better RAG retrieval
    keyword_variations = generate_keyword_variations(keyword)
    enhanced_query = f"{keyword} {' '.join(keyword_variations[:3])}"  # Use top 3 variations
    
    retrieved = retrieve_top_k(client, index, meta, enhanced_query, top_k=RETRIEVE_TOP_K)
    
    # Filter and prioritize relevant content based on keyword similarity
    relevant_chunks = []
    for m, score in retrieved:
        text = m.get("text", "").lower()
        title = m.get("section_title", "").lower()
        
        # Check if content is relevant to the keyword
        keyword_lower = keyword.lower()
        is_relevant = (
            keyword_lower in text or 
            keyword_lower in title or
            any(variation.lower() in text for variation in keyword_variations[:3]) or
            any(variation.lower() in title for variation in keyword_variations[:3])
        )
        
        if is_relevant or score > 0.7:  # Include high-scoring content even if not directly matching
            relevant_chunks.append((m, score))
    
    # If we don't have enough relevant content, use the original retrieved content
    if len(relevant_chunks) < 3:
        relevant_chunks = retrieved
    
    context_parts = []
    for m, score in relevant_chunks:
        src = m.get("source_file", "")
        ci = m.get("chunk_index", 0)
        txt = m.get("text", "")
        context_parts.append(f"--- منبع: [{src}] (chunk {ci}, score={score:.4f})\n{txt}")
    context_block = "\n\n".join(context_parts) if context_parts else ""

    perfect_ref_block = ""
    if perfect_html_reference:
        perfect_ref_block = (
            "الگوی مرجع (برای ساختار/جداسازی پاراگراف‌ها/استایل‌ها/وجود جدول و CTA):\n"
            + perfect_html_reference[:4000] + "\n\n"
        )

    prompt = (
        f"{RULES_BLOCK}\n"
        f"{EXAMPLES_BLOCK}\n\n"
        f"🎯 کلمه کلیدی اصلی: {keyword}\n"
        f"📚 داده‌های بازیابی‌شده (فقط از این منابع برای الهام‌گیری و استناد استفاده کن):\n{context_block}\n\n"
        f"{perfect_ref_block}"
        "🔍 وظیفه: ساختار مقاله را به صورت JSON تولید کن:\n"
        "{\n"
        f'  "h1_title": "عنوان H1 (شامل کلمه کلیدی اصلی "{keyword}" + عبارت ترغیب کننده)",\n'
        f'  "seo_title": "عنوان سئو (کلمه کلیدی اصلی "{keyword}" + متن جذاب)",\n'
        '  "sections": [\n'
        '    {"title": "تیتر بخش", "level": 2, "needs_table": false, "description": "توضیح کوتاه محتوا"},\n'
        '    {"title": "تیتر بخش", "level": 2, "needs_table": true, "description": "توضیح کوتاه محتوا"}\n'
        '  ]\n'
        "}\n\n"
        "⚠️ قوانین مهم ساختار:\n"
        f"- حتما کلمه کلیدی اصلی '{keyword}' را در عنوان H1 و سئو استفاده کن\n"
        "- تمام بخش‌ها باید مرتبط با موضوع '{keyword}' باشند\n"
        "- از داده‌های بازیابی‌شده برای الهام‌گیری استفاده کن، نه از دانش عمومی\n"
        "- حداقل 6 بخش داشته باش (به جز مقدمه)\n"
        "- حداقل 2 بخش مناسب برای جدول پیشنهاد کن (مقایسه، مزایا، مراحل، آمار)\n"
        "- عناوین باید جذاب و ترغیب‌کننده باشند\n"
        "- از اعداد و کلمات قدرتمند استفاده کن (10 روش، 5 مزیت، صفر تا صد)\n"
        "- هر بخش باید یک هدف مشخص داشته باشد\n"
        "- خروجی فقط JSON معتبر باشد"
    )

    generated = None
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "شما یک متخصص برنامه‌ریزی محتوا و ساختار مقاله هستید. تخصص شما در ایجاد ساختارهای بهینه و جذاب برای مقالات فارسی است که هم برای کاربران جذاب باشد و هم برای موتورهای جست‌وجو بهینه باشد. شما قوانین SEO و ساختار محتوا را به طور کامل رعایت می‌کنید."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = resp.choices[0].message.content.strip()
            out = re.sub(r"^```json\s*|\s*```$", "", out, flags=re.I)
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            last_exc = e
            wait = API_BACKOFF_BASE * (2 ** attempt)
            LOG.warning("Structure gen attempt %d failed: %s - retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to generate structure after retries: {last_exc}")

def build_section_prompt(keyword: str, section_title: str, section_level: int, section_idx: int, retrieved_chunks: List[Tuple[Dict[str,Any], float]],
                         prev_brief: Optional[Dict[str,Any]], perfect_html_reference: Optional[str], examples_block: str) -> str:
    context_parts = []
    for m, score in retrieved_chunks:
        src = m.get("source_file", "")
        ci = m.get("chunk_index", 0)
        txt = m.get("text", "")
        context_parts.append(f"--- منبع: [{src}] (chunk {ci}, score={score:.4f})\n{txt}")
    context_block = "\n\n".join(context_parts) if context_parts else ""

    prev_block = ""
    if prev_brief:
        prev_block = (
            "خلاصهٔ بخش قبلی (برای حفظ پیوستگی؛ از آن استفاده کن، ادامه بده از آخرین جمله، لحن را حفظ کن و تکرار محتوای دقیق بخش قبلی نکن):\n"
            + json.dumps(prev_brief, ensure_ascii=False, indent=2)
            + "\n\n"
        )

    perfect_ref_block = ""
    if perfect_html_reference:
        perfect_ref_block = (
            "الگوی مرجع:\n"
            + perfect_html_reference[:4000] + "\n\n"
        )

    if section_idx == 0:
        # Intro section: H1 + intro para
        task = (
            "📌 وظیفه بخش مقدمه:\n"
            " - <h1>عنوان H1</h1> را تولید کن (شامل کلمه کلیدی + عبارت ترغیب کننده)\n"
            " - پاراگراف آغازین 3-4 خطی که دقیقا با کلمه کلیدی شروع شود\n"
            " - محتوا باید هیجانی و ترغیب‌کننده باشد\n"
            " - از اعداد و آمار استفاده کن (مثل: 80% از بیماران، 5 برابر افزایش)\n"
            " - جدول اضافه نکن"
        )
    else:
        # Regular section
        needs_table = "needs_table" in section_title.lower() or any(word in section_title.lower() for word in ["مقایسه", "مزایا", "مراحل", "آمار", "جدول", "لیست"])
        table_instruction = ""
        if needs_table:
            table_instruction = (
                "\n - یک جدول مفید و مرتبط اضافه کن با استایل:\n"
                '<table style="font-family: IRANSansWeb; border-collapse: collapse; width: 100%; margin: 20px 0;">\n'
                '<tr style="background-color: #f2f2f2;">\n'
                '<th style="border: 1px solid #ddd; padding: 12px; text-align: right;">ستون 1</th>\n'
                '<th style="border: 1px solid #ddd; padding: 12px; text-align: right;">ستون 2</th>\n'
                '</tr>\n'
                '<tr>\n'
                '<td style="border: 1px solid #ddd; padding: 12px;">داده 1</td>\n'
                '<td style="border: 1px solid #ddd; padding: 12px;">داده 2</td>\n'
                '</tr>\n'
                '</table>'
            )
        
        task = (
            f"📌 وظیفه بخش: {section_title}\n"
            f" - <h{section_level}>{section_title}</h{section_level}> را تولید کن\n"
            " - محتوای جامع و مفصل برای این بخش بنویس\n"
            " - از مثال‌های عملی و کاربردی استفاده کن\n"
            " - لحن دوستانه و قابل فهم داشته باش\n"
            " - هر پاراگراف 3-4 خط باشد\n"
            f"{table_instruction}"
        )

    prompt = (
        f"{RULES_BLOCK}\n"
        f"{examples_block}\n\n"
        f"{prev_block}"
        f"🎯 بخش کنونی: {section_title} (سطح {section_level})\n"
        f"📊 شماره بخش: {section_idx}\n"
        f"🔑 کلمه کلیدی اصلی: {keyword}\n\n"
        "📚 داده‌های بازیابی‌شده (الزامی - حتما از این منابع استفاده کن):\n"
        f"{context_block}\n\n"
        f"{perfect_ref_block}"
        f"{task}\n\n"
        "⚠️ دستورالعمل‌های الزامی:\n"
        f" - حتما کلمه کلیدی '{keyword}' را در محتوا استفاده کن\n"
        " - فقط برای این بخش محتوا تولید کن؛ تکرار بخش قبلی نکن\n"
        " - جریان و پیوستگی با بخش قبلی را حفظ کن\n"
        " - کلمه کلیدی را به طور طبیعی در هر پاراگراف یک بار استفاده کن\n"
        " - از کلمات مرتبط و هم‌معنی نیز استفاده کن\n"
        " - محتوا باید انسانی، با احساس و همراه با مثال باشد\n"
        " - حتما از اطلاعات بازیابی‌شده برای دقت و جلوگیری از توهم استفاده کن\n"
        " - اگر اطلاعات کافی در منابع نیست، از دانش عمومی استفاده کن اما حتما کلمه کلیدی را رعایت کن\n"
        " - خروجی را به صورت HTML معتبر بنویس\n"
        " - در انتها، اگر از منابع استفاده کردی، <p><strong>منابع:</strong> [source_file]</p> اضافه کن\n"
    )
    return prompt

def coherence_edit(client: OpenAI, combined: str, keyword: str, rules_block: str, examples_block: str,
                   model: str = DEFAULT_CHAT_MODEL, temperature: float = DEFAULT_TEMPERATURE,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    word_count = count_words(combined)
    expand_note = f"اگر کمتر از {MIN_WORD_COUNT} کلمه است، به طور طبیعی گسترش بده." if word_count < MIN_WORD_COUNT else ""
    prompt = (
        f"{rules_block}\n"
        f"{examples_block}\n\n"
        "🔍 بررسی و بهبود مقاله:\n"
        "مقاله HTML زیر را برای موارد زیر بررسی و بهبود بده:\n"
        "✅ پیوستگی و جریان بین بخش‌ها\n"
        "✅ توزیع طبیعی و متعادل کلمه کلیدی\n"
        "✅ لحن یکپارچه و انسانی\n"
        "✅ رعایت قوانین نگارشی فارسی\n"
        "✅ کیفیت و عمق محتوا\n"
        "✅ استفاده از مثال‌های کاربردی\n"
        "✅ ترغیب‌کنندگی و جذابیت\n"
        f"📈 {expand_note}\n\n"
        "🎯 خروجی نهایی:\n"
        "- HTML بهبود یافته و بهینه\n"
        "- محتوای روان و قابل فهم\n"
        "- رعایت کامل قوانین فارسی\n"
        "- جذاب و ترغیب‌کننده\n\n"
        f"📄 مقاله فعلی:\n{combined}"
    )
    generated = None
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "شما یک ویرایشگر حرفه‌ای و متخصص بهبود محتوای فارسی هستید. تخصص شما در بهبود کیفیت، پیوستگی، و جذابیت مقالات فارسی است. شما قوانین نگارشی، SEO، و ساختار محتوا را به طور کامل رعایت می‌کنید و محتوای نهایی را بهینه‌سازی می‌کنید."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens * 2,  # Allow more for edit
            )
            generated = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            last_exc = e
            wait = API_BACKOFF_BASE * (2 ** attempt)
            LOG.warning("Edit attempt %d failed: %s - retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    if generated is None:
        LOG.warning("Coherence edit failed: %s. Returning original.", last_exc)
        return combined
    return normalize_persian_spacing_and_mi(generated)

def generate_blog(client: OpenAI, index: faiss.Index, meta: List[Dict[str,Any]],
                  keyword: str, out_path: Path, perfect_html_path: Optional[Path] = None,
                  model: str = DEFAULT_CHAT_MODEL, temperature: float = DEFAULT_TEMPERATURE,
                  max_tokens: int = DEFAULT_MAX_TOKENS):
    LOG.info("Generating comprehensive blog for keyword=%s", keyword)
    perfect_html_reference = None
    if perfect_html_path and perfect_html_path.exists():
        perfect_html_reference = perfect_html_path.read_text(encoding="utf-8", errors="ignore")

    # Phase 1: Generate comprehensive first section with H1 and 2-3 paragraphs
    LOG.info("Phase 1: Generating comprehensive introduction section...")
    phase1_content, next_section_prompt = generate_phase1_content(
        client, index, meta, keyword, perfect_html_reference, model, temperature, max_tokens
    )
    
    # Clean Phase 1 content
    phase1_content = clean_html_content(phase1_content)
    
    # Phase 2: Generate remaining content based on Phase 1 prompt
    LOG.info("Phase 2: Generating remaining content based on Phase 1 prompt...")
    phase2_content = generate_phase2_content(
        client, index, meta, keyword, next_section_prompt, perfect_html_reference, model, temperature, max_tokens
    )
    
    # Clean Phase 2 content
    phase2_content = clean_html_content(phase2_content)
    
    # Combine both phases
    combined = phase1_content + "\n" + phase2_content
    
    # Phase 3: Final validation and improvement
    LOG.info("Phase 3: Final validation and improvement...")
    
    # Perform advanced content quality check
    quality_metrics = advanced_content_quality_check(combined, keyword)
    LOG.info("Content quality metrics: %s", quality_metrics)
    
    # If quality is below threshold, perform enhanced validation
    if not quality_metrics["is_high_quality"]:
        LOG.info("Content quality below threshold, performing enhanced validation...")
        combined = validate_and_improve_content(client, combined, keyword, model, temperature, max_tokens)
        
        # Re-check quality after improvement
        quality_metrics = advanced_content_quality_check(combined, keyword)
        LOG.info("Post-improvement quality metrics: %s", quality_metrics)
    else:
        LOG.info("Content quality is adequate, performing standard validation...")
        combined = validate_and_improve_content(client, combined, keyword, model, temperature, max_tokens)
    
    # Final coherence edit
    LOG.info("Performing final coherence edit...")
    combined = coherence_edit(client, combined, keyword, RULES_BLOCK, EXAMPLES_BLOCK, model, temperature, max_tokens)

    # Final content enhancement
    if ENHANCED_CONTENT_GENERATION:
        LOG.info("Applying final content enhancements...")
        combined = enhance_content_with_examples(combined, keyword)
        combined = optimize_keyword_distribution(combined, keyword)
        combined = enhance_keyword_distribution(combined, keyword)
        combined = add_engaging_elements(combined)
        combined = normalize_persian_spacing_and_mi(combined)
        
        # Validate keyword adherence
        if not validate_keyword_adherence(combined, keyword):
            LOG.warning("Generated content may not sufficiently adhere to keyword: %s", keyword)

    word_count = count_words(combined)
    LOG.info("Final word count: %d", word_count)
    
    # Extract SEO title from H1
    seo_title = extract_seo_title_from_content(combined, keyword)

    # Wrap in full HTML with SEO title
    full_html = f"""<!DOCTYPE html>
<html lang="fa">
<head>
    <meta charset="UTF-8">
    <title>{seo_title}</title>
    <!-- Add styles for table if needed -->
</head>
<body dir="rtl">
{combined}
</body>
</html>"""

    combined = normalize_persian_spacing_and_mi(full_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")
    LOG.info("Final blog saved to %s (%d words)", out_path, count_words(combined))
    return out_path

def generate_phase1_content(client: OpenAI, index: faiss.Index, meta: List[Dict[str,Any]], 
                           keyword: str, perfect_html_reference: Optional[str], 
                           model: str, temperature: float, max_tokens: int) -> Tuple[str, str]:
    """Generate comprehensive first section with H1 and 2-3 paragraphs, plus prompt for next section."""
    
    # Get comprehensive RAG content for the keyword
    keyword_variations = generate_keyword_variations(keyword)
    enhanced_query = f"{keyword} {' '.join(keyword_variations[:5])}"  # Use more variations
    
    # Retrieve more content for comprehensive generation
    retrieved = retrieve_top_k(client, index, meta, enhanced_query, top_k=RETRIEVE_TOP_K * 2)
    
    # Use advanced RAG content selection
    selected_chunks = advanced_rag_content_selection(retrieved, keyword, "", 20)
    
    # Build comprehensive context
    context_parts = []
    for m, score in selected_chunks:
        src = m.get("source_file", "")
        ci = m.get("chunk_index", 0)
        txt = m.get("text", "")
        context_parts.append(f"--- منبع: [{src}] (chunk {ci}, score={score:.4f})\n{txt}")
    context_block = "\n\n".join(context_parts) if context_parts else ""
    
    perfect_ref_block = ""
    if perfect_html_reference:
        perfect_ref_block = (
            "الگوی مرجع (برای ساختار/جداسازی پاراگراف‌ها/استایل‌ها/وجود جدول و CTA):\n"
            + perfect_html_reference[:4000] + "\n\n"
        )
    
    # Create comprehensive Phase 1 prompt
    phase1_prompt = f"""
{RULES_BLOCK}
{EXAMPLES_BLOCK}

🎯 کلمه کلیدی اصلی: {keyword}
📚 داده‌های جامع بازیابی‌شده (از این منابع برای الهام‌گیری و استناد استفاده کن):
{context_block}

{perfect_ref_block}

🔍 وظیفه Phase 1: تولید بخش مقدمه جامع و کامل

📌 خروجی مورد نیاز:
1. <h1>عنوان H1</h1> (شامل کلمه کلیدی "{keyword}" + عبارت ترغیب کننده)
2. پاراگراف اول: 3-4 خط که دقیقا با کلمه کلیدی شروع شود
3. پاراگراف دوم: 3-4 خط ادامه موضوع
4. پاراگراف سوم: 3-4 خط تکمیل مقدمه
5. یک جدول مرتبط و مفید (اختیاری)
6. در انتها، 2-3 خط prompt برای بخش بعدی

⚠️ قوانین مهم:
- حتما کلمه کلیدی "{keyword}" را در عنوان H1 و محتوا استفاده کن
- از داده‌های بازیابی‌شده برای الهام‌گیری و دقت استفاده کن
- محتوا باید جامع، جذاب و ترغیب‌کننده باشد
- لحن انسانی، دوستانه و هیجانی داشته باش
- از مثال‌های عملی و آمار استفاده کن
- هر پاراگراف 3-4 خط باشد
- HTML معتبر تولید کن (بدون کد بلاک)
- از کلمات هیجانی و ترغیب‌کننده استفاده کن
- محتوا باید کاربر را به ادامه خواندن ترغیب کند

📝 فرمت خروجی (فقط HTML خالص):
<h1>عنوان H1 شامل کلمه کلیدی</h1>
<p>پاراگراف اول...</p>
<p>پاراگراف دوم...</p>
<p>پاراگراف سوم...</p>
[جدول اختیاری]
<p><strong>منابع:</strong> [source_files]</p>

<!-- PROMPT برای بخش بعدی -->
<div style="display: none;">
NEXT_SECTION_PROMPT: [2-3 خط توضیح برای بخش بعدی]
</div>
"""

    generated = None
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "شما یک نویسنده حرفه‌ای و متخصص تولید محتوای فارسی هستید. تخصص شما در ایجاد محتوای جامع، جذاب و ترغیب‌کننده است. شما قوانین نگارشی فارسی را به طور کامل رعایت می‌کنید و محتوای انسانی و با کیفیت تولید می‌کنید. شما از داده‌های بازیابی‌شده برای دقت و الهام‌گیری استفاده می‌کنید."},
                    {"role": "user", "content": phase1_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens * 2,  # Allow more tokens for comprehensive content
            )
            generated = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            last_exc = e
            wait = API_BACKOFF_BASE * (2 ** attempt)
            LOG.warning("Phase 1 attempt %d failed: %s - retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    
    if generated is None:
        raise RuntimeError(f"Failed to generate Phase 1 content: {last_exc}")

    # Extract next section prompt
    next_section_prompt = extract_next_section_prompt(generated)
    
    # Clean the content
    generated = normalize_persian_spacing_and_mi(generated)
    
    LOG.info("Phase 1 generated (%d chars)", len(generated))
    return generated, next_section_prompt

def generate_phase2_content(client: OpenAI, index: faiss.Index, meta: List[Dict[str,Any]], 
                           keyword: str, next_section_prompt: str, perfect_html_reference: Optional[str], 
                           model: str, temperature: float, max_tokens: int) -> str:
    """Generate remaining content based on Phase 1 prompt."""
    
    # Get diverse RAG content for remaining sections
    keyword_variations = generate_keyword_variations(keyword)
    enhanced_query = f"{keyword} {' '.join(keyword_variations[:5])}"
    
    # Retrieve diverse content for remaining sections
    retrieved = retrieve_top_k(client, index, meta, enhanced_query, top_k=RETRIEVE_TOP_K * 2)
    
    # Use advanced RAG content selection for Phase 2
    selected_chunks = advanced_rag_content_selection(retrieved, keyword, next_section_prompt, 25)
    
    # Build comprehensive context
    context_parts = []
    for m, score in selected_chunks:
        src = m.get("source_file", "")
        ci = m.get("chunk_index", 0)
        txt = m.get("text", "")
        context_parts.append(f"--- منبع: [{src}] (chunk {ci}, score={score:.4f})\n{txt}")
    context_block = "\n\n".join(context_parts) if context_parts else ""
    
    perfect_ref_block = ""
    if perfect_html_reference:
        perfect_ref_block = (
            "الگوی مرجع:\n"
            + perfect_html_reference[:4000] + "\n\n"
        )
    
    # Create comprehensive Phase 2 prompt
    phase2_prompt = f"""
{RULES_BLOCK}
{EXAMPLES_BLOCK}

🎯 کلمه کلیدی اصلی: {keyword}
📝 راهنمای بخش بعدی از Phase 1: {next_section_prompt}
📚 داده‌های جامع بازیابی‌شده (از این منابع برای الهام‌گیری و استناد استفاده کن):
{context_block}

{perfect_ref_block}

🔍 وظیفه Phase 2: تولید محتوای جامع و کامل برای ادامه مقاله

📌 خروجی مورد نیاز:
- حداقل 10-12 بخش H2 با محتوای جامع و مفصل
- هر بخش 3-4 پاراگراف 3-4 خطی
- حداقل 4 جدول مفید و مرتبط
- محتوای متنوع، جذاب و کاربردی
- استفاده از مثال‌های عملی، آمار و راهکارها
- لحن دوستانه، ترغیب‌کننده و هیجانی
- حداقل 1000-1200 کلمه محتوای جدید
- استفاده از کلمات قدرتمند و ترغیب‌کننده

⚠️ قوانین مهم:
- حتما کلمه کلیدی "{keyword}" را در محتوا استفاده کن
- از داده‌های بازیابی‌شده برای الهام‌گیری و دقت استفاده کن
- محتوا باید جامع، متنوع و جذاب باشد
- هر بخش باید هدف مشخص و کاربردی داشته باشد
- از مثال‌های عملی، آمار و راهکارهای کاربردی استفاده کن
- HTML معتبر تولید کن (بدون کد بلاک)
- در انتها منابع را ذکر کن
- محتوا باید حداقل 1000 کلمه باشد
- از کلمات هیجانی و ترغیب‌کننده استفاده کن
- هر بخش باید کاربر را به اقدام ترغیب کند

📝 فرمت خروجی (فقط HTML خالص):
<h2>عنوان بخش 1</h2>
<p>محتوای بخش...</p>
<p>ادامه محتوا...</p>
<p>تکمیل بخش...</p>

<h2>عنوان بخش 2</h2>
<p>محتوای بخش...</p>
<p>ادامه محتوا...</p>
[جدول مرتبط]

<h2>عنوان بخش 3</h2>
<p>محتوای بخش...</p>
<p>ادامه محتوا...</p>
<p>تکمیل بخش...</p>

[سایر بخش‌ها...]

<p><strong>منابع:</strong> [source_files]</p>
"""
    
    generated = None
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "شما یک نویسنده حرفه‌ای و متخصص تولید محتوای فارسی هستید. تخصص شما در ایجاد محتوای جامع، متنوع و جذاب است. شما قوانین نگارشی فارسی را به طور کامل رعایت می‌کنید و محتوای انسانی و با کیفیت تولید می‌کنید. شما از داده‌های بازیابی‌شده برای دقت و الهام‌گیری استفاده می‌کنید."},
                    {"role": "user", "content": phase2_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens * 3,  # Allow more tokens for comprehensive content
            )
            generated = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            last_exc = e
            wait = API_BACKOFF_BASE * (2 ** attempt)
            LOG.warning("Phase 2 attempt %d failed: %s - retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    
    if generated is None:
        raise RuntimeError(f"Failed to generate Phase 2 content: {last_exc}")
    
    # Clean the content
    generated = normalize_persian_spacing_and_mi(generated)
    
    LOG.info("Phase 2 generated (%d chars)", len(generated))
    return generated

def extract_next_section_prompt(content: str) -> str:
    """Extract next section prompt from Phase 1 content."""
    import re
    
    # Look for NEXT_SECTION_PROMPT in the content
    pattern = r'NEXT_SECTION_PROMPT:\s*(.+?)(?=\n|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: generate a generic prompt
    return "ادامه موضوع با جزئیات بیشتر، مثال‌های عملی و راهکارهای کاربردی"

def clean_html_content(content: str) -> str:
    """Clean HTML content by removing code blocks and fixing formatting."""
    import re
    
    # Remove HTML code blocks
    content = re.sub(r'```html\s*', '', content)
    content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
    
    # Remove any remaining code block markers
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    return content.strip()

def validate_and_improve_content(client: OpenAI, content: str, keyword: str, 
                                model: str, temperature: float, max_tokens: int) -> str:
    """Final validation and improvement phase for the complete blog."""
    
    word_count = count_words(content)
    
    validation_prompt = f"""
{RULES_BLOCK}
{EXAMPLES_BLOCK}

🎯 کلمه کلیدی اصلی: {keyword}
📄 محتوای فعلی (تعداد کلمات: {word_count}):
{content}

🔍 وظیفه: بررسی و بهبود نهایی محتوا به عنوان یک ویرایشگر حرفه‌ای

📌 بررسی‌های جامع مورد نیاز:

1. **بررسی کلمه کلیدی:**
   - آیا کلمه کلیدی "{keyword}" به اندازه کافی و طبیعی استفاده شده؟
   - آیا توزیع کلمه کلیدی در متن مناسب است؟
   - آیا از کلمات مرتبط و هم‌معنی استفاده شده؟

2. **بررسی طول محتوا:**
   - آیا محتوا حداقل 1500 کلمه است؟
   - اگر کوتاه است، بخش‌های جدید و مفید اضافه کن
   - اگر طولانی است، بخش‌های غیرضروری را حذف کن

3. **بررسی قوانین نگارشی فارسی:**
   - فاصله کاما: "من ، تو" (نه "من،تو")
   - فعل ها: "می شود" ، "می تواند" (نه "میشود")
   - فاصله کلمات: "طراحی سایت" ، "جا به جا" (نه "طراحیسایت")
   - فاصله کلمات مرکب: "راه ها" ، "راهکار های" ، "وبسایت هایی" (نه "راهها" ، "راهکارهای")
   - استفاده از "تر": طبیعی و کم (نه "سریع تر" ، "مهم تر")

4. **بررسی لحن و کیفیت:**
   - آیا لحن انسانی و دوستانه است؟
   - آیا در بعضی قسمت‌ها هیجانی و ترغیب‌کننده است؟
   - آیا متن روان و قابل فهم است؟
   - آیا مثال‌های عملی و کاربردی دارد؟
   - آیا از کلمات قدرتمند و ترغیب‌کننده استفاده شده؟

5. **بررسی ساختار و محتوا:**
   - آیا عنوان H1 شامل کلمه کلیدی است؟
   - آیا پاراگراف اول با کلمه کلیدی شروع می‌شود؟
   - آیا تمام عنوان‌ها H2 هستند (مگر زیرمجموعه باشند)؟
   - آیا هر پاراگراف 3-4 خط است؟
   - آیا حداقل 4 جدول مفید و مرتبط وجود دارد؟

6. **بررسی کامل بودن:**
   - آیا محتوا جامع و کامل است؟
   - آیا تمام جنبه‌های موضوع پوشش داده شده؟
   - آیا مثال‌ها و آمار کافی وجود دارد؟
   - آیا محتوا ترغیب‌کننده و کاربردی است؟
   - آیا محتوا کاربر را به اقدام ترغیب می‌کند؟

7. **بررسی کیفیت محتوا:**
   - آیا محتوا منحصر به فرد و خلاقانه است؟
   - آیا از آمار و داده‌های معتبر استفاده شده؟
   - آیا محتوا برای موتورهای جست‌وجو بهینه است؟
   - آیا محتوا ارزش واقعی برای کاربر دارد؟

⚠️ **دستورالعمل‌های بهبود:**
- اگر محتوا کوتاه است: بخش‌های جدید اضافه کن، مثال‌های بیشتر ارائه ده، جدول‌های مفید اضافه کن
- اگر محتوا طولانی است: بخش‌های غیرضروری را حذف کن، محتوا را فشرده کن
- اگر لحن مناسب نیست: متن را انسانی‌تر و دوستانه‌تر کن
- اگر قوانین نگارشی رعایت نشده: تمام موارد را اصلاح کن
- اگر ساختار مناسب نیست: عنوان‌ها و پاراگراف‌ها را اصلاح کن
- اگر محتوا کسل‌کننده است: کلمات هیجانی و ترغیب‌کننده اضافه کن

📝 **خروجی مورد نیاز:**
- محتوای کامل، بهبود یافته و بی‌نقص
- حداقل 1500 کلمه
- رعایت کامل تمام قوانین
- HTML معتبر و تمیز
- لحن انسانی، دوستانه و ترغیب‌کننده
- ساختار مناسب و محتوای جامع
- محتوای خلاقانه و منحصر به فرد

فقط HTML خالص تولید کنید (بدون کد بلاک).
"""
    
    generated = None
    last_exc = None
    for attempt in range(API_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "شما یک ویرایشگر حرفه‌ای و متخصص بهبود محتوای فارسی هستید. شما مانند یک انسان متخصص که قوانین نگارشی فارسی را به طور کامل می‌داند، عمل می‌کنید. تخصص شما در بررسی، بهبود و تکمیل محتوای فارسی است. شما باید محتوا را به گونه‌ای ویرایش کنید که گویی یک انسان متخصص آن را نوشته است. شما قوانین نگارشی فارسی را به طور کامل رعایت می‌کنید، لحن انسانی و دوستانه ایجاد می‌کنید، و محتوای با کیفیت و کامل تولید می‌کنید. شما باید هر بخش از محتوا را بررسی کنید و اگر لحن مناسب نیست، آن را انسانی‌تر و دوستانه‌تر کنید."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens * 2,  # Allow more tokens for comprehensive improvement
            )
            generated = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            last_exc = e
            wait = API_BACKOFF_BASE * (2 ** attempt)
            LOG.warning("Validation attempt %d failed: %s - retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)
    
    if generated is None:
        LOG.warning("Validation failed: %s. Returning original content.", last_exc)
        return content
    
    # Clean the improved content
    generated = clean_html_content(generated)
    generated = normalize_persian_spacing_and_mi(generated)
    
    LOG.info("Content validated and improved (%d chars)", len(generated))
    return generated

def extract_seo_title_from_content(content: str, keyword: str) -> str:
    """Extract SEO title from H1 content."""
    import re
    
    # Look for H1 tag
    h1_pattern = r'<h1[^>]*>(.*?)</h1>'
    match = re.search(h1_pattern, content, re.DOTALL)
    
    if match:
        h1_title = match.group(1).strip()
        # Clean HTML tags
        h1_title = re.sub(r'<[^>]+>', '', h1_title)
        return h1_title
    
    # Fallback: use keyword
    return f"{keyword} - راهنمای کامل"

def advanced_rag_content_selection(retrieved: List[Tuple[Dict[str, Any], float]], 
                                  keyword: str, section_title: str = "", 
                                  max_chunks: int = 20) -> List[Tuple[Dict[str, Any], float]]:
    """Advanced RAG content selection with diversity and relevance optimization."""
    
    if not retrieved:
        return []
    
    # Generate keyword variations for better matching
    keyword_variations = generate_keyword_variations(keyword)
    keyword_lower = keyword.lower()
    section_lower = section_title.lower()
    
    # Score each chunk based on relevance and diversity
    scored_chunks = []
    used_sources = set()
    used_titles = set()
    
    for m, score in retrieved:
        text = m.get("text", "").lower()
        title = m.get("section_title", "").lower()
        source = m.get("source_file", "")
        
        # Calculate relevance score
        relevance_score = 0.0
        
        # Direct keyword match
        if keyword_lower in text:
            relevance_score += 0.4
        if keyword_lower in title:
            relevance_score += 0.3
            
        # Section title match
        if section_lower and section_lower in text:
            relevance_score += 0.2
        if section_lower and section_lower in title:
            relevance_score += 0.15
            
        # Keyword variations match
        for variation in keyword_variations[:5]:
            if variation.lower() in text:
                relevance_score += 0.1
            if variation.lower() in title:
                relevance_score += 0.05
        
        # Original FAISS score
        relevance_score += score * 0.2
        
        # Calculate diversity score
        diversity_score = 0.0
        if source not in used_sources:
            diversity_score += 0.3
        if title not in used_titles:
            diversity_score += 0.2
        
        # Combined score
        combined_score = (RELEVANCE_WEIGHT * relevance_score + 
                         DIVERSITY_WEIGHT * diversity_score)
        
        scored_chunks.append((m, combined_score, relevance_score, diversity_score))
    
    # Sort by combined score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Select diverse and relevant chunks
    selected_chunks = []
    used_sources = set()
    used_titles = set()
    
    for m, combined_score, relevance_score, diversity_score in scored_chunks:
        source = m.get("source_file", "")
        title = m.get("section_title", "")
        
        # Prioritize high-relevance chunks
        if relevance_score > 0.5 or len(selected_chunks) < 5:
            selected_chunks.append((m, combined_score))
            used_sources.add(source)
            used_titles.add(title)
        # Add diverse chunks if we have space
        elif (diversity_score > 0.3 and 
              source not in used_sources and 
              len(selected_chunks) < max_chunks):
            selected_chunks.append((m, combined_score))
            used_sources.add(source)
            used_titles.add(title)
        
        if len(selected_chunks) >= max_chunks:
            break
    
    # If we don't have enough chunks, add the best remaining ones
    if len(selected_chunks) < 5:
        for m, combined_score, relevance_score, diversity_score in scored_chunks:
            if (m, combined_score) not in selected_chunks:
                selected_chunks.append((m, combined_score))
                if len(selected_chunks) >= 10:
                    break
    
    return selected_chunks

def generate_keyword_variations(keyword: str) -> List[str]:
    """Generate related keywords and variations for better content diversity."""
    variations = [keyword]
    
    # Add common Persian variations
    if "طراحی" in keyword:
        variations.extend([keyword.replace("طراحی", "ساخت"), keyword.replace("طراحی", "ایجاد")])
    if "سایت" in keyword:
        variations.extend([keyword.replace("سایت", "وب‌سایت"), keyword.replace("سایت", "پورتال")])
    if "پزشکی" in keyword:
        variations.extend([keyword.replace("پزشکی", "درمانی"), keyword.replace("پزشکی", "کلینیکی")])
    
    # Add LSI keywords
    if "طراحی سایت" in keyword:
        variations.extend(["سئو سایت", "بهینه‌سازی سایت", "راه‌اندازی سایت", "توسعه وب"])
    if "سئو" in keyword:
        variations.extend(["بهینه‌سازی موتور جست‌وجو", "رتبه‌بندی گوگل", "بازاریابی دیجیتال"])
    
    # Add WordPress and security specific variations
    if "وردپرس" in keyword or "wordpress" in keyword.lower():
        variations.extend(["وردپرس", "WordPress", "سیستم مدیریت محتوا", "CMS"])
    if "امنیت" in keyword or "security" in keyword.lower():
        variations.extend(["امنیت", "Security", "حفاظت", "محافظت", "ایمنی"])
    if "امنیت سایت" in keyword:
        variations.extend(["امنیت وب‌سایت", "حفاظت سایت", "امنیت آنلاین", "امنیت دیجیتال"])
    if "امنیت سایت وردپرسی" in keyword:
        variations.extend([
            "امنیت وردپرس", "حفاظت وردپرس", "امنیت سایت وردپرس", 
            "WordPress Security", "امنیت CMS", "حفاظت وب‌سایت وردپرس"
        ])
    
    return list(set(variations))  # Remove duplicates

def enhance_content_with_examples(text: str, keyword: str) -> str:
    """Enhance content with relevant examples and statistics."""
    # This function can be expanded to add more sophisticated content enhancement
    # For now, we'll rely on the AI model to generate examples based on our prompts
    return text

def optimize_keyword_distribution(text: str, keyword: str) -> str:
    """Optimize keyword distribution throughout the content."""
    # Count current keyword occurrences
    keyword_count = text.lower().count(keyword.lower())
    
    # If keyword appears too rarely, this will be handled by the AI model
    # through our enhanced prompts that emphasize natural keyword distribution
    
    return text

def add_engaging_elements(text: str) -> str:
    """Add engaging elements like questions, statistics, and calls-to-action."""
    # This function can be expanded to add more engaging elements
    # For now, we rely on the AI model to generate engaging content
    return text

def validate_keyword_adherence(text: str, keyword: str) -> bool:
    """Validate that the generated content properly adheres to the keyword."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Check if keyword appears in the content
    keyword_count = text_lower.count(keyword_lower)
    
    # Check for keyword variations
    variations = generate_keyword_variations(keyword)
    variation_count = sum(text_lower.count(var.lower()) for var in variations if var.lower() != keyword_lower)
    
    # Content should have at least 3 mentions of the keyword or its variations
    total_mentions = keyword_count + variation_count
    
    return total_mentions >= 3

def advanced_content_quality_check(content: str, keyword: str) -> Dict[str, Any]:
    """Advanced content quality validation with detailed metrics."""
    import re
    
    quality_metrics = {
        "word_count": count_words(content),
        "keyword_density": 0.0,
        "persian_typo_score": 0.0,
        "structure_score": 0.0,
        "engagement_score": 0.0,
        "completeness_score": 0.0,
        "overall_score": 0.0
    }
    
    # Word count check
    word_count = quality_metrics["word_count"]
    quality_metrics["word_count_adequate"] = word_count >= MIN_WORD_COUNT
    
    # Keyword density calculation
    text_lower = content.lower()
    keyword_lower = keyword.lower()
    keyword_count = text_lower.count(keyword_lower)
    keyword_density = (keyword_count / word_count) * 100 if word_count > 0 else 0
    quality_metrics["keyword_density"] = keyword_density
    quality_metrics["keyword_adequate"] = 0.5 <= keyword_density <= 3.0
    
    # Persian typography check
    typo_issues = 0
    total_checks = 0
    
    # Check comma spacing
    comma_pattern = r'[^\s]،[^\s]'
    comma_issues = len(re.findall(comma_pattern, content))
    typo_issues += comma_issues
    total_checks += 1
    
    # Check verb spacing
    verb_pattern = r'می[ا-ی]'
    verb_issues = len(re.findall(verb_pattern, content))
    typo_issues += verb_issues
    total_checks += 1
    
    # Check compound word spacing
    compound_issues = 0
    compound_patterns = [r'راهها', r'راهکارهای', r'وبسایتهایی']
    for pattern in compound_patterns:
        compound_issues += len(re.findall(pattern, content))
    typo_issues += compound_issues
    total_checks += 1
    
    quality_metrics["persian_typo_score"] = max(0, 1 - (typo_issues / (total_checks * 10)))
    quality_metrics["typo_adequate"] = quality_metrics["persian_typo_score"] > 0.8
    
    # Structure check
    h1_count = len(re.findall(r'<h1[^>]*>', content))
    h2_count = len(re.findall(r'<h2[^>]*>', content))
    p_count = len(re.findall(r'<p[^>]*>', content))
    table_count = len(re.findall(r'<table[^>]*>', content))
    
    structure_score = 0
    if h1_count >= 1:
        structure_score += 0.2
    if h2_count >= 6:
        structure_score += 0.3
    if p_count >= 15:
        structure_score += 0.3
    if table_count >= 2:
        structure_score += 0.2
    
    quality_metrics["structure_score"] = structure_score
    quality_metrics["structure_adequate"] = structure_score >= 0.8
    
    # Engagement check (emotional words, examples, etc.)
    emotional_words = len(re.findall(r'(تصور کنید|بیایید|حتما|مطمئنا|قطعا|بدون شک)', content))
    example_words = len(re.findall(r'(مثال|برای مثال|به عنوان مثال|مثلا)', content))
    question_words = len(re.findall(r'(\?|چگونه|چرا|چه|کدام)', content))
    
    engagement_score = min(1.0, (emotional_words + example_words + question_words) / 20)
    quality_metrics["engagement_score"] = engagement_score
    quality_metrics["engagement_adequate"] = engagement_score >= 0.3
    
    # Completeness check
    completeness_indicators = [
        r'مقدمه|آغاز|شروع',
        r'نتیجه|جمع‌بندی|خلاصه',
        r'مزایا|فواید|مزیت',
        r'معایب|نکات|توجه',
        r'راهکار|راه‌حل|پیشنهاد'
    ]
    
    completeness_score = 0
    for indicator in completeness_indicators:
        if re.search(indicator, content, re.IGNORECASE):
            completeness_score += 0.2
    
    quality_metrics["completeness_score"] = min(1.0, completeness_score)
    quality_metrics["completeness_adequate"] = completeness_score >= 0.6
    
    # Overall score calculation
    scores = [
        quality_metrics["word_count_adequate"],
        quality_metrics["keyword_adequate"],
        quality_metrics["typo_adequate"],
        quality_metrics["structure_adequate"],
        quality_metrics["engagement_adequate"],
        quality_metrics["completeness_adequate"]
    ]
    
    quality_metrics["overall_score"] = sum(scores) / len(scores)
    quality_metrics["is_high_quality"] = quality_metrics["overall_score"] >= CONTENT_QUALITY_THRESHOLD
    
    return quality_metrics

def enhance_keyword_distribution(text: str, keyword: str) -> str:
    """Enhance keyword distribution if it's insufficient."""
    if not validate_keyword_adherence(text, keyword):
        # This would trigger additional content generation or modification
        # For now, we'll rely on the AI model to handle this through prompts
        pass
    return text

def normalize_persian_spacing_and_mi(text: str) -> str:
    text = text.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")
    # mi with space
    text = re.sub(r"ن?می[\u200C\-]?(?=[\u0600-\u06FF])", lambda m: m.group(0).replace("\u200C", " ").replace("-", " ").replace("می", "می "), text)
    # comma spacing
    text = re.sub(r"\s*،\s*", " ، ", text)
    text = re.sub(r"\s*,\s*", " , ", text)
    
    # Fix new spacing rules for compound words
    text = re.sub(r'راهها', r'راه ها', text)
    text = re.sub(r'راهکارهای', r'راهکار های', text)
    text = re.sub(r'وبسایتهایی', r'وبسایت هایی', text)
    text = re.sub(r'راه([\s]*)(ها)', r'راه ها', text)
    text = re.sub(r'راهکار([\s]*)(های)', r'راهکار های', text)
    text = re.sub(r'وبسایت([\s]*)(هایی)', r'وبسایت هایی', text)
    
    # collapse spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def count_words(text: str) -> int:
    # Improved count: strip HTML tags roughly
    text = re.sub(r"<[^>]+>", "", text)  # Remove tags
    words = [w for w in re.split(r"\s+", text) if w.strip()]
    return len(words)

def parse_args():
    p = argparse.ArgumentParser(description="Generate blog using section-aware RAG.")
    p.add_argument("--keyword", required=True, help="Primary keyword (Persian).")
    p.add_argument("--index", default=DEFAULT_INDEX_PATH, help="FAISS index path.")
    p.add_argument("--meta", default=DEFAULT_META_PATH, help="Metadata JSONL path.")
    p.add_argument("--out", default="outputs/generated_blog.html", help="Output HTML path.")
    p.add_argument("--env", default=".env", help="Path to .env for OPENAI_API_KEY.")
    p.add_argument("--perfect-html", default=None, help="Optional HTML file as reference.")
    p.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Chat model.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return p.parse_args()

def main():
    args = parse_args()
    env_path = Path(args.env) if args.env else None
    if env_path and env_path.exists():
        load_env(env_path)
    client = get_openai_client()

    meta = load_meta_jsonl(Path(args.meta))
    index = load_faiss_index(Path(args.index))
    out_path = Path(args.out)
    perfect_html_path = Path(args.perfect_html) if args.perfect_html else None

    try:
        generated_path = generate_blog(client=client, index=index, meta=meta,
                                       keyword=args.keyword, out_path=out_path, perfect_html_path=perfect_html_path,
                                       model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
        LOG.info("Generation finished. Output: %s", generated_path)
    except Exception as e:
        LOG.exception("Generation failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()