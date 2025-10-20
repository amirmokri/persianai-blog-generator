# utils.py
import os
import openai
import tiktoken
from typing import List

openai.api_key = os.environ.get("OPENAI_API_KEY")

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    return chunks

def create_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    embs = []
    for t in texts:
        resp = openai.embeddings.create(model=model, input=t)
        embs.append(resp.data[0].embedding)
    return embs

def summarize_text(text: str, model: str = "gpt-4o-mini") -> str:
    prompt = f"خلاصه کوتاه و دقیق متن زیر به گونه‌ای که بتواند بخش بعدی را به طور مرتبط ادامه دهد:\n{text}"
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()
