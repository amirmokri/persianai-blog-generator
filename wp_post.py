#!/usr/bin/env python3

from __future__ import annotations
import os
import re
import sys
import json
import base64
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv
    DOTENV = True
except Exception:
    DOTENV = False

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger("wp_post")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_ENV_PATH = ".env"
DEFAULT_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0  # multiplier seconds

def load_env_file(env_path: Optional[Path]):
    if env_path and env_path.exists():
        if DOTENV:
            load_dotenv(dotenv_path=str(env_path), override=False)
            LOG.info("Loaded environment from %s", env_path)
        else:
            LOG.info("Parsing .env (simple) from %s", env_path)
            with env_path.open(encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def get_wp_config() -> Tuple[str, str, str, str, bool]:
    """
    Return (WP_API_BASE, WP_USERNAME, WP_APP_PASSWORD, WP_DEFAULT_STATUS, WP_VERIFY_SSL)
    Raises RuntimeError if required vars missing.
    """
    wp_api_base = os.environ.get("WP_API_BASE") or ((os.environ.get("WP_BASE_URL") or "").rstrip("/") + "/wp-json/wp/v2")
    wp_username = os.environ.get("WP_USERNAME", "")
    wp_app_password = (os.environ.get("WP_APP_PASSWORD", "") or "").replace(" ", "")
    wp_default_status = os.environ.get("WP_DEFAULT_STATUS", "draft")
    wp_verify_ssl = os.environ.get("WP_VERIFY_SSL", "true").lower() not in ("0", "false", "no")

    if not wp_api_base or not wp_username or not wp_app_password:
        raise RuntimeError("Missing WP_API_BASE / WP_USERNAME / WP_APP_PASSWORD in environment (.env).")

    return wp_api_base, wp_username, wp_app_password, wp_default_status, wp_verify_ssl

def _auth_header(username: str, app_password: str) -> str:
    token = base64.b64encode(f"{username}:{app_password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"

def _wp_headers(primary: bool, api_base: str, username: str, app_password: str) -> dict:
    ua = os.environ.get("HTTP_USER_AGENT") or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        if primary else
        "curl/8.0"
    )
    referer = os.environ.get("WP_BASE_URL") or (api_base.split("/wp-json")[0] if "/wp-json" in api_base else api_base)
    return {
        "Authorization": _auth_header(username, app_password),
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": ua,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": referer,
    }

def create_draft_post(api_base: str, username: str, app_password: str, title: str, slug: str,
                      html: str, excerpt: str, verify_ssl: bool, default_status: str) -> dict:
    """
    Create a post via WP REST API. Tries browser-like headers first, falls back to minimal UA + _locale=user.
    """
    url = api_base.rstrip("/") + "/posts"
    payload = {
        "status": default_status,
        "title": title,
        "slug": slug,
        "content": html,
        "excerpt": excerpt or "",
        "comment_status": "closed",
        "ping_status": "closed",
    }

    # Attempt 1: primary headers
    try:
        headers = _wp_headers(True, api_base, username, app_password)
        LOG.info("Posting draft to %s (primary headers)", url)
        r = requests.post(url, headers=headers, json=payload, verify=verify_ssl, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        LOG.warning("Network error when posting (attempt 1): %s", e)
        raise

    if r.status_code in (200, 201):
        LOG.info("Post created (status %s)", r.status_code)
        return r.json()

    # If blocked (401/403), retry with minimal UA + locale
    if r.status_code in (401, 403):
        LOG.warning("Primary request returned %s. Retrying with fallback headers.", r.status_code)
        payload2 = dict(payload)
        payload2["_locale"] = "user"
        headers2 = _wp_headers(False, api_base, username, app_password)
        r2 = requests.post(url, headers=headers2, json=payload2, verify=verify_ssl, timeout=DEFAULT_TIMEOUT)
        if r2.status_code in (200, 201):
            LOG.info("Post created on fallback (status %s)", r2.status_code)
            return r2.json()
        LOG.error("WP API error %s: %s", r2.status_code, r2.text)
        raise RuntimeError(f"WP error {r2.status_code}: {r2.text}")

    LOG.error("WP API error %s: %s", r.status_code, r.text)
    raise RuntimeError(f"WP error {r.status_code}: {r.text}")

def normalize_persian_text(text: str) -> str:
    """
    Normalize Persian text (same as generate_blog.py).
    """
    text = text.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")
    text = re.sub(r"\s*،\s*", " ، ", text)
    text = re.sub(r"\s*,\s*", " , ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def extract_title_and_excerpt_from_html(html: str, fallback_prefix_words: int = 8) -> Tuple[str, str]:
    """
    Extract <h1> title and create excerpt (~160 chars of text content).
    Handles Persian text and ensures proper encoding.
    """
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = normalize_persian_text(h1.get_text(strip=True))
    else:
        # Fallback to first words of body text
        body_text = soup.get_text(separator=" ", strip=True)
        body_text = normalize_persian_text(body_text)
        words = [w for w in body_text.split() if w.strip()]
        title = " ".join(words[:fallback_prefix_words]) if words else "پست جدید"

    # Excerpt: first 160-200 chars without tags, avoid breaking words
    plain = normalize_persian_text(soup.get_text(separator=" ", strip=True))
    excerpt = plain[:180]
    # Trim to last complete word
    if len(excerpt) == 180 and excerpt[-1] not in (" ", "،", ".", "؟"):
        last_space = excerpt.rfind(" ")
        if last_space > 100:
            excerpt = excerpt[:last_space]
    return title, excerpt

def make_slug(title: str) -> str:
    """
    Produce a URL-safe slug from title, handling Persian text.
    """
    title = normalize_persian_text(title)
    return quote_plus(title.strip())

def post_html_file_to_wp(html_path: Path, env_path: Optional[Path], title: Optional[str], slug: Optional[str], verify_ssl_override: Optional[bool] = None):
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # Load env
    if env_path:
        load_env_file(env_path)

    wp_api_base, wp_username, wp_app_password, wp_default_status, wp_verify_ssl = get_wp_config()

    # Override verify if requested
    if verify_ssl_override is not None:
        wp_verify_ssl = verify_ssl_override

    # Read HTML
    html = html_path.read_text(encoding="utf-8")

    # Extract title/excerpt
    if not title:
        title, excerpt = extract_title_and_excerpt_from_html(html)
    else:
        title = normalize_persian_text(title)
        _, excerpt = extract_title_and_excerpt_from_html(html)

    if not slug:
        slug = make_slug(title)

    # Perform network attempts with backoff
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = create_draft_post(api_base=wp_api_base, username=wp_username, app_password=wp_app_password,
                                     title=title, slug=slug, html=html, excerpt=excerpt,
                                     verify_ssl=wp_verify_ssl, default_status=wp_default_status)
            LOG.info("WP draft created: id=%s, link=%s", resp.get("id"), resp.get("link"))
            # Save response to file
            resp_path = html_path.with_name(html_path.stem + "_wp_response.json")
            resp_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            LOG.info("WP response saved to %s", resp_path)
            return resp
        except Exception as e:
            last_exc = e
            LOG.warning("Attempt %d to post failed: %s", attempt, e)
            if attempt < RETRY_ATTEMPTS:
                backoff = RETRY_BACKOFF * attempt
                LOG.info("Retrying in %.1f seconds...", backoff)
                time.sleep(backoff)
            else:
                LOG.error("All attempts failed. Last error: %s", last_exc)
                raise

def parse_args():
    p = argparse.ArgumentParser(description="Post generated HTML blog (from generate_blog.py) to WordPress as draft.")
    p.add_argument("--html", "-i", required=True, help="Path to generated HTML file (e.g., outputs/generated_blog.html).")
    p.add_argument("--env", "-e", default=DEFAULT_ENV_PATH, help="Path to .env file (optional).")
    p.add_argument("--title", "-t", default=None, help="Optional title to override extracted H1.")
    p.add_argument("--slug", "-s", default=None, help="Optional slug to override auto-generated slug.")
    p.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification (not recommended).")
    return p.parse_args()

def main():
    args = parse_args()
    env_path = Path(args.env) if args.env else None
    html_path = Path(args.html)

    try:
        resp = post_html_file_to_wp(html_path=html_path, env_path=env_path, title=args.title, slug=args.slug,
                                    verify_ssl_override=(not args.no_ssl_verify if args.no_ssl_verify else None))
        print(json.dumps(resp, ensure_ascii=False, indent=2))
    except Exception as e:
        LOG.exception("Failed to post to WordPress: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()