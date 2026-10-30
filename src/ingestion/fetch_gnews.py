"""
Bulk headline collection through Google News RSS search.

The regular RSS feeds (config/sources.yaml) only return the latest 20-50
items per outlet, so building a large dataset that way takes weeks.
Google News RSS search returns up to ~100 real, recent headlines per query,
so querying "site:<outlet> <topic>" for each outlet/topic pair collects
thousands of headlines from the same outlets in a few minutes.

Rows are stored in the same headlines table as normal ingestion, so every
later pipeline stage (NLP, sentiment, features, bias, clustering, report)
processes them unchanged.
"""
import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import feedparser
import yaml

from src.utils.db import CONFIG_DIR, get_connection

GNEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
USER_AGENT = "Mozilla/5.0 (headline-bias-detector research project)"


def load_gnews_config():
    with open(os.path.join(CONFIG_DIR, "gnews_sources.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_title(title, publisher):
    """Google News appends ' - Publisher' to every title; remove it."""
    title = (title or "").strip()
    if publisher and title.endswith(" - " + publisher):
        title = title[: -len(" - " + publisher)]
    elif " - " in title:
        title = title.rsplit(" - ", 1)[0]
    return title.strip()


def publisher_matches(entry, domain):
    """Keep only items actually published by the outlet we searched for."""
    src = entry.get("source") or {}
    href = src.get("href", "") if hasattr(src, "get") else ""
    if not href:
        return True  # no publisher info: trust the site: filter
    host = urlparse(href).netloc.lower()
    return host == domain or host.endswith("." + domain)


def fetch_query(outlet, topic):
    query = f"site:{outlet['domain']} {topic}".strip()
    feed = feedparser.parse(GNEWS_URL.format(query=quote_plus(query)), agent=USER_AGENT)

    status = getattr(feed, "status", None)
    if status and status >= 400:
        raise RuntimeError(f"HTTP {status}")

    headlines = []
    for entry in feed.entries:
        if not publisher_matches(entry, outlet["domain"]):
            continue
        publisher = (entry.get("source") or {}).get("title", "")
        text = clean_title(entry.get("title", ""), publisher)
        if len(text.split()) < 4:  # skip fragments like "Live updates"
            continue
        headlines.append({
            "headline_id": str(uuid.uuid4()),
            "raw_text": text,
            "source_name": outlet["name"],
            "source_url": entry.get("link"),
            "source_known_lean": outlet["lean"],
            "topic_category": topic,
            "published_at": entry.get("published"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
    return headlines


def insert_many(headlines):
    """Insert a batch in one transaction; duplicates (same text) are ignored."""
    conn = get_connection()
    cur = conn.cursor()
    before = conn.total_changes
    cur.executemany("""
        INSERT OR IGNORE INTO headlines (
            headline_id, raw_text, source_name, source_url, source_known_lean,
            topic_category, published_at, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [(h["headline_id"], h["raw_text"], h["source_name"], h["source_url"],
           h["source_known_lean"], h["topic_category"], h["published_at"],
           h["fetched_at"]) for h in headlines])
    conn.commit()
    added = conn.total_changes - before
    conn.close()
    return added


def count_headlines():
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM headlines").fetchone()[0]
    conn.close()
    return n


def run_gnews_ingestion(target=5000, delay=1.0):
    """Query outlet x topic pairs until the database holds `target` headlines."""
    cfg = load_gnews_config()
    outlets, topics = cfg["outlets"], cfg["topics"]

    total = count_headlines()
    print(f"Headlines already in database: {total}  (target: {target})")
    if total >= target:
        print("Target already reached.")
        return total

    failures = 0
    # Topic-major order so every outlet grows evenly if we stop early.
    for topic in topics:
        for outlet in outlets:
            try:
                headlines = fetch_query(outlet, topic)
                added = insert_many(headlines)
                total += added
                failures = 0 if headlines else failures + 1
                print(f"  {outlet['name']:<16} | {topic:<14} | fetched {len(headlines):>3} | new {added:>3} | total {total}")
            except Exception as e:
                failures += 1
                print(f"  {outlet['name']:<16} | {topic:<14} | ERROR: {e}")
            if failures >= 15:
                print("\n15 empty/failed queries in a row, stopping.")
                print("Check the internet connection, or wait 10 minutes (Google may be rate limiting) and run again.")
                return total
            if total >= target:
                print(f"\nTarget of {target} reached.")
                return total
            time.sleep(delay)  # be polite to Google News, avoids rate limiting

    print(f"\nAll queries done. Total headlines: {total}")
    if total < target:
        print("Below target: run the collect command again later (news changes daily).")
    return total
