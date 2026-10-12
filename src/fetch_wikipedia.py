"""
fetch_wikipedia.py
Fetches a small, topically-coherent set of Wikipedia articles and saves
them as plain-text documents for the RAG pipeline to ingest.

Run this with normal internet access (your laptop, Colab, HF Spaces, etc).
This sandbox environment has restricted network access, so this script
was written and structured here but not executed against the live
Wikipedia API. It uses the well-known `wikipedia` PyPI package, which
wraps the MediaWiki API.

Usage:
    python fetch_wikipedia.py
"""

import json
import os
import time

import wikipedia

# Topic set: trains / rail transport in Germany & Europe.
# Chosen because it (a) gives a coherent domain for a recruiter-relevant
# Q&A demo, (b) loosely echoes the original "Deutsche Bahn FAQ" idea
# without needing to scrape a corporate site, and (c) has well-developed,
# fact-dense Wikipedia articles that produce good retrieval test questions.
TOPICS = [
    "Deutsche Bahn",
    "ICE (train)",
    "Rail transport in Germany",
    "High-speed rail",
    "S-Bahn",
    "Regional-Express",
    "Eurail",
    "Trans-Europ Express",
    "Rail transport",
    "InterCity (Germany)",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_articles(topics=TOPICS, output_dir=OUTPUT_DIR):
    wikipedia.set_lang("en")
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            # Pick the first disambiguation option rather than failing
            page = wikipedia.page(e.options[0], auto_suggest=False)
        except wikipedia.exceptions.PageError:
            print(f"  [skip] No page found for '{topic}'")
            continue

        filename = f"{page.title.replace(' ', '_').replace('/', '_')}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.content)

        manifest.append(
            {
                "title": page.title,
                "url": page.url,
                "filename": filename,
                "chars": len(page.content),
            }
        )
        print(f"  [ok] {page.title} -> {filename} ({len(page.content)} chars)")
        time.sleep(0.5)  # be polite to the API

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFetched {len(manifest)} articles to {output_dir}")
    return manifest


if __name__ == "__main__":
    fetch_articles()
