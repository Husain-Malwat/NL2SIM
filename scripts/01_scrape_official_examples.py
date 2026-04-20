#!/usr/bin/env python3
"""
Scrape official MuMax3 examples from mumax.github.io/examples.html.

This script fetches live HTML from the official MuMax3 website, parses the
examples page to extract all code blocks and their metadata, and writes
them as .mx3 files with a JSON scrape log.

Usage:
    python3 01_scrape_official_examples.py

Dependencies:
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import json
import re
import hashlib
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

EXAMPLES_URL = "https://mumax.github.io/examples.html"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "official_examples"
LOG_FILE = OUT_DIR / "_scrape_log.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "mumax3-dataset-builder/1.0 (academic research)",
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_examples_page(url: str) -> str:
    """Fetch the raw HTML from the examples page."""
    print(f"Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"  Got {len(resp.text)} bytes, status {resp.status_code}")
    return resp.text


def parse_examples_from_html(html: str) -> list[dict]:
    """
    Parse the examples page HTML to extract example scripts.

    The page structure is:
        <h2 id="exampleN">Title</h2>
        <p>description text...</p>
        <pre><code>...script code...</code></pre>
        <h3>output</h3>
        ... images ...

    We extract the title, description, and code for each example.
    """
    try:
        from bs4 import BeautifulSoup
        return _parse_with_bs4(html)
    except ImportError:
        print("  beautifulsoup4 not installed, falling back to regex parser")
        return _parse_with_regex(html)


def _parse_with_bs4(html: str) -> list[dict]:
    """Parse using BeautifulSoup for robust HTML handling."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    examples = []

    # Find all h2 headings — each is an example section
    h2_tags = soup.find_all("h2")

    for idx, h2 in enumerate(h2_tags):
        title = h2.get_text(strip=True)
        example_id = h2.get("id", f"example{idx + 1}")

        # Skip non-example headings
        if not title or title.lower().startswith("back to top"):
            continue

        # Collect description paragraphs and code blocks until next h2
        desc_parts = []
        code_parts = []

        # Some h2 tags are wrapped inside <a> tags; start from the right parent
        start_element = h2
        if h2.parent and h2.parent.name == "a":
            start_element = h2.parent

        sibling = start_element.find_next_sibling()

        while sibling and sibling.name != "h2":
            if sibling.name == "pre":
                code_tag = sibling.find("code")
                code_text = code_tag.get_text() if code_tag else sibling.get_text()
                code_parts.append(code_text.strip())
            elif sibling.name == "a" and sibling.find("pre"):
                # Some examples wrap <pre> inside <a id="exampleN">
                pre_tag = sibling.find("pre")
                code_tag = pre_tag.find("code")
                code_text = code_tag.get_text() if code_tag else pre_tag.get_text()
                code_parts.append(code_text.strip())
            elif sibling.name == "p" and not code_parts:
                # Only collect description before first code block
                desc_parts.append(sibling.get_text(strip=True))
            elif sibling.name == "h3":
                # "output" heading — stop collecting
                break
            sibling = sibling.find_next_sibling()

        if not code_parts:
            continue

        script_code = "\n".join(code_parts)
        description = " ".join(desc_parts)

        examples.append({
            "id": example_id,
            "title": title,
            "description": description,
            "script": script_code,
            "index": idx + 1,
        })

    return examples


def _parse_with_regex(html: str) -> list[dict]:
    """Fallback regex parser when BeautifulSoup is not available."""
    examples = []

    # Pattern: <h2 ...>Title</h2> ... <pre><code>...code...</code></pre>
    # Split by h2 tags
    sections = re.split(r'<h2[^>]*>', html)

    for idx, section in enumerate(sections[1:], 1):  # skip before first h2
        # Extract title
        title_match = re.match(r'(.*?)</h2>', section, re.S)
        if not title_match:
            continue
        title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
        if not title or "back to top" in title.lower():
            continue

        # Extract id from preceding tag if available
        id_match = re.search(r'id="([^"]*)"', sections[idx] if idx < len(sections) else "")
        example_id = id_match.group(1) if id_match else f"example{idx}"

        # Extract code blocks: <pre><code>...</code></pre>
        code_blocks = re.findall(r'<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>', section, re.S)

        if not code_blocks:
            # Try without <code> wrapper
            code_blocks = re.findall(r'<pre[^>]*>(.*?)</pre>', section, re.S)

        if not code_blocks:
            continue

        # Clean HTML entities
        script_code = "\n".join(code_blocks)
        script_code = (script_code
                       .replace("&lt;", "<")
                       .replace("&gt;", ">")
                       .replace("&amp;", "&")
                       .replace("&quot;", '"')
                       .replace("&#39;", "'"))
        # Strip remaining HTML tags
        script_code = re.sub(r'<[^>]+>', '', script_code).strip()

        # Extract description (text between </h2> and first <pre>)
        desc_section = section[:section.find("<pre")] if "<pre" in section else ""
        desc_text = re.sub(r'<[^>]+>', '', desc_section).strip()
        # Take first paragraph only
        desc_lines = [l.strip() for l in desc_text.split('\n') if l.strip()]
        description = " ".join(desc_lines[:3])

        examples.append({
            "id": example_id,
            "title": title,
            "description": description,
            "script": script_code,
            "index": idx,
        })

    return examples


def write_examples(examples: list[dict]) -> list[dict]:
    """Write extracted examples to .mx3 files and return metadata log."""
    log_entries = []

    for ex in examples:
        # Sanitize filename
        safe_name = re.sub(r'[^\w\s-]', '', ex["title"]).strip()
        safe_name = re.sub(r'\s+', '_', safe_name).lower()
        filename = f"example{ex['index']:02d}_{safe_name}.mx3"

        # Add header comment
        header = f"// {ex['title']}\n// Source: {EXAMPLES_URL}#{ex['id']}\n\n"
        content = header + ex["script"]

        out_path = OUT_DIR / filename
        out_path.write_text(content)

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        entry = {
            "source": "official_examples",
            "source_url": f"{EXAMPLES_URL}#{ex['id']}",
            "title": ex["title"],
            "description": ex["description"][:200],
            "local_file": str(out_path.relative_to(Path(__file__).parent.parent)),
            "filename": filename,
            "content_hash": content_hash,
            "lines": len(content.splitlines()),
            "bytes": len(content.encode()),
            "index": ex["index"],
        }
        log_entries.append(entry)
        print(f"  [{ex['index']:2d}] {filename} ({entry['lines']} lines)")

    return log_entries


def main():
    print("=" * 60)
    print("Official MuMax3 Examples Scraper")
    print("=" * 60)

    # Fetch live HTML
    html = fetch_examples_page(EXAMPLES_URL)

    # Parse examples
    examples = parse_examples_from_html(html)
    print(f"\nExtracted {len(examples)} examples from HTML\n")

    if not examples:
        print("ERROR: No examples found. The page structure may have changed.")
        print("Falling back to seed data (01_official_examples_seed.py)")
        return

    # Write to disk
    log_entries = write_examples(examples)

    # Write scrape log
    log_data = {
        "scrape_url": EXAMPLES_URL,
        "scrape_time": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(log_entries),
        "entries": log_entries,
    }
    LOG_FILE.write_text(json.dumps(log_data, indent=2))
    print(f"\nScrape log: {LOG_FILE}")
    print(f"Total: {len(log_entries)} examples written to {OUT_DIR}")


if __name__ == "__main__":
    main()
