"""Scraper for League of Legends patch notes from the official website."""

import re
import time
import json
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tqdm import tqdm

from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.leagueoflegends.com"
PATCH_INDEX_URL = f"{BASE_URL}/en-us/news/tags/patch-notes/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class PatchNote(BaseModel):
    patch_version: str
    title: str
    url: str
    content: str


def discover_patch_urls(max_pages: int = 5) -> list[dict]:
    """
    Discover patch note article URLs from the LoL news page.

    The site uses a content API for loading articles. We try that first,
    then fall back to HTML scraping.
    """
    articles = []

    # Strategy 1: Try the content API (used by the site's JS)
    api_url = f"{BASE_URL}/page-data/en-us/news/game-updates/page-data.json"
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Navigate the Gatsby page-data structure
            nodes = (
                data.get("result", {})
                .get("data", {})
                .get("allContentstackArticles", {})
                .get("nodes", [])
            )
            for node in nodes:
                url_obj = node.get("url", {})
                url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                title = node.get("title", "")
                if "patch" in title.lower() and "notes" in title.lower():
                    version = extract_patch_version(title, url)
                    if version:
                        articles.append({
                            "patch_version": version,
                            "title": title,
                            "url": url if url.startswith("http") else f"{BASE_URL}{url}",
                        })
            if articles:
                logger.info(f"Found {len(articles)} patch notes via content API")
                return articles
    except Exception as e:
        logger.debug(f"Content API approach failed: {e}")

    # Strategy 2: Scrape HTML pages
    logger.info("Falling back to HTML scraping for patch note discovery")
    for page in range(1, max_pages + 1):
        url = PATCH_INDEX_URL if page == 1 else f"{PATCH_INDEX_URL}?page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Look for article links containing "patch" and "notes"
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True).lower()
                if "patch" in href and "notes" in href:
                    title_text = link.get_text(strip=True)
                    full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                    version = extract_patch_version(title_text, href)
                    if version and not any(a["patch_version"] == version for a in articles):
                        articles.append({
                            "patch_version": version,
                            "title": title_text,
                            "url": full_url,
                        })

            time.sleep(1)  # Be polite
        except Exception as e:
            logger.warning(f"Failed to scrape page {page}: {e}")
            break

    # Strategy 3: Generate known URLs for recent patches
    if not articles:
        logger.info("Generating URLs for known recent patch versions")
        articles = generate_known_patch_urls()

    logger.info(f"Discovered {len(articles)} patch note URLs total")
    return articles


def generate_known_patch_urls(seasons: list[int] = None, patches_per_season: int = 24) -> list[dict]:
    """
    Generate URLs for known patch versions.
    LoL patches follow the pattern: patch-{season}-{number}-notes
    Season 14 = 2024, Season 15 = 2025, etc.
    """
    if seasons is None:
        seasons = [14, 15]  # 2024 and 2025

    articles = []
    for season in seasons:
        for patch_num in range(1, patches_per_season + 1):
            version = f"{season}.{patch_num}"
            url = f"{BASE_URL}/en-us/news/game-updates/patch-{season}-{patch_num}-notes/"
            articles.append({
                "patch_version": version,
                "title": f"Patch {version} Notes",
                "url": url,
            })
    return articles


def extract_patch_version(title: str, url: str) -> str | None:
    """Extract patch version like '14.10' from title or URL."""
    # Try URL first: /patch-14-10-notes/
    url_match = re.search(r"patch-(\d+)-(\d+)", url)
    if url_match:
        return f"{url_match.group(1)}.{url_match.group(2)}"

    # Try title: "Patch 14.10 Notes"
    title_match = re.search(r"patch\s+(\d+\.\d+)", title, re.IGNORECASE)
    if title_match:
        return title_match.group(1)

    return None


def scrape_patch_note(url: str) -> str:
    """Scrape the content of a single patch note page."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # The patch notes content is typically inside an article or main content div
    content_selectors = [
        "div[data-testid='article-content']",
        "article",
        "div.style__Content-sc",
        "div.patch-notes-container",
        "#patch-notes-container",
        "main",
    ]

    content_div = None
    for selector in content_selectors:
        content_div = soup.select_one(selector)
        if content_div:
            break

    if not content_div:
        # Fallback: get the largest text block in the page
        content_div = soup.find("body")

    # Clean up: remove scripts, styles, nav elements
    for tag in content_div.find_all(["script", "style", "nav", "footer", "header", "iframe"]):
        tag.decompose()

    # Extract structured text preserving headers
    lines = []
    for element in content_div.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "div"]
    ):
        text = element.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue

        if element.name in ("h1", "h2"):
            lines.append(f"\n## {text}\n")
        elif element.name in ("h3", "h4"):
            lines.append(f"\n### {text}\n")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    content = "\n".join(lines)

    # Deduplicate consecutive identical lines (common in scraped HTML)
    deduped_lines = []
    for line in content.split("\n"):
        if not deduped_lines or line.strip() != deduped_lines[-1].strip():
            deduped_lines.append(line)

    return "\n".join(deduped_lines)


def scrape_all(
    max_patches: int | None = None,
    delay: float = 2.0,
    seasons: list[int] | None = None,
) -> list[PatchNote]:
    """
    Scrape all discoverable patch notes.

    Args:
        max_patches: Maximum number of patches to scrape (None = all).
        delay: Seconds to wait between requests.
        seasons: Which seasons to target (default [14, 15]).

    Returns:
        List of PatchNote objects with scraped content.
    """
    articles = discover_patch_urls()
    if max_patches:
        articles = articles[:max_patches]

    results = []
    for article in tqdm(articles, desc="Scraping patch notes"):
        cache_path = RAW_DATA_DIR / f"patch_{article['patch_version'].replace('.', '_')}.json"

        # Use cached version if available
        if cache_path.exists():
            logger.info(f"Using cached: {article['patch_version']}")
            with open(cache_path) as f:
                cached = json.load(f)
            results.append(PatchNote(**cached))
            continue

        try:
            logger.info(f"Scraping patch {article['patch_version']} from {article['url']}")
            content = scrape_patch_note(article["url"])

            if len(content.strip()) < 100:
                logger.warning(f"Patch {article['patch_version']}: content too short, skipping")
                continue

            patch = PatchNote(
                patch_version=article["patch_version"],
                title=article["title"],
                url=article["url"],
                content=content,
            )

            # Cache to disk
            with open(cache_path, "w") as f:
                json.dump(patch.model_dump(), f, indent=2)

            results.append(patch)
            time.sleep(delay)

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"Patch {article['patch_version']} not found (404), skipping")
            else:
                logger.warning(f"HTTP error for {article['patch_version']}: {e}")
        except Exception as e:
            logger.warning(f"Failed to scrape {article['patch_version']}: {e}")

    logger.info(f"Successfully scraped {len(results)} patch notes")
    return results


if __name__ == "__main__":
    patches = scrape_all(max_patches=5)
    for p in patches:
        print(f"  {p.patch_version}: {p.title} ({len(p.content)} chars)")
