"""Prototype image scraper — for early-stage data collection ONLY.

Scrapes ice cream product images from DuckDuckGo image search
(no API key required) and saves them organised by category.
Intended as a stand-in before real vending machine photos arrive.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Search queries for each category
CATEGORY_QUERIES: dict[str, str] = {
    "magnum_classic":        "magnum classic ice cream bar product",
    "magnum_almond":         "magnum almond ice cream bar product",
    "magnum_white":          "magnum white chocolate ice cream",
    "cornetto_vanilla":      "cornetto vanilla ice cream cone",
    "cornetto_chocolate":    "cornetto chocolate ice cream cone",
    "cornetto_strawberry":   "cornetto strawberry ice cream cone",
    "popsicle_fruit":        "fruit popsicle ice lolly product",
    "popsicle_chocolate":    "chocolate popsicle ice cream bar",
    "sandwich_ice_cream":    "ice cream sandwich product packaged",
    "cup_ice_cream":         "ice cream cup pot product",
}


def _ddg_image_urls(query: str, max_results: int = 20) -> list[str]:
    """Fetch image URLs via DuckDuckGo image search HTML (no API key)."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/?q={encoded}&iax=images&ia=images"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("DDG request failed for '%s': %s", query, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    urls: list[str] = []
    for tag in soup.find_all("img"):
        src = tag.get("src", "")
        if src.startswith("http") and not "duckduckgo.com" in src:
            urls.append(src)
        if len(urls) >= max_results:
            break
    return urls


def _download_image(url: str, dest: Path, session: requests.Session) -> bool:
    """Download a single image. Returns True on success."""
    try:
        resp = session.get(url, headers=_HEADERS, timeout=8, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        log.debug("Failed to download %s: %s", url, exc)
        return False


def scrape_images(
    output_dir: str | Path,
    categories: list[str] | None = None,
    max_per_category: int = 20,
    delay_between_requests: float = 1.5,
) -> dict[str, int]:
    """Scrape product images and save them organised by category.

    Args:
        output_dir: Root directory; images saved to <output_dir>/<category>/.
        categories: Subset of categories to scrape; None = all.
        max_per_category: Maximum images to download per category.
        delay_between_requests: Sleep time between HTTP calls (rate limiting).

    Returns:
        Dict mapping category → number of images successfully downloaded.
    """
    output_dir = Path(output_dir)
    stats: dict[str, int] = {}
    session = requests.Session()

    targets = {k: v for k, v in CATEGORY_QUERIES.items()
               if categories is None or k in categories}

    for category, query in targets.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        log.info("Scraping category '%s' (query: '%s')", category, query)

        urls = _ddg_image_urls(query, max_results=max_per_category * 2)
        saved = 0

        for i, url in enumerate(urls):
            if saved >= max_per_category:
                break
            dest = cat_dir / f"{category}_{saved:04d}.jpg"
            if dest.exists():
                saved += 1
                continue
            if _download_image(url, dest, session):
                saved += 1
                log.debug("  [%d/%d] saved %s", saved, max_per_category, dest.name)
            time.sleep(delay_between_requests)

        stats[category] = saved
        log.info("  → %d images saved for '%s'", saved, category)

    log.info("Scraping complete: %s", stats)
    return stats


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--output-dir", default="data/raw/scraped", show_default=True)
    @click.option("--max-per-category", default=20, show_default=True)
    @click.option("--delay", default=1.5, show_default=True)
    @click.option("--categories", default=None, help="Comma-separated list; default=all")
    def cli(output_dir, max_per_category, delay, categories):
        """Scrape prototype ice cream images."""
        cats = categories.split(",") if categories else None
        stats = scrape_images(output_dir, cats, max_per_category, delay)
        print(json.dumps(stats, indent=2))

    cli()
