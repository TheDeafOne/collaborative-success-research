#!/usr/bin/env python3
"""
MusicBrainz API proof-of-concept

- Searches for a few artist names using the /ws/2/artist search endpoint
- Looks up each found artist by MBID to fetch extra fields
- Prints a tiny summary (name, MBID, country, disambiguation, a few tags)
- Demonstrates polite rate limiting and proper User-Agent per MusicBrainz docs

Run:
    python musicbrainz_poc.py
"""
from __future__ import annotations

import time
import requests
from typing import List, Dict, Any

BASE_URL = "https://musicbrainz.org/ws/2"
# Per MusicBrainz etiquette, set a meaningful User-Agent: app-name/version (contact)
# Replace the contact info with something appropriate for your project/org.
HEADERS = {
    "User-Agent": "MB-POC-Script/1.0 (contact: example@example.com)"
}

# Keep it tiny—just a few well-known names for the demo.
ARTISTS = [
    "The Beatles",
    "Beyoncé",
    "Radiohead",
    "Miles Davis",
]

# MusicBrainz recommends ~1 request/second on average.
REQUEST_DELAY_SECONDS = 1.1


def mb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper for GET calls with consistent headers and basic error handling."""
    url = f"{BASE_URL}/{path}"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def search_artist_by_name(name: str) -> Dict[str, Any] | None:
    """Return the best matching artist (dict) for a given name using the search endpoint."""
    # Search API does not support 'inc=' parameters. We just get basic match data here.
    # See: /ws/2/artist?query=...&fmt=json
    params = {
        "query": f'artist:"{name}"',
        "limit": 1,
        "fmt": "json",
    }
    data = mb_get("artist", params)
    artists = data.get("artists", [])
    return artists[0] if artists else None


def lookup_artist_by_mbid(mbid: str) -> Dict[str, Any]:
    """Lookup an artist by MBID, including a few useful extras like tags and aliases."""
    # Lookup supports 'inc=' to include extra sub-entities.
    # Example: /ws/2/artist/{mbid}?inc=tags+aliases&fmt=json
    params = {
        "inc": "tags+aliases",
        "fmt": "json",
    }
    return mb_get(f"artist/{mbid}", params)


def summarize_artist(artist_json: Dict[str, Any]) -> str:
    """Make a small, friendly summary line for an artist lookup JSON."""
    name = artist_json.get("name", "Unknown")
    mbid = artist_json.get("id", "N/A")
    country = artist_json.get("country", "—")
    disambig = artist_json.get("disambiguation", "")
    tags = [t.get("name") for t in (artist_json.get("tags") or [])]
    top_tags = ", ".join(tags[:3]) if tags else "—"
    parts = [f"{name}"]
    if disambig:
        parts.append(f"({disambig})")
    parts.append(f"- MBID: {mbid}")
    if country != "—":
        parts.append(f"- Country: {country}")
    parts.append(f"- Tags: {top_tags}")
    return " ".join(parts)


def main(names: List[str]) -> None:
    for i, name in enumerate(names):
        if i:
            time.sleep(REQUEST_DELAY_SECONDS)  # be nice to the API

        print(f"\nSearching for: {name}")
        best = search_artist_by_name(name)
        if not best:
            print("  No results 😕")
            continue

        mbid = best.get("id")
        print(f"  Found MBID: {mbid} — fetching details…")

        time.sleep(REQUEST_DELAY_SECONDS)  # rate-limit between requests
        full = lookup_artist_by_mbid(mbid)
        print("  " + summarize_artist(full))


if __name__ == "__main__":
    main(ARTISTS)
