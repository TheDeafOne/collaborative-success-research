#!/usr/bin/env python3
# harvest_top_artists_tracks_env.py

import argparse
import csv
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Set

import requests
from dotenv import load_dotenv

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
MB_BASE = "https://musicbrainz.org/ws/2"

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("harvest.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def sleep_throttle(seconds: float):
    time.sleep(seconds)


def requests_get_json(url: str, params=None, headers=None, timeout=30, retries=3, backoff=1.5):
    """GET JSON with retry/backoff"""
    attempt = 0
    while True:
        try:
            r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            attempt += 1
            log.warning(f"Request error ({url}): {e}, attempt {attempt}")
            if attempt >= retries:
                log.error(f"Giving up on {url}")
                raise
            sleep_throttle(backoff * attempt)


# ------------------------------------------------------------------------------
# Last.fm: top artists
# ------------------------------------------------------------------------------

def fetch_top_artists_lastfm(api_key: str, target_count=1000, page_limit=200):
    out = []
    page = 1
    while len(out) < target_count:
        params = {
            "method": "chart.getTopArtists",
            "api_key": api_key,
            "limit": page_limit,
            "page": page,
            "format": "json",
        }
        data = requests_get_json(LASTFM_BASE, params=params)
        artists = data.get("artists", {}).get("artist", [])
        if not artists:
            break

        for idx, a in enumerate(artists, start=1):
            rank = (page - 1) * page_limit + idx
            out.append({
                "name": a.get("name", "").strip(),
                "mbid": a.get("mbid", "").strip(),
                "listeners": int(a.get("listeners", "0")) if a.get("listeners") else None,
                "playcount": int(a.get("playcount", "0")) if a.get("playcount") else None,
                "rank": rank,
            })
            if len(out) >= target_count:
                break

        page += 1
        sleep_throttle(0.5)
    return out


# ------------------------------------------------------------------------------
# MusicBrainz helpers
# ------------------------------------------------------------------------------

def mb_user_agent_header(app_name: str, contact: str):
    return {"User-Agent": f"{app_name} ( {contact} )"}


def resolve_artist_mbid_mb(artist_name: str, ua_header: Dict[str, str]) -> Optional[str]:
    """Resolve artist MBID if missing."""
    params = {"query": f'artist:"{artist_name}"', "limit": 1, "fmt": "json"}
    data = requests_get_json(f"{MB_BASE}/artist", params=params, headers=ua_header)
    sleep_throttle(1.0)
    artists = data.get("artists", [])
    return artists[0].get("id") if artists else None


def get_all_releases_for_artist(artist_mbid: str, ua_header: Dict[str, str]) -> List[Dict]:
    """Fetch all releases for a given artist."""
    releases = []
    limit = 100
    offset = 0
    while True:
        params = {"artist": artist_mbid, "limit": limit, "offset": offset, "fmt": "json"}
        data = requests_get_json(f"{MB_BASE}/release", params=params, headers=ua_header)
        page = data.get("releases", [])
        if not page:
            break
        releases.extend(page)
        offset += limit
        sleep_throttle(1)
    return releases


def get_tracks_for_release(release_mbid: str, ua_header: Dict[str, str]):
    """Fetch a release’s full tracklist (with logging)."""
    log.info(f"Fetching tracks for release {release_mbid}")
    params = {"inc": "recordings", "fmt": "json"}

    try:
        rel = requests_get_json(f"{MB_BASE}/release/{release_mbid}", params=params, headers=ua_header)
    except Exception as e:
        log.error(f"Failed to fetch release {release_mbid}: {e}")
        return None, []

    sleep_throttle(1)
    tracks = []
    for medium in rel.get("media", []):
        for t in medium.get("tracks", []):
            rec = t.get("recording", {}) or {}
            tracks.append({
                "release_id": rel.get("id"),
                "release_title": rel.get("title"),
                "release_date": rel.get("date"),
                "track_title": t.get("title"),
                "recording_id": rec.get("id"),
                "recording_title": rec.get("title"),
                "length_ms": t.get("length") or rec.get("length"),
            })
            # log.info(f"release_title: {rel.get('title')}")
    log.info(f"Found {len(tracks)} tracks in release {release_mbid}")
    return rel, tracks


# ------------------------------------------------------------------------------
# CSV IO
# ------------------------------------------------------------------------------

def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    log.info(f"Wrote {len(rows)} rows → {path}")


# ------------------------------------------------------------------------------
# Main harvesting workflow
# ------------------------------------------------------------------------------

def harvest(api_key: str, target_artists: int, ua_header: Dict[str, str]):
    artists_csv = f"artists_top_{target_artists}.csv"
    tracks_csv = f"tracks_top_{target_artists}.csv"

    # --- Skip if artists file already exists ---
    if os.path.exists(artists_csv):
        log.info(f"Skipping Last.fm fetch — using existing {artists_csv}")
        artists = []
        with open(artists_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                artists.append(row)
    else:
        log.info(f"Fetching top {target_artists} artists from Last.fm")
        artists = fetch_top_artists_lastfm(api_key, target_count=target_artists)
        log.info("Resolving MBIDs for missing artists...")
        for a in artists:
            if not a["mbid"]:
                try:
                    a["mbid"] = resolve_artist_mbid_mb(a["name"], ua_header)
                except Exception as e:
                    log.warning(f"MB search failed for {a['name']}: {e}")
        write_csv(artists_csv, ["rank", "name", "mbid", "listeners", "playcount"], artists)

    # --- Prepare to write tracks incrementally ---
    write_header = not os.path.exists(tracks_csv)
    fieldnames = [
        "artist_name", "artist_mbid", "release_id", "release_title",
        "release_date", "track_title", "recording_id",
        "recording_title", "length_ms"
    ]

    with open(tracks_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # --- Collect album tracklists per artist ---
        for idx, artist in enumerate(artists, start=1):
            mbid = artist.get("mbid")
            name = artist.get("name")
            if not mbid:
                log.warning(f"[{idx}/{len(artists)}] Skipping {name} (no MBID).")
                continue

            log.info(f"[{idx}/{len(artists)}] Collecting releases for {name}")
            try:
                releases = get_all_releases_for_artist(mbid, ua_header)
            except Exception as e:
                log.error(f"Failed to get releases for {name}: {e}")
                continue

            seen: Set[str] = set()
            for rel in releases:
                rid = rel.get("id")
                if not rid or rid in seen:
                    continue
                seen.add(rid)

                _, tracks = get_tracks_for_release(rid, ua_header)
                if not tracks:
                    continue

                for t in tracks:
                    t["artist_name"] = name
                    t["artist_mbid"] = mbid
                    writer.writerow({
                        "artist_name": t.get("artist_name"),
                        "artist_mbid": t.get("artist_mbid"),
                        "release_id": t.get("release_id"),
                        "release_title": t.get("release_title"),
                        "release_date": t.get("release_date"),
                        "track_title": t.get("track_title"),
                        "recording_id": t.get("recording_id"),
                        "recording_title": t.get("recording_title"),
                        "length_ms": t.get("length_ms"),
                    })
                f.flush()  # ensure progress saved to disk
                log.info(f"  Wrote {len(tracks)} tracks from release {rid}")


# ------------------------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        log.error("❌ Missing LASTFM_API_KEY in .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Harvest top artists and album tracklists.")
    parser.add_argument("--target-artists", type=int, default=100, help="Number of top artists to fetch (default 100)")
    parser.add_argument("--mb-app-name", default="YourApp/1.0", help="MusicBrainz User-Agent app name")
    parser.add_argument("--mb-contact", default="you@example.com", help="MusicBrainz contact email or URL")
    args = parser.parse_args()

    ua_header = mb_user_agent_header(args.mb_app_name, args.mb_contact)
    harvest(api_key, args.target_artists, ua_header)


if __name__ == "__main__":
    main()
