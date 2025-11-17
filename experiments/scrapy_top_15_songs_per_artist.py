#!/usr/bin/env python3
# harvest_top_artists_tracks_env.py

import argparse
import csv
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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

# ------------------------------------------------------------------------------
# CSV IO helpers (NEW)
# ------------------------------------------------------------------------------

def _norm_name(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None

def read_processed_artists_from_tracks_csv(path: str) -> Tuple[set, set]:
    """
    Return (processed_mbids, processed_names_norm) from an existing tracks CSV.
    We'll consider an artist "done" if either their MBID appears OR their
    normalized name appears in the file.
    """
    mbids = set()
    names = set()
    if not os.path.exists(path):
        return mbids, names

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mbid = (row.get("artist_mbid") or "").strip()
            name = _norm_name(row.get("artist_name"))
            if mbid:
                mbids.add(mbid)
            if name:
                names.add(name)
    return mbids, names


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
# Last.fm: top tracks per artist (LIMIT 15)
# ------------------------------------------------------------------------------

def fetch_top_tracks_for_artist_lastfm(api_key: str, artist_name: Optional[str] = None,
                                       artist_mbid: Optional[str] = None, limit: int = 15):
    """Fetch top tracks for a given artist via Last.fm (artist.getTopTracks)."""
    if not artist_name and not artist_mbid:
        raise ValueError("fetch_top_tracks_for_artist_lastfm requires artist_name or artist_mbid")

    params = {
        "method": "artist.getTopTracks",
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    # Prefer MBID if we have it (more robust)
    if artist_mbid:
        params["mbid"] = artist_mbid
    else:
        params["artist"] = artist_name

    data = requests_get_json(LASTFM_BASE, params=params)
    tracks = data.get("toptracks", {}).get("track", [])
    out = []
    for idx, t in enumerate(tracks, start=1):
        # Last.fm sometimes returns a single dict instead of list when limit=1; normalize
        name = t.get("name") if isinstance(t, dict) else None
        if not name:
            continue
        playcount = t.get("playcount")
        listeners = t.get("listeners")
        out.append({
            "track_rank": idx,
            "track_title": name.strip(),
            "lastfm_listeners": int(listeners) if listeners is not None else None,
            "lastfm_playcount": int(playcount) if playcount is not None else None,
        })
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


def resolve_recording_mbid_mb(artist_name: str, track_title: str, ua_header: Dict[str, str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Resolve a MusicBrainz recording MBID (and length if available) for an artist + track title.
    Uses a focused recording search. Returns (recording_id, length_ms) or (None, None).
    """
    # Tight recording query; quotes help reduce fuzzy mismatches
    query = f'recording:"{track_title}" AND artist:"{artist_name}"'
    params = {
        "query": query,
        "limit": 1,
        "fmt": "json"
    }
    try:
        data = requests_get_json(f"{MB_BASE}/recording", params=params, headers=ua_header)
    except Exception as e:
        log.warning(f"MB recording search failed for {artist_name} - {track_title}: {e}")
        return None, None
    finally:
        # Respect MB rate limiting recommendations
        sleep_throttle(1.0)

    recs = data.get("recordings", [])
    if not recs:
        return None, None
    rec = recs[0]
    rec_id = rec.get("id")
    length_ms = rec.get("length")
    return rec_id, length_ms

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
def harvest(api_key: str, target_artists: int, ua_header: Dict[str, str], per_artist_limit: int = 15):
    artists_csv = f"artists_top_{target_artists}.csv"
    tracks_csv = f"tracks_top_{target_artists}_top{per_artist_limit}.csv"

    # --- Skip if artists file already exists ---
    if os.path.exists(artists_csv):
        log.info(f"Skipping Last.fm fetch — using existing {artists_csv}")
        artists = []
        with open(artists_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in ("rank", "listeners", "playcount"):
                    if k in row and row[k]:
                        try:
                            row[k] = int(row[k])
                        except Exception:
                            pass
                artists.append(row)
    else:
        log.info(f"Fetching top {target_artists} artists from Last.fm")
        artists = fetch_top_artists_lastfm(api_key, target_count=target_artists)
        log.info("Resolving MBIDs for missing artists...")
        for a in artists:
            if not a.get("mbid"):
                try:
                    a["mbid"] = resolve_artist_mbid_mb(a["name"], ua_header)
                except Exception as e:
                    log.warning(f"MB search failed for {a['name']}: {e}")
        write_csv(artists_csv, ["rank", "name", "mbid", "listeners", "playcount"], artists)

    # --- NEW: figure out which artists are already in the tracks CSV ---
    processed_mbids, processed_names_norm = read_processed_artists_from_tracks_csv(tracks_csv)
    if processed_mbids or processed_names_norm:
        log.info(
            f"Found {len(processed_mbids)} artists by MBID and "
            f"{len(processed_names_norm)} by name already in {tracks_csv}; will skip them."
        )

    # --- Prepare to write tracks (top N per artist) ---
    write_header = not os.path.exists(tracks_csv)
    fieldnames = [
        "artist_rank", "artist_name", "artist_mbid",
        "track_rank", "track_title",
        "lastfm_listeners", "lastfm_playcount",
        "mb_recording_id", "mb_recording_length_ms"
    ]

    with open(tracks_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, artist in enumerate(artists, start=1):
            name = artist.get("name")
            mbid = (artist.get("mbid") or "").strip() or None
            name_norm = _norm_name(name)

            # --- NEW: skip if we've already processed this artist ---
            already_done = (mbid and mbid in processed_mbids) or (name_norm and name_norm in processed_names_norm)
            if already_done:
                log.info(f"[{idx}/{len(artists)}] Skipping {name} — already present in {tracks_csv}")
                continue

            log.info(f"[{idx}/{len(artists)}] Fetching top {per_artist_limit} tracks for {name}")

            try:
                top_tracks = fetch_top_tracks_for_artist_lastfm(
                    api_key=api_key,
                    artist_name=name if not mbid else None,
                    artist_mbid=mbid if mbid else None,
                    limit=per_artist_limit
                )
            except Exception as e:
                log.error(f"Failed to get top tracks for {name}: {e}")
                continue

            if not top_tracks:
                log.info(f"No top tracks returned for {name}")
                continue

            for t in top_tracks:
                rec_id, rec_len = resolve_recording_mbid_mb(name, t["track_title"], ua_header)
                row = {
                    "artist_rank": artist.get("rank"),
                    "artist_name": name,
                    "artist_mbid": mbid,
                    "track_rank": t.get("track_rank"),
                    "track_title": t.get("track_title"),
                    "lastfm_listeners": t.get("lastfm_listeners"),
                    "lastfm_playcount": t.get("lastfm_playcount"),
                    "mb_recording_id": rec_id,
                    "mb_recording_length_ms": rec_len,
                }
                writer.writerow(row)
                f.flush()

            # --- NEW: update in-memory processed sets so repeated names/mbids in the same run are skipped too
            if mbid:
                processed_mbids.add(mbid)
            if name_norm:
                processed_names_norm.add(name_norm)

            log.info(f"  Wrote {len(top_tracks)} rows for {name}")

# ------------------------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        log.error("❌ Missing LASTFM_API_KEY in .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Harvest top artists and their top 15 tracks.")
    parser.add_argument("--target-artists", type=int, default=100,
                        help="Number of top artists to fetch (default 100)")
    parser.add_argument("--mb-app-name", default="YourApp/1.0",
                        help="MusicBrainz User-Agent app name")
    parser.add_argument("--mb-contact", default="you@example.com",
                        help="MusicBrainz contact email or URL")
    parser.add_argument("--per-artist-limit", type=int, default=15,
                        help="Number of top tracks to fetch per artist (default 15)")
    args = parser.parse_args()

    ua_header = mb_user_agent_header(args.mb_app_name, args.mb_contact)
    harvest(api_key, args.target_artists, ua_header, per_artist_limit=args.per_artist_limit)

if __name__ == "__main__":
    main()
