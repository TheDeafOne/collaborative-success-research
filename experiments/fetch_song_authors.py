#!/usr/bin/env python3
# mb_authors_from_top_tracks_resume.py
"""
Incrementally fetch song authors from MusicBrainz for top tracks CSV.

Resumable:
- Writes one JSONL line at a time (flush after every write)
- Skips already-fetched recordings by checking existing output JSONL
"""

import argparse
import csv
import json
import logging
import sys
import time
from typing import Dict, Any, List, Optional, Set, Tuple

import requests

MB_BASE = "https://musicbrainz.org/ws/2"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("mb-authors")

# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def sleep_throttle(seconds: float):
    time.sleep(seconds)

def requests_get_json(url: str, params=None, headers=None, timeout=30, retries=3, backoff=1.5) -> Dict[str, Any]:
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
# MusicBrainz
# ------------------------------------------------------------------------------

AUTHOR_REL_TYPES = {
    "composer", "lyricist", "writer", "co-composer", "librettist", "translator", "arranger"
}

def mb_user_agent_header(app_name: str, contact: str) -> Dict[str, str]:
    return {"User-Agent": f"{app_name} ({contact})"}

def get_recording_with_works(recording_id: str, ua_header: Dict[str, str]) -> Dict[str, Any]:
    params = {"inc": "work-rels+artist-credits+artist-rels", "fmt": "json"}
    data = requests_get_json(f"{MB_BASE}/recording/{recording_id}", params=params, headers=ua_header)
    sleep_throttle(1.0)
    return data

def get_work_with_authors(work_id: str, ua_header: Dict[str, str]) -> Dict[str, Any]:
    params = {"inc": "artist-rels", "fmt": "json"}
    data = requests_get_json(f"{MB_BASE}/work/{work_id}", params=params, headers=ua_header)
    sleep_throttle(1.0)
    return data

def extract_authors_from_work(work_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    authors = []
    for rel in work_obj.get("relations", []) or []:
        if (rel.get("type") or "").lower() in AUTHOR_REL_TYPES:
            artist = rel.get("artist") or {}
            authors.append({
                "name": artist.get("name"),
                "artist_mbid": artist.get("id"),
                "relation_type": rel.get("type"),
                "attributes": rel.get("attributes") or [],
                "target_type": "work",
            })
    return authors

def extract_authors_from_recording(recording_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    authors = []
    for rel in recording_obj.get("relations", []) or []:
        if (rel.get("type") or "").lower() in AUTHOR_REL_TYPES:
            artist = rel.get("artist") or {}
            authors.append({
                "name": artist.get("name"),
                "artist_mbid": artist.get("id"),
                "relation_type": rel.get("type"),
                "attributes": rel.get("attributes") or [],
                "target_type": "recording",
            })
    return authors

# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def read_recording_ids_from_csv(path: str) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = (row.get("mb_recording_id") or "").strip()
            if not rec_id or rec_id in seen:
                continue
            seen.add(rec_id)
            rows.append({
                "recording_id": rec_id,
                "recording_title": row.get("recording_title") or row.get("track_title"),
                "artist_name": row.get("artist_name"),
                "artist_mbid": row.get("artist_mbid"),
            })
    log.info(f"Loaded {len(rows)} unique recording ids from {path}")
    return rows

def load_completed_ids(output_path: str) -> Set[str]:
    """Return set of already written recording_ids from JSONL."""
    completed = set()
    try:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    rid = obj.get("recording_id")
                    if rid:
                        completed.add(rid)
                except json.JSONDecodeError:
                    continue
        log.info(f"Resuming: found {len(completed)} already-processed recordings in {output_path}")
    except FileNotFoundError:
        log.info("No existing output file found — starting fresh")
    return completed

def append_jsonl(output_path: str, obj: Dict[str, Any]):
    """Append one JSON line and flush immediately."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def harvest_authors(input_csv: str, output_jsonl: str, ua_header: Dict[str, str]):
    in_rows = read_recording_ids_from_csv(input_csv)
    done_ids = load_completed_ids(output_jsonl)

    for idx, r in enumerate(in_rows, start=1):
        rec_id = r["recording_id"]
        if rec_id in done_ids:
            log.info(f"[{idx}/{len(in_rows)}] Skipping already done: {rec_id}")
            continue

        log.info(f"[{idx}/{len(in_rows)}] Fetching authors for {rec_id}")

        try:
            rec_obj = get_recording_with_works(rec_id, ua_header)
        except Exception as e:
            log.error(f"Failed recording lookup {rec_id}: {e}")
            continue

        # Linked works
        work_ids = [rel["work"]["id"] for rel in rec_obj.get("relations", []) if rel.get("target-type") == "work" and rel.get("work", {}).get("id")]

        authors = []
        works = []
        if work_ids:
            for wid in work_ids:
                try:
                    work_obj = get_work_with_authors(wid, ua_header)
                    works.append({
                        "work_id": work_obj.get("id"),
                        "work_title": work_obj.get("title"),
                        "type": work_obj.get("type"),
                        "languages": work_obj.get("languages"),
                    })
                    authors.extend(extract_authors_from_work(work_obj))
                except Exception as e:
                    log.warning(f"Work fetch failed {wid}: {e}")
        else:
            authors.extend(extract_authors_from_recording(rec_obj))

        # Deduplicate
        seen = set()
        deduped = []
        for a in authors:
            key = (a.get("artist_mbid"), a.get("relation_type"))
            if key not in seen:
                seen.add(key)
                deduped.append(a)

        result = {
            "recording_id": rec_id,
            "recording_title": r.get("recording_title") or rec_obj.get("title"),
            "artist_name": r.get("artist_name"),
            "artist_mbid": r.get("artist_mbid"),
            "authors": deduped,
            "works": works,
        }

        append_jsonl(output_jsonl, result)
        done_ids.add(rec_id)
        log.info(f"✅ Wrote authors for {rec_id}")

def main():
    ap = argparse.ArgumentParser(description="Fetch song authors from MusicBrainz incrementally.")
    ap.add_argument("--input", required=True, help="Input CSV with mb_recording_id column.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--mb-app-name", default="YourApp/1.0", help="MusicBrainz User-Agent app name.")
    ap.add_argument("--mb-contact", default="you@example.com", help="MusicBrainz contact email or URL.")
    args = ap.parse_args()

    ua_header = mb_user_agent_header(args.mb_app_name, args.mb_contact)
    harvest_authors(args.input, args.output, ua_header)

if __name__ == "__main__":
    main()
