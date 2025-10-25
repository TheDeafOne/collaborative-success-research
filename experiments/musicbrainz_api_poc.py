#!/usr/bin/env python3
"""
MusicBrainz API "kitchen-sink" demo for Artist + Work (fixed)

- Browse recordings WITHOUT 'releases' in inc (invalid for browse)
- Then lookup each recording by MBID WITH 'releases' to get release info

Run:
  python mb_kitchensink_demo.py
"""

from __future__ import annotations
import sys, time, requests
from typing import Any, Dict, Iterable, List

BASE_URL = "https://musicbrainz.org/ws/2"
HEADERS = {"User-Agent": "MB-KitchenSink-Demo/1.1 (contact: your-email@example.com)"}
REQUEST_DELAY_SECONDS = 1.1

DEFAULT_ARTISTS = ["The Beatles", "Beyoncé", "Radiohead", "Miles Davis"]


def throttled():
    time.sleep(REQUEST_DELAY_SECONDS)


def mb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}/{path}"
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


# ---- Search ----
def search_artist_by_name(name: str, limit: int = 1) -> List[Dict[str, Any]]:
    data = mb_get(
        "artist", {"query": f'artist:"{name}"', "fmt": "json", "limit": limit}
    )
    return data.get("artists", []) or []


def search_work_by_title(title: str, limit: int = 1) -> List[Dict[str, Any]]:
    data = mb_get("work", {"query": f'work:"{title}"', "fmt": "json", "limit": limit})
    return data.get("works", []) or []


# ---- Lookups ----
def lookup_artist(
    mbid: str,
    inc: Iterable[str] = (
        "aliases",
        "tags",
        "ratings",
        "genres",
        "annotation",
        "area-rels",
        "artist-rels",
        "event-rels",
        "instrument-rels",
        "label-rels",
        "place-rels",
        "recording-rels",
        "release-rels",
        "release-group-rels",
        "series-rels",
        "url-rels",
        "work-rels",
    ),
) -> Dict[str, Any]:
    return mb_get(f"artist/{mbid}", {"fmt": "json", "inc": "+".join(inc)})


def lookup_work(
    mbid: str,
    inc: Iterable[str] = (
        "aliases",
        "tags",
        "ratings",
        "genres",
        "annotation",
        "artist-rels",
        "recording-rels",
        "work-rels",
        "url-rels",
    ),
) -> Dict[str, Any]:
    return mb_get(f"work/{mbid}", {"fmt": "json", "inc": "+".join(inc)})


def lookup_recording(
    mbid: str,
    inc: Iterable[str] = (
        # NOW it's valid to ask for releases here
        "releases",
        "isrcs",
        "artist-credits",
        "tags",
        "ratings",
        "genres",
        "annotation",
        "work-rels",
        "url-rels",
    ),
) -> Dict[str, Any]:
    return mb_get(f"recording/{mbid}", {"fmt": "json", "inc": "+".join(inc)})


# ---- Browse ----
def browse_release_groups_for_artist(
    artist_mbid: str, limit: int = 5
) -> Dict[str, Any]:
    return mb_get(
        "release-group",
        {"fmt": "json", "artist": artist_mbid, "limit": limit, "offset": 0},
    )


def browse_releases_for_artist(artist_mbid: str, limit: int = 5) -> Dict[str, Any]:
    return mb_get(
        "release",
        {
            "fmt": "json",
            "artist": artist_mbid,
            "limit": limit,
            "offset": 0,
            "inc": "labels+release-groups",
        },
    )


def browse_recordings_for_artist(artist_mbid: str, limit: int = 5) -> Dict[str, Any]:
    # IMPORTANT: don't include 'releases' here; it is invalid for recording browse
    # We'll fetch releases via a follow-up lookup per recording.
    return mb_get(
        "recording",
        {
            "fmt": "json",
            "artist": artist_mbid,
            "limit": limit,
            "offset": 0,
            "inc": "artist-credits",
        },
    )


def browse_works_for_artist(artist_mbid: str, limit: int = 5) -> Dict[str, Any]:
    return mb_get(
        "work",
        {
            "fmt": "json",
            "artist": artist_mbid,
            "limit": limit,
            "offset": 0,
            "inc": "aliases",
        },
    )


# ---- Pretty helpers ----
def kv(k: str, v: Any) -> str:
    if v in (None, "", [], {}):
        return ""
    if isinstance(v, list):
        return (
            f"- {k}: {', '.join(str(x) for x in v[:10])}{' …' if len(v) > 10 else ''}"
        )
    return f"- {k}: {v}"


def summarize_relations(rels: List[Dict[str, Any]], n: int = 8) -> List[str]:
    out = []
    for r in rels[:n]:
        target = (
            r.get("artist")
            or r.get("work")
            or r.get("label")
            or r.get("recording")
            or r.get("release-group")
            or r.get("release")
            or r.get("url")
            or {}
        )
        name = (
            target.get("name")
            or target.get("title")
            or target.get("resource")
            or "(unnamed)"
        )
        out.append(f"{r.get('type', 'relation')} → {name}")
    if len(rels) > n:
        out.append(f"… and {len(rels) - n} more")
    return out


def print_artist_summary(artist: Dict[str, Any]) -> None:
    print(f"\nARTIST LOOKUP: {artist.get('name')} [{artist.get('id')}]")
    core = [
        kv("sort-name", artist.get("sort-name")),
        kv("type", artist.get("type")),
        kv("disambiguation", artist.get("disambiguation")),
        kv("country", artist.get("country")),
        kv("area", (artist.get("area") or {}).get("name")),
        kv("begin-area", (artist.get("begin-area") or {}).get("name")),
        kv("end-area", (artist.get("end-area") or {}).get("name")),
        kv(
            "life-span",
            f"{(artist.get('life-span') or {}).get('begin', '')} — {(artist.get('life-span') or {}).get('end', '')} (ended={(artist.get('life-span') or {}).get('ended')})",
        ),
        kv("ipis", artist.get("ipis")),
        kv("isnis", artist.get("isnis")),
        kv("aliases", [a.get("name") for a in (artist.get("aliases") or [])]),
        kv("tags", [t.get("name") for t in (artist.get("tags") or [])]),
        kv("genres", [g.get("name") for g in (artist.get("genres") or [])]),
        kv("rating", (artist.get("rating") or {}).get("value")),
        kv("rating count", (artist.get("rating") or {}).get("count")),
    ]
    print("\n".join([c for c in core if c]))
    rels = artist.get("relations") or []
    if rels:
        print("\nRelations (sample):")
        print("\n".join(summarize_relations(rels, 12)))


def print_browse_samples(artist_mbid: str) -> None:
    print("\n--- BROWSE SAMPLES ---")

    throttled()
    rgs = browse_release_groups_for_artist(artist_mbid)
    rgs_list = rgs.get("release-groups") or []
    if rgs_list:
        print("\nRelease Groups:")
        for rg in rgs_list:
            print(
                f"- {rg.get('title')} [{rg.get('id')}] — primary-type: {rg.get('primary-type')}"
            )

    throttled()
    rels = browse_releases_for_artist(artist_mbid)
    rels_list = rels.get("releases") or []
    if rels_list:
        print("\nReleases:")
        for r in rels_list:
            labels = [
                li.get("label", {}).get("name")
                for li in (r.get("label-info") or [])
                if li.get("label")
            ]
            print(
                f"- {r.get('title')} [{r.get('id')}] — date: {r.get('date')} — status: {r.get('status')} — labels: {', '.join([x for x in labels if x])}"
            )

    throttled()
    recs = browse_recordings_for_artist(artist_mbid)
    recs_list = recs.get("recordings") or []
    if recs_list:
        print("\nRecordings (with follow-up release lookups):")
        for rec in recs_list:
            title = rec.get("title")
            rid = rec.get("id")
            length_ms = rec.get("length")
            seconds = int(length_ms / 1000) if isinstance(length_ms, int) else None
            credited = ", ".join(
                [
                    c.get("name")
                    for c in (rec.get("artist-credit") or [])
                    if c.get("name")
                ]
            )
            print(f"- {title} [{rid}] — {seconds or '?'}s — credited: {credited}")

            # follow-up: lookup this recording to fetch releases
            throttled()
            try:
                rfull = lookup_recording(rid)
                rels = rfull.get("releases") or []
                if rels:
                    sample = ", ".join(
                        [rel.get("title", "(untitled)") for rel in rels[:3]]
                    )
                    print(f"    releases: {sample}{' …' if len(rels) > 3 else ''}")
            except requests.HTTPError as e:
                print(f"    (recording lookup failed: {e})")

    throttled()
    works = browse_works_for_artist(artist_mbid)
    works_list = works.get("works") or []
    if works_list:
        print("\nWorks:")
        for w in works_list:
            print(f"- {w.get('title')} [{w.get('id')}] — type: {w.get('type')}")


def print_work_details(work_json: Dict[str, Any]) -> None:
    print(f"\nWORK LOOKUP: {work_json.get('title')} [{work_json.get('id')}]")
    fields = [
        kv("type", work_json.get("type")),
        kv("disambiguation", work_json.get("disambiguation")),
        kv("language", work_json.get("language")),
        kv("languages", work_json.get("languages")),
        kv("iswcs", work_json.get("iswcs")),
        kv("aliases", [a.get("name") for a in (work_json.get("aliases") or [])]),
        kv("tags", [t.get("name") for t in (work_json.get("tags") or [])]),
        kv("genres", [g.get("name") for g in (work_json.get("genres") or [])]),
        kv("rating", (work_json.get("rating") or {}).get("value")),
        kv("rating count", (work_json.get("rating") or {}).get("count")),
    ]
    print("\n".join([f for f in fields if f]))
    rels = work_json.get("relations") or []
    if rels:
        print("Relations (sample):")
        print("\n".join(summarize_relations(rels, 12)))


# ---- Orchestration ----
def demo_for_artist_name(name: str) -> None:
    print(f"\n### SEARCH ARTIST: {name}")
    res = search_artist_by_name(name, limit=1)
    if not res:
        print("No artist results.")
        return
    a0 = res[0]
    print(f"Best match: {a0.get('name')} [{a0.get('id')}]  score={a0.get('score')}")

    throttled()
    full = lookup_artist(a0["id"])
    print_artist_summary(full)
    print_browse_samples(a0["id"])

    # Pull a couple works in detail
    throttled()
    w = browse_works_for_artist(a0["id"], limit=2)
    for wk in w.get("works") or []:
        throttled()
        wfull = lookup_work(wk["id"])
        print_work_details(wfull)


def demo_for_work_title(title: str) -> None:
    print(f"\n### SEARCH WORK: {title}")
    ws = search_work_by_title(title, limit=1)
    if not ws:
        print("No work results.")
        return
    w0 = ws[0]
    print(f"Best match: {w0.get('title')} [{w0.get('id')}] type={w0.get('type')}")
    throttled()
    wfull = lookup_work(w0["id"])
    print_work_details(wfull)


def main(argv: List[str]) -> None:
    if not argv:
        for name in DEFAULT_ARTISTS:
            demo_for_artist_name(name)
        demo_for_work_title("So What")
        return
    if argv[0] == "--work":
        demo_for_work_title(" ".join(argv[1:]).strip() or "Hallelujah")
    else:
        demo_for_artist_name(" ".join(argv).strip())


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except requests.HTTPError as e:
        print("\nHTTP error:", e)
        try:
            print("Response:", e.response.json())
        except Exception:
            print("Response:", getattr(e, "response", None) and e.response.text)
    except requests.RequestException as e:
        print("\nNetwork error:", e)
