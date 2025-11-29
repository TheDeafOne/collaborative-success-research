#!/usr/bin/env python3
"""
spotify_artist_early_features.py

Compute early-career, artist-level features from Spotify for the first N years
after an artist's debut release date. No classes. Any function that needs a
token fetches/caches it automatically from env vars.

Env:
  export SPOTIFY_CLIENT_ID=...
  export SPOTIFY_CLIENT_SECRET=...

Usage:
  python spotify_artist_early_features.py --artist "Billie Eilish" --years 3 --market US --include-track-popularity
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import sys
import time
import typing as t
from collections import Counter

import requests

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# ---------------------------
# Token cache (auto)
# ---------------------------
_TOKEN: t.Optional[str] = None
_TOKEN_EXPIRY: float = 0.0


def _ensure_token(timeout: int = 15) -> str:
    """Fetch/refresh and cache a client-credentials token from env."""
    global _TOKEN, _TOKEN_EXPIRY
    if _TOKEN and time.time() < (_TOKEN_EXPIRY - 30):
        return _TOKEN

    cid = os.getenv("SPOTIFY_CLIENT_ID")
    cs = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not cs:
        sys.exit(
            "Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your environment."
        )

    auth = base64.b64encode(f"{cid}:{cs}".encode()).decode()
    r = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {auth}"},
        timeout=timeout,
    )
    if r.status_code != 200:
        sys.exit(f"[auth] Token error {r.status_code}: {r.text}")
    data = r.json()
    _TOKEN = data["access_token"]
    _TOKEN_EXPIRY = time.time() + int(data.get("expires_in", 3600))
    return _TOKEN


def _get(url: str, params: dict | None = None, timeout: int = 15) -> dict:
    token = _ensure_token()

    max_attempts = 4  # raises after this many tries
    base_delay = 0.5  # starting delay in seconds
    max_delay = 43200  # maximum backoff sleep

    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
                timeout=timeout,
            )
        except Exception as exc:
            # treat network failure as retryable
            delay = min(base_delay * (2**attempt), max_delay)
            print("network error or something:", exc)
            time.sleep(delay)
            continue

        # -----------------------
        # 429: respect Retry-After or fall back to exponential backoff
        # -----------------------
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "0"))
            backoff = base_delay * (2**attempt)
            sleep_time = max(retry_after, backoff)
            print(f"backing off for {sleep_time} seconds")
            time.sleep(min(sleep_time, max_delay))
            continue

        # -----------------------
        # 401 on first attempt → refresh and retry
        # -----------------------
        if resp.status_code == 401 and attempt == 0:
            _refresh_token()
            token = _ensure_token()
            continue

        # -----------------------
        # Success
        # -----------------------
        if 200 <= resp.status_code < 300:
            return resp.json()

        # -----------------------
        # Other retryable failures
        # -----------------------
        delay = min(base_delay * (2**attempt), max_delay)
        time.sleep(delay)

    # Out of attempts
    raise Exception(
        f"[GET] {url} failed after {max_attempts} attempts: "
        f"{resp.status_code}: {resp.text}"
    )


def _refresh_token():
    global _TOKEN, _TOKEN_EXPIRY
    _TOKEN = None
    _TOKEN_EXPIRY = 0.0
    _ensure_token()


def get_multiple_artist_success_metrics(artist_ids: list[str]) -> list[dict]:
    """
    Return the current number of followers for an artist, given their Spotify artist ID.

    Uses the /artists/{id} endpoint and the existing _get helper / token machinery.
    Exits with an error message if the artist cannot be fetched.
    """
    url = f"{API}/artists?ids={','.join(artist_ids)}"
    data = _get(url).get("artists")
    artist_popularity_metrics = []
    for artist_data in data:
        if artist_data is None:
            continue
        artist_id = artist_data.get("id")
        followers = artist_data.get("followers", {}).get("total")
        popularity = artist_data.get("popularity", {})
        if followers is None and popularity is None:
            print(f"Could not find follower count for artist id '{artist_id}'.")
            continue
        artist_popularity_metrics.append(
            dict(
                artist_id=artist_id,
                followers=followers if followers else 0,
                popularity=popularity if popularity else popularity,
            )
        )

    return artist_popularity_metrics


# ---------------------------
# Lookups (no token arguments)
# ---------------------------
def search_artist_id(artist_name: str, limit: int = 1) -> str:
    """Resolve an artist ID by name via Search API."""
    data = _get(f"{API}/search", {"q": artist_name, "type": "artist", "limit": limit})
    items = data.get("artists", {}).get("items", [])
    if not items:
        sys.exit(f"No artist found for '{artist_name}'.")
    return items[0]["id"]


def get_artist_albums(
    artist_id: str,
    market: str = "US",
    include_groups: str = "album,single,compilation,appears_on",
    limit: int = 50,
    _inter_page_sleep: float = 1,  # small throttle between pages
) -> list[dict]:
    albums: list[dict] = []
    url = f"{API}/artists/{artist_id}/albums"
    params = {
        "market": market,
        "include_groups": include_groups,
        "limit": limit,
        "offset": 0,
    }

    while True:
        print(url, params)
        page = _get(url, params)
        items = page.get("items", []) or []
        albums.extend(items)
        next_url = page.get("next")

        if not next_url:
            break

        # Spotify sometimes returns a full 'next' URL; if present, use it directly.
        if next_url.startswith("http"):
            url = next_url
            params = None
        else:
            params["offset"] += limit

        # tiny, polite pause between pages
        time.sleep(_inter_page_sleep)

    return _dedupe_albums(albums)


def _dedupe_albums(albums: t.List[dict]) -> t.List[dict]:
    seen, out = set(), []
    for a in albums:
        if a["id"] in seen:
            continue
        seen.add(a["id"])
        out.append(a)
    return out


def _parse_release_date(s: str, precision: str) -> dt.date:
    if precision == "day":
        return dt.date.fromisoformat(s)
    if precision == "month":
        y, m = s.split("-")
        return dt.date(int(y), int(m), 1)
    return dt.date(int(s[:4]), 1, 1)


def get_album_tracks(
    album_id: str, market: str = "US", _inter_page_sleep: float = 0.5
) -> list[dict]:
    tracks, limit, offset = [], 50, 0
    url = f"{API}/albums/{album_id}/tracks"
    while True:
        page = _get(url, {"market": market, "limit": limit, "offset": offset})
        items = page.get("items", []) or []
        tracks.extend(items)
        if len(items) < limit:
            break
        offset += limit
        time.sleep(_inter_page_sleep)
    return tracks


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def get_tracks_popularity(track_ids: t.List[str]) -> dict[str, int]:
    """Batch fetch popularity for up to 50 tracks per request."""
    pop: dict[str, int] = {}
    for batch in _chunk(track_ids, 50):
        data = _get(f"{API}/tracks", {"ids": ",".join(batch)})
        for t_obj in data.get("tracks", []):
            if t_obj and t_obj.get("id"):
                pop[t_obj["id"]] = t_obj.get("popularity")
    return pop


# ---------------------------
# Feature engineering
# ---------------------------
def compute_features_for_first_n_years(
    artist_id: str,
    years: int,
    market: str = "US",
    include_track_popularity: bool = False,
) -> dict:
    """
    Artist-level, *early-career* features from the first `years` after debut.

    Cadence & timing:
      - debut_date, window_cutoff_date, window_years
      - releases_total
      - releases_per_year
      - avg_days_between_releases
      - release_velocity_releases_per_day / per_year (slope of cumulative releases vs time)
      - gap_median_days, gap_std_days, max_dry_spell_days
      - front_loading_index (fraction of releases in first half of window)

    Collaboration:
      - tracks_total
      - collab_track_rate
      - unique_collaborator_count

    Labels & markets:
      - label_diversity_count
      - label_churn
      - label_hhi (Herfindahl index over labels)
      - market_coverage_count (union of available_markets)

    Track text/flags & duration:
      - explicit_rate
      - remix_rate
      - acoustic_rate
      - duration_ms_mean / median / min / max

    Popularity (current, not ex-ante; optional):
      - track_popularity_mean / median / min / max
      - share_hits_50 (popularity ≥ 50)
      - share_hits_70 (popularity ≥ 70)
    """

    # ----- helpers (local) -----
    from statistics import median, pstdev

    def _least_squares_slope(xs, ys):
        n = len(xs)
        if n < 2:
            return None
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if denom == 0:
            return None
        return (n * sxy - sx * sy) / denom

    def _compute_release_velocity(
        release_dates: list[dt.date], debut_date: dt.date
    ) -> dict:
        if not release_dates:
            return {
                "release_velocity_releases_per_day": None,
                "release_velocity_releases_per_year": None,
            }
        d_sorted = sorted(release_dates)
        xs = []
        ys = []
        cum = 0
        for d in d_sorted:
            cum += 1
            xs.append((d - debut_date).days)
            ys.append(cum)
        slope = _least_squares_slope(xs, ys)  # releases per day
        return {
            "release_velocity_releases_per_day": slope,
            "release_velocity_releases_per_year": slope * 365.25
            if slope is not None
            else None,
        }

    def _cadence_stats(release_dates: list[dt.date]) -> dict:
        d_sorted = sorted(release_dates)
        if len(d_sorted) < 2:
            return {
                "gap_median_days": None,
                "gap_std_days": None,
                "max_dry_spell_days": None,
            }
        gaps = [(d_sorted[i] - d_sorted[i - 1]).days for i in range(1, len(d_sorted))]
        return {
            "gap_median_days": float(median(gaps)),
            "gap_std_days": float(pstdev(gaps)) if len(gaps) > 1 else 0.0,
            "max_dry_spell_days": max(gaps),
        }

    def _front_loading_index(
        release_dates: list[dt.date], debut_date: dt.date, cutoff: dt.date
    ):
        if not release_dates:
            return None
        mid = debut_date + (cutoff - debut_date) / 2
        first_half = sum(1 for d in release_dates if d < mid)
        total = len(release_dates)
        return first_half / total if total > 0 else None

    def _track_text_and_flag_features(all_tracks: list[dict]) -> dict:
        titles = [t.get("name") or "" for t in all_tracks]
        exp_flags = [bool(t.get("explicit")) for t in all_tracks]
        durs = [
            t.get("duration_ms") for t in all_tracks if t.get("duration_ms") is not None
        ]

        def pct(n, d):
            return (n / d) if d else None

        remix = sum(1 for s in titles if "remix" in s.lower())
        acoustic = sum(1 for s in titles if "acoustic" in s.lower())

        out = {
            "explicit_rate": pct(sum(exp_flags), len(exp_flags)),
            "remix_rate": pct(remix, len(titles) or 0),
            "acoustic_rate": pct(acoustic, len(titles) or 0),
        }

        if durs:
            s = sorted(durs)
            mid = len(s) // 2
            med_val = float(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2)
            out.update(
                {
                    "duration_ms_mean": sum(durs) / len(durs),
                    "duration_ms_median": med_val,
                    "duration_ms_min": s[0],
                    "duration_ms_max": s[-1],
                }
            )
        else:
            out.update(
                {
                    "duration_ms_mean": None,
                    "duration_ms_median": None,
                    "duration_ms_min": None,
                    "duration_ms_max": None,
                }
            )
        return out

    def _popularity_thresholds(pop_vals: list[int]) -> dict:
        if not pop_vals:
            return {"share_hits_50": None, "share_hits_70": None}
        n = len(pop_vals)
        share_50 = sum(1 for v in pop_vals if v >= 50) / n
        share_70 = sum(1 for v in pop_vals if v >= 70) / n
        return {"share_hits_50": share_50, "share_hits_70": share_70}

    # ----- main body -----

    albums = get_artist_albums(artist_id, market=market)
    if not albums:
        sys.exit("Artist has no albums/singles in this market.")

    # Determine debut date across all releases
    dated = []
    for a in albums:
        rd = _parse_release_date(
            a["release_date"], a.get("release_date_precision", "day")
        )
        dated.append((rd, a))
    dated.sort(key=lambda x: x[0])
    debut_date = dated[0][0]
    cutoff = debut_date.replace(year=debut_date.year + years)

    # Filter albums within first N years
    window_albums = [a for d, a in dated if d < cutoff]
    if not window_albums:
        sys.exit(f"No releases found within the first {years} years after debut.")

    # Aggregate info
    release_dates: list[dt.date] = []
    album_type_counts: Counter[str] = Counter()
    labels_counter: Counter[str] = Counter()
    labels_by_release: list[str] = []
    markets_union: set[str] = set()
    track_rows: list[dict] = []
    all_album_tracks: list[dict] = []

    for a in window_albums:
        rd = _parse_release_date(
            a["release_date"], a.get("release_date_precision", "day")
        )
        release_dates.append(rd)
        album_type_counts[a.get("album_type", "unknown")] += 1
        label = a.get("label") or ""
        labels_counter[label] += 1
        labels_by_release.append(label)
        markets_union.update(a.get("available_markets", []))

        tracks = get_album_tracks(a["id"], market=market)
        all_album_tracks.extend(tracks)

        for t_obj in tracks:
            artists_meta = t_obj.get("artists", []) or []
            credited_ids = [x["id"] for x in artists_meta if x.get("id")]
            is_collab = len(credited_ids) > 1
            track_rows.append(
                {
                    "id": t_obj.get("id"),
                    "is_collab": is_collab,
                    "artist_ids": credited_ids,
                }
            )

    # Cadence basics
    release_dates.sort()
    gaps = [
        (release_dates[i] - release_dates[i - 1]).days
        for i in range(1, len(release_dates))
    ]
    avg_gap_days = (sum(gaps) / len(gaps)) if gaps else None

    # Normalize by actual elapsed years in-window (cap at today's date)
    elapsed_days = (min(cutoff, dt.date.today()) - debut_date).days
    releases_per_year = len(release_dates) / (
        elapsed_days / 365.25 if elapsed_days > 0 else 1
    )

    # Collaboration stats
    collab_tracks = sum(1 for r in track_rows if r["is_collab"])
    collab_rate = (collab_tracks / len(track_rows)) if track_rows else 0.0
    unique_collaborators = set(aid for r in track_rows for aid in r["artist_ids"])

    # Label diversity & markets
    label_diversity = len({k for k in labels_counter.keys() if k})
    market_coverage_count = len(markets_union)

    # Release-velocity and cadence extras
    velocity_feats = _compute_release_velocity(release_dates, debut_date)
    cadence_feats = _cadence_stats(release_dates)
    front_load = _front_loading_index(release_dates, debut_date, cutoff)

    # Track-level text/flags & durations
    track_text_feats = _track_text_and_flag_features(all_album_tracks)

    # (Optional) Popularity stats (current, not ex-ante)
    track_ids = [r["id"] for r in track_rows if r["id"]]
    popularity_stats: dict = {}
    if include_track_popularity and track_ids:
        pops = get_tracks_popularity(track_ids)
        vals = [v for v in pops.values() if v is not None]
        if vals:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            median_val = (
                vals_sorted[mid]
                if len(vals_sorted) % 2
                else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
            )
            popularity_stats = {
                "track_popularity_mean": sum(vals) / len(vals),
                "track_popularity_median": median_val,
                "track_popularity_max": max(vals),
                "track_popularity_min": min(vals),
            }
            popularity_stats.update(_popularity_thresholds(vals))
        else:
            popularity_stats = {
                "track_popularity_mean": None,
                "track_popularity_median": None,
                "track_popularity_max": None,
                "track_popularity_min": None,
                "share_hits_50": None,
                "share_hits_70": None,
            }
    elif include_track_popularity:
        # no track ids, but flag was on
        popularity_stats = {
            "track_popularity_mean": None,
            "track_popularity_median": None,
            "track_popularity_max": None,
            "track_popularity_min": None,
            "share_hits_50": None,
            "share_hits_70": None,
        }

    # Label churn & concentration

    features = {
        "artist_id": artist_id,
        "debut_date": debut_date.isoformat(),
        "window_years": years,
        "window_cutoff_date": cutoff.isoformat(),
        # basic cadence
        "releases_total": len(window_albums),
        "releases_per_year": releases_per_year,
        "avg_days_between_releases": avg_gap_days,
        # collaboration
        "tracks_total": len(track_rows),
        "collab_track_rate": collab_rate,
        "unique_collaborator_count": max(
            0, len(unique_collaborators) - 1
        ),  # rough subtract for main artist
        # labels & markets
        "label_diversity_count": label_diversity,
        "market_coverage_count": market_coverage_count,
        # album types
        "album_type_counts": dict(album_type_counts),
        # front loading
        "front_loading_index": front_load,
    }

    # merge all the extra blocks
    features.update(velocity_feats)
    features.update(cadence_feats)
    features.update(track_text_feats)
    features.update(popularity_stats)

    return features


def get_artist_popularity_by_year(
    artist_id: str,
    market: str = "US",
    include_appears_on: bool = False,
) -> list[dict]:
    """
    Return a list of per-year popularity metrics for the artist's tracks,
    spanning from debut release year to last release year, inclusive.

    Notes:
      - Popularity is Spotify's *current* 0–100 track popularity (not historical).
      - We map each track to the release year of the album/single it came from.
      - If a track appears on multiple releases, we keep the *earliest* year seen.
      - By default we exclude 'appears_on' to focus on the artist's own releases.
        Set include_appears_on=True to include featured/collab appearances.
      - Requires helpers: get_artist_albums, get_album_tracks, _parse_release_date, get_tracks_popularity.

    Returns: list of dicts sorted by year:
      [
        {
          "year": 2017,
          "track_count": 12,
          "pop_mean": 62.25,
          "pop_median": 64.0,
          "pop_min": 25,
          "pop_p25": 51.0,
          "pop_p75": 73.0,
          "pop_max": 85,
        },
        ...
      ]
    """
    # 1) Pull all releases (albums/singles, optionally appears_on)
    include_groups = "album,single,compilation" + (
        ",appears_on" if include_appears_on else ""
    )
    albums = get_artist_albums(artist_id, market=market, include_groups=include_groups)
    if not albums:
        return []

    # 2) Build track->year map (use earliest year for duplicates)
    track_year: dict[str, int] = {}
    for a in albums:
        rd = _parse_release_date(
            a["release_date"], a.get("release_date_precision", "day")
        )
        y = rd.year
        tracks = get_album_tracks(a["id"], market=market)
        for t in tracks:
            tid = t.get("id")
            if not tid:
                continue
            if tid not in track_year or y < track_year[tid]:
                track_year[tid] = y

    if not track_year:
        return []

    # 3) Fetch popularity for all unique tracks
    pops = get_tracks_popularity(list(track_year.keys()))
    if not pops:
        # If /tracks is temporarily blocked (unlikely), fall back to empty metrics
        return []

    # 4) Group popularity values by year
    year_to_vals: dict[int, list[int]] = {}
    for tid, y in track_year.items():
        val = pops.get(tid)
        if val is None:
            continue
        year_to_vals.setdefault(y, []).append(val)

    if not year_to_vals:
        return []

    # 5) Produce per-year summary stats
    def _median(vals: list[int]) -> float:
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        if n % 2:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0

    def _percentile(vals: list[int], p: float) -> float:
        # Simple nearest-rank percentile (p in [0,1])
        s = sorted(vals)
        if not s:
            return float("nan")
        k = max(1, int(round(p * (len(s) + 1)))) - 1
        k = min(k, len(s) - 1)
        return float(s[k])

    years_sorted = sorted(year_to_vals.keys())
    out: list[dict] = []
    for y in years_sorted:
        vals = year_to_vals[y]
        if not vals:
            continue
        out.append(
            {
                "year": y,
                "track_count": len(vals),
                "pop_mean": sum(vals) / len(vals),
                "pop_median": _median(vals),
                "pop_min": min(vals),
                "pop_p25": _percentile(vals, 0.25),
                "pop_p75": _percentile(vals, 0.75),
                "pop_max": max(vals),
            }
        )
    return out
