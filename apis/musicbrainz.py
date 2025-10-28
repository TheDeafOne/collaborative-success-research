# Minimal MusicBrainz artist lookup (single + multiple MBIDs)
# Drop this in a Jupyter cell.

from __future__ import annotations

import time
import typing as t

import requests

MB_BASE_URL = "https://musicbrainz.org/ws/2"
MB_HEADERS = {"User-Agent": "nepo-music/0.1 (kwoodbu1@jh.edu)"}


class MBError(Exception):
    """Simple wrapper for MusicBrainz API errors."""


def _mb_get(
    path: str,
    params: dict | None = None,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.2,
    timeout: float = 20.0,
) -> dict:
    """
    Internal helper for GET requests to MusicBrainz WS/2 with basic retry/backoff.
    Returns parsed JSON or raises MBError.
    """
    url = f"{MB_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    params = dict(params or {})
    params.setdefault("fmt", "json")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=MB_HEADERS, timeout=timeout)
            # Handle rate-limit response explicitly if present
            if resp.status_code == 503 or resp.status_code == 502:
                # Service busy; back off
                time.sleep(backoff_seconds * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff_seconds * attempt)
    raise MBError(
        f"MusicBrainz request failed after {max_retries} attempts: {last_err}"
    )


def lookup_artist(
    mbid: str,
    *,
    includes: t.Iterable[str] | None = None,
) -> dict:
    """
    Look up a single artist by MBID.

    Parameters
    ----------
    mbid : str
        MusicBrainz Artist MBID (UUID).
    includes : iterable of str, optional
        Extra sub-entities to include. Common options:
        ['aliases','tags','genres','url-rels','artist-rels','label-rels','work-rels','recordings','release-groups','releases']
        (Be careful with 'recordings'/'releases'—they can be large.)
    """
    params = {}
    if includes:
        # Space-separated 'inc' per MusicBrainz spec
        params["inc"] = " ".join(includes)
    return _mb_get(f"artist/{mbid}", params=params)


def lookup_artists(
    mbids: t.Iterable[str],
    *,
    includes: t.Iterable[str] | None = None,
    per_request_sleep: float = 1.1,
    stop_on_error: bool = False,
) -> dict[str, dict | MBError]:
    """
    Look up multiple artists by MBID sequentially (no true batch in WS/2).

    Parameters
    ----------
    mbids : iterable of str
        One or more MusicBrainz Artist MBIDs (UUIDs).
    includes : iterable of str, optional
        Passed through to lookup_artist (see above).
    per_request_sleep : float
        Seconds to sleep between calls to be polite (>= 1.0 recommended).
    stop_on_error : bool
        If True, raise on first error. If False, collect errors per MBID.

    Returns
    -------
    dict
        { mbid: artist_json_or_MBError, ... }
    """
    results: dict[str, dict | MBError] = {}
    for i, mbid in enumerate(mbids):
        try:
            results[mbid] = lookup_artist(mbid, includes=includes)
        except MBError as e:
            if stop_on_error:
                raise
            results[mbid] = e
        # polite rate limiting between calls (skip sleep after last)
        if per_request_sleep and i != len(list(mbids)) - 1:
            time.sleep(per_request_sleep)
    return results


# Add to your existing MusicBrainz helper cell


def get_artist_works(
    artist_mbid: str,
    *,
    limit: int = 100,
    offset: int = 0,
    max_results: int | None = None,
    per_request_sleep: float = 1.0,
) -> list[dict]:
    """
    Fetch all works (compositions) associated with a given artist MBID.

    Parameters
    ----------
    artist_mbid : str
        MusicBrainz Artist MBID (UUID).
    limit : int
        Max results per API call (MusicBrainz allows up to 100).
    offset : int
        Starting offset for paging.
    max_results : int | None
        Optional cap on total works fetched.
    per_request_sleep : float
        Delay between paginated calls to respect rate limiting.

    Returns
    -------
    list of dict
        Each dict is a MusicBrainz 'work' object.
    """
    works: list[dict] = []
    total = None

    while True:
        params = {
            "artist": artist_mbid,
            "limit": limit,
            "offset": offset,
            "fmt": "json",
        }
        data = _mb_get("work", params)
        page = data.get("works", [])
        works.extend(page)
        total = data.get("work-count", total)

        # Stop if we've hit the end or reached max_results
        if len(page) < limit:
            break
        if max_results and len(works) >= max_results:
            works = works[:max_results]
            break

        offset += limit
        time.sleep(per_request_sleep)

    return works
