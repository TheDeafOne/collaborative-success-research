# Local MusicBrainz artist lookup using PostgreSQL (no REST)
# Drop this in a Jupyter cell.

from __future__ import annotations

import os
import time
import typing as t
from dataclasses import dataclass

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg import errors as pg_errors
except Exception as e:
    raise RuntimeError(
        "This cell requires psycopg (v3). Install with: pip install psycopg[binary]"
    ) from e


# ----------------------------
# Config / connection handling
# ----------------------------


@dataclass
class MBDBConfig:
    # Prefer DSN if provided:
    #   MB_DSN="postgresql://musicbrainz:musicbrainz@localhost:5432/musicbrainz?options=-csearch_path%3Dmusicbrainz%2Cpublic"
    dsn: str | None = os.environ.get("MB_DSN", "postgresql://musicbrainz:musicbrainz@localhost:5432/musicbrainz")

    # If no DSN, we’ll try these (host None -> autodiscover)
    host: str | None = os.environ.get("MB_HOST")  # e.g. "localhost" or "musicbrainz-docker-db-1"
    port: int = int(os.environ.get("MB_PORT", "5432"))

    # IMPORTANT: this is the DB **name**, not the container name
    dbname: str = os.environ.get("MB_DBNAME", "musicbrainz")

    # musicbrainz-docker defaults
    user: str = os.environ.get("MB_USER", "musicbrainz")
    password: str | None = os.environ.get("MB_PASSWORD") or "musicbrainz"

    # Canonical schema
    search_path: str = os.environ.get("MB_SEARCH_PATH", "musicbrainz,public")

MB_HEADERS = {"User-Agent": "nepo-music-local/0.1 (kwoodbu1@jh.edu)"}


class MBError(Exception):
    """Simple wrapper for MusicBrainz DB errors."""


_DB_CONN: psycopg.Connection | None = None
_DB_CONF = MBDBConfig()


def configure_mb_connection(**kwargs) -> None:
    """
    Optional helper if you want to override connection parameters at runtime.
    Example:
        configure_mb_connection(user="postgres", password="secret", dbname="musicbrainz", host="db")
    """
    global _DB_CONF, _DB_CONN
    _DB_CONF = MBDBConfig(**{**_DB_CONF.__dict__, **kwargs})
    # Reset cached connection so it reconnects with new settings
    if _DB_CONN and not _DB_CONN.closed:
        _DB_CONN.close()
    _DB_CONN = None

from psycopg.rows import dict_row
from psycopg.pq import TransactionStatus

def _get_conn():
    global _DB_CONN
    if _DB_CONN and not getattr(_DB_CONN, "closed", True):
        return _DB_CONN

    last_err = None

    # 1) DSN path
    if _DB_CONF.dsn:
        try:
            _DB_CONN = psycopg.connect(_DB_CONF.dsn, row_factory=dict_row)
            _DB_CONN.autocommit = True
        except Exception as e:
            last_err = e
            _DB_CONN = None

    # 2) Host/port discovery path
    if not _DB_CONN:
        candidates: list[tuple[str, int]] = []
        if _DB_CONF.host:
            candidates.append((_DB_CONF.host, _DB_CONF.port))
        candidates += [
            ("localhost", _DB_CONF.port),
            ("127.0.0.1", _DB_CONF.port),
            ("musicbrainz-docker-db-1", 5432),
            ("db", 5432),
            ("postgres", 5432),
        ]
        for host, port in candidates:
            try:
                _DB_CONN = psycopg.connect(
                    host=host,
                    port=port,
                    dbname=_DB_CONF.dbname,
                    user=_DB_CONF.user,
                    password=_DB_CONF.password,
                    connect_timeout=2,
                    row_factory=dict_row,
                )
                _DB_CONN.autocommit = True
                _DB_CONF.host, _DB_CONF.port = host, port
                break
            except Exception as e:
                last_err = e
                _DB_CONN = None

    if not _DB_CONN:
        raise MBError(f"Failed to connect to MusicBrainz DB. Last error: {last_err}")

    # Set schema search order (safe in autocommit)
    with _DB_CONN.cursor() as cur:
        cur.execute("SET search_path TO " + _DB_CONF.search_path)

    return _DB_CONN


# ----------------------------
# Utilities
# ----------------------------

def _ymd_to_iso(y: int | None, m: int | None, d: int | None) -> str | None:
    """
    Convert Y/M/D integer parts to MusicBrainz-style ISO strings.
    Returns 'YYYY-MM-DD', 'YYYY-MM', 'YYYY', or None if year is missing.
    """
    if not y:
        return None
    if not m:
        return f"{int(y):04d}"
    if not d:
        return f"{int(y):04d}-{int(m):02d}"
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

def _fetchone(sql: str, params: tuple | dict) -> dict | None:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    except Exception:
        # reset aborted transaction if not autocommit
        if not getattr(conn, "autocommit", False):
            conn.rollback()
        raise

def _fetchall(sql: str, params: tuple | dict) -> list[dict]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception:
        if not getattr(conn, "autocommit", False):
            conn.rollback()
        raise
# ----------------------------
# Core lookups
# ----------------------------

def _get_artist_row_by_gid(artist_gid: str) -> dict | None:
    """
    Fetch the artist core row (including internal integer id) by MBID (gid).
    """
    sql = """
        SELECT
            a.id                AS artist_int_id,
            a.gid::text         AS id,
            a.name              AS name,
            a.sort_name         AS sort_name,
            a.comment           AS comment,
            a.ended             AS ended,
            a.begin_date_year   AS b_y, a.begin_date_month AS b_m, a.begin_date_day AS b_d,
            a.end_date_year     AS e_y, a.end_date_month   AS e_m, a.end_date_day   AS e_d,
            at.name             AS type,
            ar.gid::text        AS area_id,
            ar.name             AS area_name
        FROM artist a
        LEFT JOIN artist_type at ON at.id = a.type
        LEFT JOIN area ar        ON ar.id = a.area
        WHERE a.gid = %s
        LIMIT 1
    """
    return _fetchone(sql, (artist_gid,))


def _get_aliases(artist_id: int) -> list[dict]:
    sql = """
        SELECT
            aa.name,
            aa.sort_name,
            aat.name     AS type,
            aa.locale,
            aa.primary_for_locale AS primary,
            aa.begin_date_year AS b_y, aa.begin_date_month AS b_m, aa.begin_date_day AS b_d,
            aa.end_date_year   AS e_y, aa.end_date_month   AS e_m, aa.end_date_day   AS e_d
        FROM artist_alias aa
        LEFT JOIN artist_alias_type aat ON aat.id = aa.type
        WHERE aa.artist = %s
        ORDER BY aa.name, aa.locale NULLS LAST
    """
    rows = _fetchall(sql, (artist_id,))
    out: list[dict] = []
    for r in rows:
        out.append({
            "name": r["name"],
            "sort-name": r["sort_name"],
            "locale": r["locale"],
            "type": r["type"],
            "primary": bool(r["primary"]) if r["primary"] is not None else None,
            "begin": _ymd_to_iso(r["b_y"], r["b_m"], r["b_d"]),
            "end": _ymd_to_iso(r["e_y"], r["e_m"], r["e_d"]),
        })
    return out


def _get_tags(artist_id: int) -> list[dict]:
    sql = """
        SELECT t.name, at.count
        FROM artist_tag at
        JOIN tag t ON t.id = at.tag
        WHERE at.artist = %s
        ORDER BY at.count DESC, t.name
    """
    rows = _fetchall(sql, (artist_id,))
    return [{"name": r["name"], "count": int(r["count"])} for r in rows]


def _get_genres(artist_id: int) -> list[dict]:
    try:
        sql = """
            SELECT g.name
            FROM artist_genre ag
            JOIN genre g ON g.id = ag.genre
            WHERE ag.artist = %s
            ORDER BY g.name
        """
        rows = _fetchall(sql, (artist_id,))
        return [{"name": r["name"]} for r in rows]
    except Exception as e:
        # Older schemas may not have artist_genre; ignore undefined table.
        if isinstance(e.__cause__, pg_errors.UndefinedTable) or isinstance(e, pg_errors.UndefinedTable):
            # make sure the connection isn't left aborted
            conn = _get_conn()
            if not getattr(conn, "autocommit", False):
                conn.rollback()
            return []
        raise



def _get_url_rels(artist_id: int) -> list[dict]:
    """
    Return a subset of relations for target-type 'url'.
    """
    sql = """
        SELECT
            lt.name        AS rel_type,
            u.url          AS resource,
            l.begin_date_year AS b_y, l.begin_date_month AS b_m, l.begin_date_day AS b_d,
            l.end_date_year   AS e_y, l.end_date_month   AS e_m, l.end_date_day   AS e_d
        FROM l_artist_url lau
        JOIN link l       ON l.id = lau.link
        JOIN link_type lt ON lt.id = l.link_type
        JOIN url u        ON u.id = lau.entity1
        WHERE lau.entity0 = %s
        ORDER BY lt.name, u.url
    """
    rows = _fetchall(sql, (artist_id,))
    rels: list[dict] = []
    for r in rows:
        rels.append({
            "type": r["rel_type"],
            "target-type": "url",
            "direction": "forward",
            "begin": _ymd_to_iso(r["b_y"], r["b_m"], r["b_d"]),
            "end": _ymd_to_iso(r["e_y"], r["e_m"], r["e_d"]),
            "url": {"resource": r["resource"]},
        })
    return rels


# -----------------------------------
# Public API (drop-in style)
# -----------------------------------

def lookup_artist(
    mbid: str,
    *,
    includes: t.Iterable[str] | None = None,
) -> dict:
    """
    Look up a single artist by MBID from the local MusicBrainz PostgreSQL DB.

    Supported `includes`:
        'aliases'  -> returns artist aliases
        'tags'     -> returns tag names with counts
        'genres'   -> returns genre names (if artist_genre/genre tables exist)
        'url-rels' -> returns URL relations in 'relations'
    """
    includes = set(includes or [])
    row = _get_artist_row_by_gid(mbid)
    if not row:
        raise MBError(f"No artist found for MBID {mbid}")

    # Core fields modeled after WS/2
    artist_json: dict = {
        "id": row["id"],
        "name": row["name"],
        "sort-name": row["sort_name"],
        "disambiguation": row["comment"] or "",
        "type": row["type"],
        "life-span": {
            "begin": _ymd_to_iso(row["b_y"], row["b_m"], row["b_d"]),
            "end": _ymd_to_iso(row["e_y"], row["e_m"], row["e_d"]),
            "ended": bool(row["ended"]) if row["ended"] is not None else None,
        },
    }

    if row["area_id"] or row["area_name"]:
        artist_json["area"] = {"id": row["area_id"], "name": row["area_name"]}

    # Includes
    artist_int_id = int(row["artist_int_id"])

    if "aliases" in includes:
        artist_json["aliases"] = _get_aliases(artist_int_id)

    if "tags" in includes:
        artist_json["tags"] = _get_tags(artist_int_id)

    if "genres" in includes:
        artist_json["genres"] = _get_genres(artist_int_id)

    if "url-rels" in includes:
        # WS/2 nests all relations in a single "relations" array
        # Here we only add URL relations. Extend similarly for work/label/etc.
        artist_json.setdefault("relations", [])
        artist_json["relations"].extend(_get_url_rels(artist_int_id))

    return artist_json


def lookup_artists(
    mbids: t.Iterable[str],
    *,
    includes: t.Iterable[str] | None = None,
    per_request_sleep: float = 0.0,
    stop_on_error: bool = False,
) -> dict[str, dict | MBError]:
    """
    Look up multiple artists by MBID sequentially (no sleep required for local DB).

    Returns
    -------
    dict
        { mbid: artist_json_or_MBError, ... }
    """
    results: dict[str, dict | MBError] = {}
    mbid_list = list(mbids)  # avoid consuming generators twice
    for i, mbid in enumerate(mbid_list):
        try:
            results[mbid] = lookup_artist(mbid, includes=includes)
        except MBError as e:
            if stop_on_error:
                raise
            results[mbid] = e
        if per_request_sleep and i != len(mbid_list) - 1:
            time.sleep(per_request_sleep)
    return results


def get_artist_works(
    artist_mbid: str,
    *,
    limit: int = 100,
    offset: int = 0,
    max_results: int | None = None,
    per_request_sleep: float = 0.0,
) -> list[dict]:
    """
    Fetch works (compositions) associated with a given artist MBID
    via the l_artist_work link table.

    Returns a list of 'work' dicts similar to WS/2 items:
        { "id": <work MBID>, "title": <name>, "type": <work type>, "disambiguation": <comment> }
    """
    # Resolve artist int id
    arow = _get_artist_row_by_gid(artist_mbid)
    if not arow:
        raise MBError(f"No artist found for MBID {artist_mbid}")
    artist_id = int(arow["artist_int_id"])

    # Optionally get the total count (useful if you want to mirror WS/2 pagination externally)
    # total_sql = "SELECT count(*) AS c FROM l_artist_work WHERE entity0 = %s"
    # total = _fetchone(total_sql, (artist_id,))["c"]

    fetched: list[dict] = []

    while True:
        sql = """
            SELECT
                w.gid::text AS id,
                w.name      AS title,
                wt.name     AS type,
                w.comment   AS disambiguation
            FROM l_artist_work law
            JOIN work w       ON w.id = law.entity1
            LEFT JOIN work_type wt ON wt.id = w.type
            WHERE law.entity0 = %s
            ORDER BY w.name, w.id
            LIMIT %s OFFSET %s
        """
        rows = _fetchall(sql, (artist_id, limit, offset))
        page = [
            {
                "id": r["id"],
                "title": r["title"],
                "type": r["type"],
                "disambiguation": r["disambiguation"] or "",
            }
            for r in rows
        ]
        fetched.extend(page)

        # Break conditions: last page, or max_results reached
        if len(page) < limit:
            break
        if max_results is not None and len(fetched) >= max_results:
            fetched = fetched[:max_results]
            break

        offset += limit
        if per_request_sleep:
            time.sleep(per_request_sleep)

    return fetched
