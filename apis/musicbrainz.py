#!/usr/bin/env python3
"""
musicbrainz_artist_songs_psycopg.py

Query a local MusicBrainz PostgreSQL database for an artist's songs (recordings)
by the artist's MBID (UUID). Returns a list of dicts.

Defaults connect to:
  host=localhost port=5432 dbname=musicbrainz_db user=musicbrainz password=musicbrainz

Env var overrides:
  MUSICBRAINZ_DB_HOST, MUSICBRAINZ_DB_PORT, MUSICBRAINZ_DB_NAME,
  MUSICBRAINZ_DB_USER, MUSICBRAINZ_DB_PASSWORD

Requires:
  pip install "psycopg[binary]"
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg import sql
import time


def _connect() -> psycopg.Connection:
    """Create a new psycopg (v3) connection using env vars or sensible defaults."""
    host = os.getenv("MUSICBRAINZ_DB_HOST", "localhost")
    port = int(os.getenv("MUSICBRAINZ_DB_PORT", "5432"))
    dbname = os.getenv("MUSICBRAINZ_DB_NAME", "musicbrainz_db")
    user = os.getenv("MUSICBRAINZ_DB_USER", "musicbrainz")
    password = os.getenv("MUSICBRAINZ_DB_PASSWORD", "musicbrainz")

    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def get_artist_songs_by_mbid(
    artist_mbid: str,
    *,
    unique_recordings: bool = True,  # kept for signature compatibility (ignored)
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    FAST: Return songs (works) the artist CREATED, with earliest known release date
    and contributors.

    Returns rows with keys:
      - song_title
      - release_date_year, release_date_month, release_date_day
      - contributors: List[{"artist_mbid": str, "artist_name": str, "roles": [str, ...]}]

    Notes:
      - "Created" = artist has an authorship relationship to the Work:
        composer / lyricist / writer / librettist.
      - Earliest date is computed from any recording linked to the work using
        recording_first_release_date (materialized view).
      - Does NOT traverse track/medium/release; much faster.
    """
    UUID(artist_mbid)

    # The authorship roles we consider as "created"
    # If you want to widen/narrow later, edit this tuple.
    author_roles = ("composer", "lyricist", "writer", "librettist")

    sql = f"""
    WITH lt_auth AS (
        SELECT id, name
        FROM link_type
        WHERE name = ANY (%s)
    ),
    -- Works the requested artist authored (any of the roles above)
    authored_works AS (
        SELECT DISTINCT w.id AS work_id, w.name AS work_name
        FROM artist a
        JOIN l_artist_work law      ON law.entity0 = a.id
        JOIN link lk                ON lk.id = law.link
        JOIN lt_auth lt             ON lt.id = lk.link_type
        JOIN work w                 ON w.id = law.entity1
        WHERE a.gid = %s
    ),
    -- Earliest known release-event date for ANY recording of each work.
    -- Use ROW_NUMBER over ordered dates and keep rn = 1 (fast, index-friendly).
    work_earliest_date AS (
        SELECT
            w.work_id,
            rfrd.year  AS release_date_year,
            rfrd.month AS release_date_month,
            rfrd.day   AS release_date_day,
            ROW_NUMBER() OVER (
                PARTITION BY w.work_id
            ) AS rn
        FROM authored_works w
        JOIN l_recording_work lrw ON lrw.entity1 = w.work_id   -- work -> recording
        JOIN recording r          ON r.id = lrw.entity0
        LEFT JOIN recording_first_release_date rfrd
               ON rfrd.recording = r.id
    ),
    -- All contributors (authors) to each work, aggregated to JSON
    work_contributors AS (
        SELECT
            w.work_id,
            jsonb_agg(
                DISTINCT jsonb_build_object(
                    'artist_mbid', ar.gid,
                    'artist_name', ar.name,
                    'roles',       roles.roles
                )
            ) AS contributors
        FROM authored_works w
        JOIN l_artist_work law2 ON law2.entity1 = w.work_id
        JOIN link lk2           ON lk2.id = law2.link
        JOIN lt_auth lt2        ON lt2.id = lk2.link_type
        JOIN artist ar          ON ar.id = law2.entity0
        -- aggregate roles-per-artist for this work
        JOIN LATERAL (
            SELECT array_agg(DISTINCT lt3.name) AS roles
            FROM l_artist_work law3
            JOIN link lk3     ON lk3.id = law3.link
            JOIN lt_auth lt3  ON lt3.id = lk3.link_type
            WHERE law3.entity1 = w.work_id AND law3.entity0 = ar.id
        ) roles ON TRUE
        GROUP BY w.work_id
    )
    SELECT
        w.work_name AS song_title,
        d.release_date_year,
        d.release_date_month,
        d.release_date_day,
        co.contributors
    FROM authored_works w
    LEFT JOIN work_earliest_date d
           ON d.work_id = w.work_id AND d.rn = 1
    LEFT JOIN work_contributors co
           ON co.work_id = w.work_id
    { "LIMIT %s" if limit is not None else "" }
    """

    params: Tuple[Any, ...]
    if limit is None:
        params = (list(author_roles), artist_mbid)
    else:
        params = (list(author_roles), artist_mbid, int(limit))

    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows: List[Dict[str, Any]] = cur.fetchall()

    return rows

# helper (optional) if you still want an ISO field in the consumer
def _iso_from_ymd(y, m, d) -> Optional[str]:
    if y is None:
        return None
    if m is None:
        return f"{y:04d}"
    if d is None:
        return f"{y:04d}-{m:02d}"
    return f"{y:04d}-{m:02d}-{d:02d}"

def stream_artists_songs_by_mbids(
    artist_mbids: Sequence[str],
    *,
    unique_recordings: bool = True,      # kept for signature compatibility (ignored here)
    path: str = "temp_table",            # "temp_table" (best for huge lists) or "unnest"
    itersize: int = 50_000,              # rows fetched per round-trip
    set_work_mem: Optional[str] = "256MB",  # None to skip (scoped to txn)
    prefer_earliest: bool = False,       # kept for signature compatibility (ignored here)
) -> Iterator[Dict[str, Any]]:
    """
    Stream songs (works) CREATED by any of the given artists.
    Returns minimal payload per (artist, work) row:

      - song_title
      - release_date_year, release_date_month, release_date_day
      - contributors: JSONB array of {artist_mbid, artist_name, roles}

    Notes:
      - "Created" = artist has authorship relationship to the Work
        (composer / lyricist / writer / librettist).
      - Earliest date comes from recording_first_release_date across ANY
        recording linked to the work (fast path).
      - We intentionally avoid track/medium/release joins for speed.
    """
    if not artist_mbids:
        return

    # Validate UUIDs, preserve input order while deduping
    seen, cohort = set(), []
    for mbid in artist_mbids:
        UUID(mbid)
        if mbid not in seen:
            seen.add(mbid)
            cohort.append(mbid)

    # roles that count as "created"
    author_roles = ("composer", "lyricist", "writer", "librettist")

    # Core query: select authored works for cohort artists, compute earliest date from any recording,
    # and aggregate contributors (all authors) as JSON.
    #
    # We output one row per (artist, work). If you prefer one row per work across all input artists,
    # change the SELECT/GROUP BY at the bottom to drop artist fields.
    def _make_sql(use_temp_table: bool) -> Tuple[str, Tuple[Any, ...]]:
        if use_temp_table:
            cohort_cte = "SELECT gid FROM tmp_artist_gid"
            params: Tuple[Any, ...] = (list(author_roles),)
        else:
            cohort_cte = "SELECT * FROM unnest(%s::uuid[])"
            params = (list(author_roles), cohort)

        sql_text = f"""
        WITH cohort(gid) AS ({cohort_cte}),
        lt_auth AS (
            SELECT id, name
            FROM link_type
            WHERE name = ANY (%s)
        ),
        -- map cohort artist gid -> id
        cohort_artist AS (
            SELECT a.id AS artist_id, a.gid AS artist_mbid, a.name AS artist_name
            FROM artist a
            JOIN cohort c ON c.gid = a.gid
        ),
        -- works authored by cohort artists
        authored AS (
            SELECT DISTINCT
                ca.artist_id,
                ca.artist_mbid,
                ca.artist_name,
                w.id   AS work_id,
                w.name AS work_name
            FROM cohort_artist ca
            JOIN l_artist_work law ON law.entity0 = ca.artist_id
            JOIN link lk           ON lk.id = law.link
            JOIN lt_auth lt        ON lt.id = lk.link_type
            JOIN work w            ON w.id = law.entity1
        ),
        -- earliest known release-event date for ANY recording of each work
        work_first_date AS (
            SELECT work_id,
                   rfrd.year  AS release_date_year,
                   rfrd.month AS release_date_month,
                   rfrd.day   AS release_date_day,
                   ROW_NUMBER() OVER (
                       PARTITION BY work_id
                   ) AS rn
            FROM authored a
            JOIN l_recording_work lrw ON lrw.entity1 = a.work_id
            JOIN recording r          ON r.id = lrw.entity0
            LEFT JOIN recording_first_release_date rfrd ON rfrd.recording = r.id
        ),
        -- all contributors (authors) to each work as JSON
        work_contrib AS (
            SELECT
                a.work_id,
                jsonb_agg(
                    DISTINCT jsonb_build_object(
                        'artist_mbid', ar.gid,
                        'artist_name', ar.name,
                        'roles',
                        (
                          SELECT array_agg(DISTINCT lt3.name)
                          FROM l_artist_work law3
                          JOIN link lk3  ON lk3.id = law3.link
                          JOIN lt_auth lt3 ON lt3.id = lk3.link_type
                          WHERE law3.entity1 = a.work_id AND law3.entity0 = ar.id
                        )
                    )
                ) AS contributors
            FROM authored a
            JOIN l_artist_work law2 ON law2.entity1 = a.work_id
            JOIN link lk2           ON lk2.id = law2.link
            JOIN lt_auth lt2        ON lt2.id = lk2.link_type
            JOIN artist ar          ON ar.id = law2.entity0
            GROUP BY a.work_id
        )
        SELECT
            a.artist_mbid,             -- included to partition the stream; drop if not needed
            a.artist_name,
            a.work_name AS song_title,
            d.release_date_year,
            d.release_date_month,
            d.release_date_day,
            co.contributors
        FROM authored a
        LEFT JOIN work_first_date d ON d.work_id = a.work_id AND d.rn = 1
        LEFT JOIN work_contrib   co ON co.work_id = a.work_id
        """
        return sql_text, params

    with _connect() as conn:
        # scope memory bump to this txn
        if set_work_mem:
            with conn.cursor() as cset:
                cset.execute(sql.SQL("SET LOCAL work_mem TO {}").format(sql.Literal(set_work_mem)))

        if path == "unnest":
            query, params = _make_sql(use_temp_table=False)
        else:
            # TEMP TABLE path for huge lists
            with conn.cursor() as cprep:
                cprep.execute("CREATE TEMP TABLE tmp_artist_gid (gid uuid PRIMARY KEY) ON COMMIT DROP")
                with cprep.copy("COPY tmp_artist_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cprep.execute("ANALYZE tmp_artist_gid")
            query, params = _make_sql(use_temp_table=True)

        # Server-side cursor for streaming
        with conn.cursor(name="mbz_song_stream", row_factory=dict_row) as cur:
            cur.itersize = itersize
            cur.execute(query, params)
            for row in cur:
                # If you want an ISO field in the stream, uncomment:
                # row["release_date_iso"] = _iso_from_ymd(
                #     row.get("release_date_year"),
                #     row.get("release_date_month"),
                #     row.get("release_date_day"),
                # )
                yield {
                    # If you truly don't want artist identity in each row, remove the next two fields.
                    "artist_mbid": row["artist_mbid"],
                    "artist_name": row["artist_name"],
                    "song_title": row["song_title"],
                    "release_date_year": row["release_date_year"],
                    "release_date_month": row["release_date_month"],
                    "release_date_day": row["release_date_day"],
                    "contributors": row["contributors"],
                }

if __name__ == "__main__":
    import json
    test_mbid = 'c8b03190-306c-4120-bb0b-6f2ebfc06ea9'

    start = time.time()
    data = get_artist_songs_by_mbid(
        test_mbid
    )
    end = time.time()
    print(end - start)
    
    print(len(data), data[0])
