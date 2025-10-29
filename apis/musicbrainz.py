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

# helper: fast role-id → role-name translation (done client-side)
def _map_roles_inplace(contributors: List[Dict[str, Any]], role_id_to_name: Dict[int, str]) -> None:
    for c in contributors or []:
        ids = c.pop("role_ids", []) or []
        c["roles"] = [role_id_to_name.get(rid, str(rid)) for rid in ids]

def stream_artists_songs_by_mbids(
    artist_mbids: Sequence[str],
    *,
    path: str = "temp_table",            # "temp_table" (best for huge cohorts) or "unnest"
    itersize: int = 50_000,              # rows fetched per round-trip
    set_work_mem: Optional[str] = "512MB",
    disable_jit: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Stream songs (works) CREATED by any of the given artists, returning minimal fields:

      - song_title
      - release_date_year, release_date_month, release_date_day (from recording_first_release_date)
      - contributors: [{artist_mbid, artist_name, roles: [str, ...]}]

    Notes:
      - "Created" = the artist has an authorship relationship to the Work
        (composer / lyricist / writer / librettist) via l_artist_work/link.
      - One row per distinct Work across the whole cohort (deduped).
      - No ORDER BY for maximum throughput; sort downstream if desired.
    """
    if not artist_mbids:
        return

    # Validate & dedupe input while preserving order
    seen, cohort = set(), []
    for mbid in artist_mbids:
        UUID(mbid)
        if mbid not in seen:
            seen.add(mbid)
            cohort.append(mbid)

    # Resolve authorship link_type IDs once (avoids per-row join to link_type)
    AUTHOR_ROLE_NAMES = ("composer", "lyricist", "writer", "librettist")

    with _connect() as conn:
        # Session-local performance knobs
        with conn.cursor() as cset:
            if set_work_mem:
                cset.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(set_work_mem)))
            if disable_jit:
                cset.execute("SET LOCAL jit = off")
            # Safe with TEMP writes; reduces fsync latency for this session
            cset.execute("SET LOCAL synchronous_commit = off")

        # Fetch link_type ids and build a mapping for client-side role name translation
        with conn.cursor() as c_lt:
            c_lt.execute(
                "SELECT id, name FROM link_type WHERE name = ANY(%s)",
                (list(AUTHOR_ROLE_NAMES),),
            )
            rows = c_lt.fetchall()
            if not rows:
                raise RuntimeError("Expected authorship link types not found.")
            role_id_to_name = {r[0]: r[1] for r in rows}
            role_ids = list(role_id_to_name.keys())  # used to filter lk.link_type

        # Build cohort: TEMP table path (best for huge lists) or unnest
        use_temp = (path != "unnest")
        if use_temp:
            with conn.cursor() as cprep:
                cprep.execute("CREATE TEMP TABLE tmp_artist_gid (gid uuid PRIMARY KEY) ON COMMIT DROP")
                with cprep.copy("COPY tmp_artist_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cprep.execute("ANALYZE tmp_artist_gid")
                # Map UUIDs -> integer artist ids once; keep a narrow temp table of ints
                cprep.execute("""
                    CREATE TEMP TABLE tmp_artist_id AS
                    SELECT a.id
                    FROM artist a
                    JOIN tmp_artist_gid g ON g.gid = a.gid
                """)
                cprep.execute("CREATE INDEX ON tmp_artist_id (id)")
                cprep.execute("ANALYZE tmp_artist_id")

        # Core (no ORDER BY): one row per distinct work authored by ANY cohort artist
        if use_temp:
            cohort_sql = "SELECT id FROM tmp_artist_id"
            params: Tuple[Any, ...] = (role_ids,)
        else:
            cohort_sql = "SELECT a.id FROM artist a JOIN (SELECT * FROM unnest(%s::uuid[])) x(gid) ON a.gid = x.gid"
            params = (cohort, role_ids)

        sql_text = f"""
        WITH cohort_artist(id) AS ({cohort_sql}),

        -- Works authored by any cohort artist (deduped to one row per work)
        authored_works AS (
            SELECT DISTINCT w.id AS work_id, w.name AS work_name
            FROM cohort_artist ca
            JOIN l_artist_work law ON law.entity0 = ca.id
            JOIN link lk           ON lk.id = law.link
            JOIN work w            ON w.id = law.entity1
            WHERE lk.link_type = ANY (%s)
        ),

        -- Earliest known (year, month, day) across ANY recording of the work
        work_first_date AS (
            SELECT work_id,
                   rfrd.year  AS release_date_year,
                   rfrd.month AS release_date_month,
                   rfrd.day   AS release_date_day,
                   ROW_NUMBER() OVER (
                       PARTITION BY work_id
                   ) AS rn
            FROM authored_works w
            JOIN l_recording_work lrw ON lrw.entity1 = w.work_id
            JOIN recording r          ON r.id = lrw.entity0
            LEFT JOIN recording_first_release_date rfrd ON rfrd.recording = r.id
        ),

        -- Pre-aggregate roles per (work, contributor-artist) using link_type IDs (no join to link_type)
        role_sets AS (
            SELECT
                law.entity1                 AS work_id,
                law.entity0                 AS artist_id,
                array_agg(DISTINCT lk.link_type) AS role_ids
            FROM l_artist_work law
            JOIN link lk ON lk.id = law.link
            WHERE lk.link_type = ANY (%s)
            GROUP BY law.entity1, law.entity0
        ),

        -- Final contributor list per work (role_ids translated to names in Python)
        work_contrib AS (
            SELECT
                rs.work_id,
                jsonb_agg(
                    jsonb_build_object(
                        'artist_mbid', ar.gid,
                        'artist_name', ar.name,
                        'role_ids',    rs.role_ids
                    )
                ) AS contributors
            FROM role_sets rs
            JOIN artist ar ON ar.id = rs.artist_id
            GROUP BY rs.work_id
        )

        SELECT
            w.work_name AS song_title,
            d.release_date_year,
            d.release_date_month,
            d.release_date_day,
            co.contributors
        FROM authored_works w
        LEFT JOIN work_first_date d ON d.work_id = w.work_id AND d.rn = 1
        LEFT JOIN work_contrib   co ON co.work_id = w.work_id
        """

        # Server-side cursor for streaming
        with conn.cursor(name="mbz_song_stream_fast", row_factory=dict_row) as cur:
            cur.itersize = itersize
            if use_temp:
                # two %s placeholders in the query, both for role_ids
                cur.execute(sql_text, (role_ids, role_ids))
            else:
                # three placeholders: cohort (uuid[]), role_ids, role_ids
                cur.execute(sql_text, (cohort, role_ids, role_ids))
            for row in cur:
                _map_roles_inplace(row.get("contributors"), role_id_to_name)
                yield row

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
