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
    itersize: int = 100_000,
    path: str = "temp_table",           # "temp_table" or "unnest"
    set_work_mem: Optional[str] = "512MB",
    disable_jit: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one object per artist:
      {
        "artist_mbid": <UUID>,
        "works": [
          {
            "song_title": <str>,
            "release_date_year": <int|None>,
            "release_month": <int|None>,
            "contributor_mbids": <List[UUID]>   # all authors for that work
          }, ...
        ]
      }
    """
    if not artist_mbids:
        return

    # Validate & dedupe while preserving order
    seen, cohort = set(), []
    for mbid in artist_mbids:
        try:
            UUID(mbid)
            if mbid not in seen:
                seen.add(mbid)
                cohort.append(mbid)
        except:
            print(f'error reading in {mbid}')

    AUTHOR_ROLE_NAMES = ("composer", "lyricist", "writer", "librettist")

    with _connect() as conn:
        # Session-local knobs
        with conn.cursor() as cset:
            if set_work_mem:
                cset.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(set_work_mem)))
            if disable_jit:
                cset.execute("SET LOCAL jit = off")
            cset.execute("SET LOCAL synchronous_commit = off")

        # Resolve link_type IDs once; we’ll filter by ID to avoid the join at runtime
        with conn.cursor() as c_lt:
            c_lt.execute("SELECT id FROM link_type WHERE name = ANY(%s)", (list(AUTHOR_ROLE_NAMES),))
            role_ids = [r[0] for r in c_lt.fetchall()]
            if not role_ids:
                raise RuntimeError("Authorship link types not found.")

        use_temp = (path != "unnest")

        # Build cohort → artist ids (ints) + mbids (for final output)
        if use_temp:
            with conn.cursor() as cprep:
                cprep.execute("CREATE TEMP TABLE tmp_artist_gid (gid uuid PRIMARY KEY) ON COMMIT DROP")
                with cprep.copy("COPY tmp_artist_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cprep.execute("ANALYZE tmp_artist_gid")

                cprep.execute("""
                    CREATE TEMP TABLE tmp_artist AS
                    SELECT a.id, a.gid
                    FROM artist a
                    JOIN tmp_artist_gid g ON g.gid = a.gid
                """)
                cprep.execute("CREATE INDEX ON tmp_artist (id)")
                cprep.execute("ANALYZE tmp_artist")
        else:
            with conn.cursor() as cprep:
                cprep.execute("""
                    CREATE TEMP TABLE tmp_artist AS
                    SELECT a.id, a.gid
                    FROM artist a
                    JOIN (SELECT * FROM unnest(%s::uuid[])) x(gid) ON a.gid = x.gid
                """, (cohort,))
                cprep.execute("CREATE INDEX ON tmp_artist (id)")
                cprep.execute("ANALYZE tmp_artist")

        # 1) Per-artist authored works (keep artist_id so we can group later)
        with conn.cursor() as c1:
            c1.execute("""
                CREATE TEMP TABLE tmp_artist_authored_works AS
                SELECT DISTINCT
                    ta.id    AS artist_id,
                    ta.gid   AS artist_mbid,
                    w.id     AS work_id,
                    w.name   AS work_name
                FROM tmp_artist ta
                JOIN l_artist_work law ON law.entity0 = ta.id
                JOIN link lk           ON lk.id = law.link AND lk.link_type = ANY (%s::int[])
                JOIN work w            ON w.id = law.entity1
            """, (role_ids,))
            c1.execute("CREATE INDEX ON tmp_artist_authored_works (artist_id, work_id)")
            c1.execute("CREATE INDEX ON tmp_artist_authored_works (work_id)")
            c1.execute("ANALYZE tmp_artist_authored_works")

        # 2) Earliest known release date per work (MIN over synthetic date)
        with conn.cursor() as c2:
            c2.execute("""
                CREATE TEMP TABLE tmp_work_first_date AS
                SELECT w.work_id,
                       EXTRACT(YEAR  FROM MIN(make_date(rfrd.year,
                                                        COALESCE(rfrd.month, 12),
                                                        COALESCE(rfrd.day, 31))))::int AS release_date_year,
                       EXTRACT(MONTH FROM MIN(make_date(rfrd.year,
                                                        COALESCE(rfrd.month, 12),
                                                        COALESCE(rfrd.day, 31))))::int AS release_month
                FROM tmp_artist_authored_works w
                JOIN l_recording_work lrw ON lrw.entity1 = w.work_id
                JOIN recording_first_release_date rfrd ON rfrd.recording = lrw.entity0
                WHERE rfrd.year IS NOT NULL
                GROUP BY w.work_id
            """)
            c2.execute("CREATE INDEX ON tmp_work_first_date (work_id)")
            c2.execute("ANALYZE tmp_work_first_date")

        # 3) Contributor MBIDs per work (all authors)
        with conn.cursor() as c3:
            c3.execute("""
                CREATE TEMP TABLE tmp_work_contrib_mbids AS
                SELECT
                    law.entity1 AS work_id,
                    array_agg(DISTINCT ar.gid ORDER BY ar.gid) AS contributor_mbids
                FROM l_artist_work law
                JOIN link   lk ON lk.id = law.link AND lk.link_type = ANY (%s::int[])
                JOIN artist ar ON ar.id = law.entity0
                JOIN tmp_artist_authored_works w ON w.work_id = law.entity1
                GROUP BY law.entity1
            """, (role_ids,))
            c3.execute("CREATE INDEX ON tmp_work_contrib_mbids (work_id)")
            c3.execute("ANALYZE tmp_work_contrib_mbids")

        # 4) Stream rows ordered by artist_id so we can yield per-artist blocks
        sql_stream = """
            SELECT
                a.artist_mbid,
                a.work_name AS song_title,
                d.release_date_year,
                d.release_month,
                co.contributor_mbids
            FROM tmp_artist_authored_works a
            LEFT JOIN tmp_work_first_date      d  ON d.work_id = a.work_id
            LEFT JOIN tmp_work_contrib_mbids   co ON co.work_id = a.work_id
            ORDER BY a.artist_id, a.work_name
        """

        with conn.cursor(name="mbz_song_stream_grouped", row_factory=dict_row) as cur:
            cur.itersize = itersize
            cur.execute(sql_stream)

            current_artist = None
            bucket: List[Dict[str, Any]] = []

            for row in cur:
                artist_mbid = row["artist_mbid"]
                if current_artist is None:
                    current_artist = artist_mbid
                if artist_mbid != current_artist:
                    # flush previous artist block
                    yield {"artist_mbid": current_artist, "works": bucket}
                    current_artist = artist_mbid
                    bucket = []

                bucket.append({
                    "song_title": row["song_title"],
                    "release_date_year": row["release_date_year"],
                    "release_month": row["release_month"],
                    "contributor_mbids": row["contributor_mbids"] or [],
                })

            # flush last
            if current_artist is not None:
                yield {"artist_mbid": current_artist, "works": bucket}

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
