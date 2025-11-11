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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg import sql
import time

import math


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


# ---------- one-time infra: indexes you likely want ----------
def ensure_perf_indexes() -> None:
    """
    Create helpful indexes if they don't already exist. Safe to call multiple times.
    Run this once on your MBz DB (outside hot paths).
    """
    stmts = [
        # Speed authored-work lookups in both directions
        """
        CREATE INDEX IF NOT EXISTS idx_l_artist_work_entity0_link_entity1
        ON l_artist_work (entity0, link, entity1);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_l_artist_work_entity1_link_entity0
        ON l_artist_work (entity1, link, entity0);
        """,
        # Recording -> work
        """
        CREATE INDEX IF NOT EXISTS idx_l_recording_work_entity0_entity1
        ON l_recording_work (entity0, entity1);
        """,
        # Artist credit traversals
        """
        CREATE INDEX IF NOT EXISTS idx_artist_credit_name_artist
        ON artist_credit_name (artist);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_artist_credit_name_ac
        ON artist_credit_name (artist_credit);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_recording_artist_credit
        ON recording (artist_credit);
        """,
        # Role filter
        """
        CREATE INDEX IF NOT EXISTS idx_link_link_type
        ON link (link_type);
        """,
    ]

    with _connect() as conn, conn.cursor() as cur:
        for s in stmts:
            cur.execute(s)
        conn.commit()


def refresh_work_contrib_mv(concurrently: bool = True) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"REFRESH MATERIALIZED VIEW {'CONCURRENTLY' if concurrently else ''} work_contrib_mbids;"
        )
        conn.commit()


# ---------- one-time infra: materialized view for contributors ----------
def ensure_work_contrib_mv() -> None:
    """
    Creates and (re)indexes a materialized view mapping work_id -> contributor_mbids (authors).
    Call once after you ingest/refresh MBz. Refresh later with refresh_work_contrib_mv().
    """
    create_mv = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS work_contrib_mbids AS
    WITH role_ids AS (
      SELECT id FROM link_type WHERE name IN ('composer','lyricist','writer','librettist')
    )
    SELECT
      law.entity1 AS work_id,
      array_agg(DISTINCT ar.gid ORDER BY ar.gid) AS contributor_mbids
    FROM l_artist_work law
    JOIN link   lk ON lk.id = law.link AND lk.link_type IN (SELECT id FROM role_ids)
    JOIN artist ar ON ar.id = law.entity0
    GROUP BY law.entity1;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(create_mv)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_work_contrib_mbids_work_id ON work_contrib_mbids (work_id);"
        )
        conn.commit()


def stream_artists_songs_by_mbids(
    artist_mbids,
    *,
    include_aliases: bool = False,
    include_country: bool = True,
    include_region_city: bool = True,
    include_perf_roles: bool = True,  # toggles l_artist_recording path in roles only
    include_genres: bool = True,  # needs work_genre/genre or work_tag/tag
    include_collaborators: bool = True,  # needs work_contrib_roles MV
    input_path: str = "temp_table",
    work_mem: str = "2GB",
    disable_jit: bool = True,
    itersize: int = 100_000,
):
    # ---- normalize inputs (same as before) ----
    raw = list(artist_mbids or [])
    if not raw:
        return

    def _coerce_uuid(s):
        from uuid import UUID

        try:
            s = str(s).strip()
            return str(UUID(s)) if s else None
        except Exception:
            return None

    def _dedupe(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    valid = [u for u in (_coerce_uuid(x) for x in raw) if u]
    cohort = _dedupe(valid)
    if not cohort:
        return

    from psycopg import sql
    from psycopg.rows import dict_row

    with _connect() as conn:
        with conn.cursor() as cset:
            if work_mem:
                cset.execute(
                    sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(work_mem))
                )
            if disable_jit:
                cset.execute("SET LOCAL jit = off")
            cset.execute("SET LOCAL synchronous_commit = off")
            cset.execute("SET LOCAL max_parallel_workers_per_gather = 8")
            cset.execute("SET LOCAL parallel_leader_participation = on")

        # Build inp (UNLOGGED for speed)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS inp;")
            if input_path == "temp_table":
                cur.execute(
                    "CREATE TEMP TABLE tmp_input_gid (gid uuid PRIMARY KEY) ON COMMIT DROP;"
                )
                with cur.copy(
                    "COPY tmp_input_gid (gid) FROM STDIN WITH (FORMAT text)"
                ) as cp:
                    cp.write("\n".join(cohort) + "\n")
                cur.execute("""
                    CREATE UNLOGGED TABLE inp AS
                    SELECT a.id, a.gid, a.name, a.area, a.begin_area
                    FROM artist a JOIN tmp_input_gid i ON i.gid = a.gid;
                """)
            else:
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE inp AS
                    SELECT a.id, a.gid, a.name, a.area, a.begin_area
                    FROM artist a
                    JOIN unnest(%s::uuid[]) x(gid) ON a.gid = x.gid;
                """,
                    (cohort,),
                )
            cur.execute("CREATE INDEX ON inp (id); ANALYZE inp;")

        # Detect optional tables/MVs
        def _exists(tab):
            with conn.cursor() as c:
                c.execute("SELECT to_regclass(%s)", (tab,))
                return c.fetchone()[0] is not None

        has_wfrd = _exists("work_first_release_date")
        has_wcr = _exists("work_contrib_roles")
        use_work_genre = _exists("work_genre") and _exists("genre")
        use_work_tag = _exists("work_tag") and _exists("tag")

        # Build the SQL parts conditionally
        role_ids_cte = "WITH role_ids AS (SELECT id, name FROM link_type WHERE name = ANY (%%s::text[]))"

        aliases_cte = (
            """
        ,aliases AS (
          SELECT aa.artist, array_agg(DISTINCT aa.name ORDER BY aa.name) AS alias_names
          FROM artist_alias aa JOIN inp i ON i.id = aa.artist
          GROUP BY aa.artist
        )"""
            if include_aliases
            else ""
        )

        country_cte = (
            """
        ,country AS (
          SELECT i.id AS artist_id, a1.name AS country_label
          FROM inp i LEFT JOIN area a1 ON a1.id = i.area
        )"""
            if include_country
            else ""
        )

        rc_cte = (
            """
        ,region_city AS (
          SELECT i.id AS artist_id, a_city.name AS region_city_label
          FROM inp i LEFT JOIN area a_city ON a_city.id = i.begin_area
        )"""
            if include_region_city
            else ""
        )

        # Roles: authorship always; performance optional (for the top-level "roles" string)
        author_roles_cte = """
        ,author_roles AS (
          SELECT i.id AS artist_id, array_agg(DISTINCT lt.name) AS roles
          FROM inp i
          JOIN l_artist_work law ON law.entity0 = i.id
          JOIN link lk ON lk.id = law.link
          JOIN role_ids lt ON lt.id = lk.link_type
          GROUP BY i.id
        )"""
        perf_roles_cte = (
            """
        ,perf_roles AS (
          SELECT i.id AS artist_id, array_agg(DISTINCT COALESCE(lt2.name,'performer')) AS roles
          FROM inp i
          JOIN l_artist_recording lar ON lar.entity0 = i.id
          JOIN link lk2 ON lk2.id = lar.link
          LEFT JOIN link_type lt2 ON lt2.id = lk2.link_type
          GROUP BY i.id
        )"""
            if include_perf_roles
            else ""
        )
        artist_roles_cte = """
        ,artist_roles AS (
          SELECT artist_id, array_agg(DISTINCT role) AS role_list
          FROM (
            SELECT artist_id, unnest(roles) AS role FROM author_roles
            %s
          ) u
          GROUP BY artist_id
        )""" % (
            "UNION ALL SELECT artist_id, unnest(roles) FROM perf_roles"
            if include_perf_roles
            else ""
        )

        # Works: authored ∪ performed
        rw_cte = """
        ,rw AS (
          SELECT i.id AS artist_id, w.id AS work_id, w.name AS work_name
          FROM inp i
          JOIN l_artist_work law ON law.entity0 = i.id
          JOIN link lk ON lk.id = law.link AND lk.link_type IN (SELECT id FROM role_ids)
          JOIN work w ON w.id = law.entity1
          UNION ALL
          SELECT i.id, w.id, w.name
          FROM inp i
          JOIN artist_credit_name acn ON acn.artist = i.id
          JOIN artist_credit ac ON ac.id = acn.artist_credit
          JOIN recording r ON r.artist_credit = ac.id
          JOIN l_recording_work lrw ON lrw.entity0 = r.id
          JOIN work w ON w.id = lrw.entity1
        ),
        rw_dedup AS (
          SELECT artist_id, work_id, MIN(work_name) AS work_name
          FROM rw GROUP BY artist_id, work_id
        )"""

        # Genres source
        if include_genres and use_work_genre:
            genres_cte = """
            ,work_genres AS (
              SELECT d.artist_id, d.work_id, array_agg(DISTINCT g.name ORDER BY g.name) AS genres
              FROM rw_dedup d
              JOIN work_genre wg ON wg.work = d.work_id
              JOIN genre g ON g.id = wg.genre
              GROUP BY d.artist_id, d.work_id
            )"""
            genres_join = "LEFT JOIN work_genres wg ON wg.artist_id = d.artist_id AND wg.work_id = d.work_id"
        elif include_genres and use_work_tag:
            genres_cte = """
            ,work_genres AS (
              SELECT d.artist_id, d.work_id, array_agg(DISTINCT t.name ORDER BY t.name) AS genres
              FROM rw_dedup d
              JOIN work_tag wt ON wt.work = d.work_id
              JOIN tag t ON t.id = wt.tag
              GROUP BY d.artist_id, d.work_id
            )"""
            genres_join = "LEFT JOIN work_genres wg ON wg.artist_id = d.artist_id AND wg.work_id = d.work_id"
        else:
            genres_cte = ""
            genres_join = ""

        # First release date + collaborators via MV joins if present
        date_join = (
            "LEFT JOIN work_first_release_date wfrd ON wfrd.work_id = d.work_id"
            if has_wfrd
            else ""
        )
        collab_join = (
            "LEFT JOIN work_contrib_roles wcr ON wcr.work_id = d.work_id"
            if (include_collaborators and has_wcr)
            else ""
        )

        # --------- SQL returns flat rows (no jsonb_agg) ----------
        q = f"""
        {role_ids_cte}
        {aliases_cte}
        {country_cte}
        {rc_cte}
        {author_roles_cte}
        {perf_roles_cte}
        {artist_roles_cte}
        {rw_cte}
        {genres_cte}

        SELECT
          i.gid          AS mbid,
          i.name         AS artist_name,
          ar.role_list   AS roles_array,
          %s             AS aliases_array,
          %s             AS country_label,
          %s             AS region_city_label,
          d.work_id,
          d.work_name,
          %s             AS first_date,
          %s             AS genres_array,
          %s             AS collaborators_json
        FROM inp i
        LEFT JOIN artist_roles ar ON ar.artist_id = i.id
        {("LEFT JOIN aliases al ON al.artist = i.id" if include_aliases else "")}
        {("LEFT JOIN country  c ON c.artist_id = i.id" if include_country else "")}
        {("LEFT JOIN region_city rc ON rc.artist_id = i.id" if include_region_city else "")}
        LEFT JOIN rw_dedup d ON d.artist_id = i.id
        {date_join}
        {genres_join}
        {collab_join}
        ;
        """

        # SELECT param placeholders to avoid rewriting the SQL:
        aliases_sel = "al.alias_names" if include_aliases else "NULL::text[]"
        country_sel = "c.country_label" if include_country else "NULL::text"
        rc_sel = "rc.region_city_label" if include_region_city else "NULL::text"
        date_sel = (
            "to_char(wfrd.first_date, 'YYYY-MM-DD')" if has_wfrd else "NULL::text"
        )
        genres_sel = (
            "wg.genres"
            if (include_genres and (use_work_genre or use_work_tag))
            else "NULL::text[]"
        )
        collab_sel = (
            "wcr.collaborators"
            if (include_collaborators and has_wcr)
            else "'[]'::jsonb"
        )

        q = q % (aliases_sel, country_sel, rc_sel, date_sel, genres_sel, collab_sel)

        params = (list(AUTHOR_ROLES),)

        # Stream rows and assemble JSON in Python
        with conn.cursor(name="mbz_turbo", row_factory=dict_row) as cur:
            cur.itersize = itersize
            cur.execute(q, params)

            # bucket by artist
            buckets = {}

            def _ensure(mbid, name, roles, aliases, country, rc):
                b = buckets.get(mbid)
                if b is None:
                    buckets[mbid] = b = {
                        "mbid": mbid,
                        "artist_name": name,
                        "roles": ", ".join(roles or []) if roles else "",
                        "aliases": aliases or [],
                        "country": country,
                        "region_city": rc,
                        "works": [],
                    }
                return b

            for row in cur:
                mbid = str(row["mbid"])
                b = _ensure(
                    mbid,
                    row["artist_name"],
                    row["roles_array"],
                    row["aliases_array"],
                    row["country_label"],
                    row["region_city_label"],
                )
                if row["work_id"] is not None:
                    b["works"].append(
                        {
                            "id": row["work_id"],
                            "name": row["work_name"],
                            "release_date": row["first_date"],
                            "genres": row["genres_array"] or [],
                            "collaborators": list(row["collaborators_json"] or []),
                        }
                    )

            # yield in input order
            seen = set()
            for mbid in cohort:
                if mbid in buckets:
                    seen.add(mbid)
                    yield buckets[mbid]
                else:
                    yield {
                        "mbid": mbid,
                        "artist_name": None,
                        "roles": "",
                        "aliases": [],
                        "country": None,
                        "region_city": None,
                        "works": [],
                    }


def get_artist_resources(artist_mbid, work_mem=None):
    """
    Given an artist MBID, return all associated external resource URLs.

    Args:
        artist_mbid (str): MusicBrainz Identifier (UUID) of the artist.
        work_mem (str | None): Optional PostgreSQL work_mem setting (e.g., '64MB').

    Returns:
        list[dict]: Each dict contains {'url': str, 'link_type': str, 'begin_date': str, 'end_date': str}.
    """
    query = sql.SQL("""
        SELECT
            u.id AS url_id,
            u.url AS url,
            lt.name AS link_type,
            l.begin_date_year,
            l.end_date_year
        FROM artist a
        JOIN l_artist_url lau ON lau.entity0 = a.id
        JOIN link l ON l.id = lau.link
        JOIN link_type lt ON lt.id = l.link_type
        JOIN url u ON u.id = lau.entity1
        WHERE a.gid = {artist_mbid}
        ORDER BY lt.name;
    """).format(artist_mbid=sql.Literal(artist_mbid))

    with _connect() as conn:
        with conn.cursor() as cset:
            # Optional work_mem tuning
            if work_mem:
                cset.execute(
                    sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(work_mem))
                )

            cset.execute(query)
            rows = cset.fetchall()

    return [
        {
            "url": row[1],
            "link_type": row[2],
            "begin_date": row[3],
            "end_date": row[4],
        }
        for row in rows
    ]


def _batched(iterable: Iterable[str], n: int) -> Iterator[List[str]]:
    """Yield lists of size <= n from an iterable without loading all items."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


_PLATFORM_SUBSTRINGS = [
    # YouTube & YouTube Music
    "youtube.com",
    "youtu.be",
    "music.youtube.com",
    # SoundCloud
    "soundcloud.com",
    # Spotify
    "open.spotify.com",
    "spotify.com",
    # Last.fm
    "last.fm",
    # Twitter / X
    "twitter.com",
    "x.com",
    # Instagram (user list had 'instagramr' — assuming Instagram)
    "instagram.com",
]


def stream_artist_resources(
    mbids: Iterable[str],
    work_mem: str | None = None,
    batch_size: int = 10_000,
) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    """
    Stream (artist_mbid, resources[]) for artists having URLs matching selected platforms.
    Skips artists with no matching resources.

    resources[] = [{ 'url': str, 'link_type': str }]

    Notes:
    - psycopg3-safe: no % wildcards in the SQL text; all patterns are parameters.
    - Uses a server-side cursor to avoid loading all rows into memory.
    - Uses a.gid = ANY(%s) to pass batched MBIDs.
    """

    # Build OR’ed ILIKE clauses as placeholders (u.url ILIKE %s OR ...)
    like_clause = " OR ".join(["u.url ILIKE %s"] * len(_PLATFORM_SUBSTRINGS))

    # Plain string is fine with psycopg3; we only pass parameters via %s
    query = f"""
        SELECT
            a.gid AS artist_mbid,
            lt.name AS link_type,
            u.url AS url
        FROM artist a
        JOIN l_artist_url lau ON lau.entity0 = a.id
        JOIN link l ON l.id = lau.link
        JOIN link_type lt ON lt.id = l.link_type
        JOIN url u ON u.id = lau.entity1
        WHERE a.gid = ANY(%s)
          AND ({like_clause})
        ORDER BY a.gid;
    """

    # Precompute the LIKE parameter list once (e.g., '%spotify%')
    like_params = [f"%{s}%" for s in _PLATFORM_SUBSTRINGS]

    with _connect() as conn:
        # SET LOCAL applies within the current transaction
        if work_mem:
            with conn.cursor() as c:
                c.execute("SET LOCAL work_mem = %s", (work_mem,))

        for batch in _batched(mbids, batch_size):
            # Server-side (named) cursor for streaming rows
            with conn.cursor(name="artist_resource_stream") as cur:
                # Parameters: first the array of mbids, then the like patterns
                cur.execute(query, (batch, *like_params))

                current_artist = None
                current_resources: List[Dict[str, str]] = []

                for artist_mbid, link_type, url in cur:
                    if artist_mbid != current_artist:
                        # emit previous artist if present
                        if current_artist is not None and current_resources:
                            yield current_artist, current_resources
                        current_artist = artist_mbid
                        current_resources = []

                    current_resources.append(
                        {
                            "link_type": link_type,
                            "url": url,
                        }
                    )

                # emit the final artist from this batch
                if current_artist is not None and current_resources:
                    yield current_artist, current_resources
