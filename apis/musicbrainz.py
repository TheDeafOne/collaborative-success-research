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

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import UUID

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


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
      array_agg(DISTINCT ar.gid) AS contributor_mbids
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

AUTHOR_ROLES = [
    # writing
    "writer",
    "composer",
    "lyricist",
    "librettist",
    "scriptwriter",
    "translator",
    "producer",
    "singer",
    "vocal", 
    "artist",

    "engineer",
    # arranging / scoring
    "arranger",
    "instrument arranger",
    "orchestrator",
    "vocal arranger",

    # transformations / editorial
    "adapter",
    "revised by",
    "reconstructed by",
]


def stream_artists_songs_by_mbids(
    artist_mbids,
    *,
    include_country: bool = True,
    include_region_city: bool = True,
    include_perf_roles: bool = True,  # toggles l_artist_recording path in roles only
    include_genres: bool = True,      # needs work_genre/genre or work_tag/tag
    include_collaborators: bool = True,  # needs work_contrib_roles MV
    input_path: str = "temp_table",
    work_mem: str = "2GB",
    disable_jit: bool = True,
    itersize: int = 10_000,
):
    """
    Yields one dictionary per artist MBID containing all aggregated artist–work–recording–release
    metadata. Each yielded object has the structure:

    {
    "mbid": str,
    "artist_name": str | None,
    "roles": str,                      # comma-separated artist roles
    "country": str | None,             # artist.country area name (if enabled)
    "region_city": str | None,         # artist.begin_area name (if enabled)

    "works": [
        {
        "id": int,
        "name": str,
        "first_release_date": "YYYY-MM-DD" | None,
        "genres": list[str],           # from work_genre/genre or work_tag/tag (if enabled)
        "collaborators": list[json],   # from work_contrib_roles (if enabled)
        },
        ...
    ],

    "recordings": [
        {
        "id": int,
        "name": str,
        "length_ms": int | None,
        "work_id": int | None,         # associated work
        "release_id": int | None,      # release containing this recording
        },
        ...
    ],

    "releases": [
        {
        "id": int,
        "title": str,
        "date": "YYYY-MM-DD" | None,   # earliest release_event date
        "release_group_id": int | None,
        "label_ids": list[str],        # distinct label.gid values
        "label_names": list[str],      # distinct label names
        },
        ...
    ],
    }

    The generator preserves the input MBID order and emits placeholder objects for any
    MBIDs not found in the database. All per-artist lists are fully deduplicated and
    pre-aggregated inside PostgreSQL for performance.
    """

    # ---- normalize inputs ----
    raw = list(artist_mbids or [])
    if not raw:
        return

    def _coerce_uuid(s):
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

        # Helper to check for optional objects
        def _exists(tab):
            with conn.cursor() as c:
                c.execute("SELECT to_regclass(%s)", (tab,))
                return c.fetchone()[0] is not None

        has_wfrd = _exists("work_first_release_date")
        has_wcr = _exists("work_contrib_roles")
        use_work_genre = _exists("work_genre") and _exists("genre")
        use_work_tag = _exists("work_tag") and _exists("tag")

        # -------------------------------------------------------------
        # 1. Build inp (cohort artists)
        # -------------------------------------------------------------
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
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE inp AS
                    SELECT a.id, a.gid, a.name, a.area, a.begin_area
                    FROM artist a
                    JOIN tmp_input_gid i ON i.gid = a.gid;
                    """
                )
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
            cur.execute("CREATE INDEX ON inp (id);")
            cur.execute("CREATE INDEX ON inp (gid);")
            cur.execute("ANALYZE inp;")

        # -------------------------------------------------------------
        # 2. Roles (author + performance)
        # -------------------------------------------------------------
        with conn.cursor() as cur:
            # role_ids: link types we care about
            cur.execute("DROP TABLE IF EXISTS role_ids;")
            cur.execute(
                """
                CREATE TEMP TABLE role_ids AS
                SELECT id, name
                FROM link_type
                WHERE name = ANY (%s::text[]);
                """,
                (list(AUTHOR_ROLES),),
            )
            cur.execute("CREATE INDEX ON role_ids (id);")
            cur.execute("ANALYZE role_ids;")

            # author roles: via l_artist_work
            cur.execute("DROP TABLE IF EXISTS author_roles;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE author_roles AS
                SELECT i.id AS artist_id,
                       array_agg(DISTINCT lt.name) AS roles
                FROM inp i
                JOIN l_artist_work law ON law.entity0 = i.id
                JOIN link lk ON lk.id = law.link
                JOIN role_ids lt ON lt.id = lk.link_type
                GROUP BY i.id;
                """
            )
            cur.execute("CREATE INDEX ON author_roles (artist_id);")
            cur.execute("ANALYZE author_roles;")

            # performance roles: via l_artist_recording
            if include_perf_roles:
                cur.execute("DROP TABLE IF EXISTS perf_roles;")
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE perf_roles AS
                    SELECT i.id AS artist_id,
                           array_agg(DISTINCT COALESCE(lt2.name,'performer')) AS roles
                    FROM inp i
                    JOIN l_artist_recording lar ON lar.entity0 = i.id
                    JOIN link lk2 ON lk2.id = lar.link
                    LEFT JOIN link_type lt2 ON lt2.id = lk2.link_type
                    GROUP BY i.id;
                    """
                )
                cur.execute("CREATE INDEX ON perf_roles (artist_id);")
                cur.execute("ANALYZE perf_roles;")
            else:
                cur.execute("DROP TABLE IF EXISTS perf_roles;")
                cur.execute(
                    "CREATE UNLOGGED TABLE perf_roles (artist_id int, roles text[]);"
                )

            # final artist_roles
            cur.execute("DROP TABLE IF EXISTS artist_roles;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE artist_roles AS
                SELECT artist_id,
                       array_agg(DISTINCT role) AS role_list
                FROM (
                  SELECT artist_id, unnest(roles) AS role
                  FROM author_roles
                  UNION ALL
                  SELECT artist_id, unnest(roles) AS role
                  FROM perf_roles
                ) u
                GROUP BY artist_id;
                """
            )
            cur.execute("CREATE INDEX ON artist_roles (artist_id);")
            cur.execute("ANALYZE artist_roles;")

        # -------------------------------------------------------------
        # 3. Country / region_city
        # -------------------------------------------------------------
        if include_country:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS country;")
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE country AS
                    SELECT i.id AS artist_id, a1.name AS country_label
                    FROM inp i
                    LEFT JOIN area a1 ON a1.id = i.area;
                    """
                )
                cur.execute("CREATE INDEX ON country (artist_id);")
                cur.execute("ANALYZE country;")

        if include_region_city:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS region_city;")
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE region_city AS
                    SELECT i.id AS artist_id, a_city.name AS region_city_label
                    FROM inp i
                    LEFT JOIN area a_city ON a_city.id = i.begin_area;
                    """
                )
                cur.execute("CREATE INDEX ON region_city (artist_id);")
                cur.execute("ANALYZE region_city;")

        # -------------------------------------------------------------
        # 4. Workset: artist -> works (authored or via performance)
        # -------------------------------------------------------------
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS workset;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE workset AS
                WITH authored AS (
                  SELECT i.id AS artist_id, w.id AS work_id, w.name AS work_name
                  FROM inp i
                  JOIN l_artist_work law ON law.entity0 = i.id
                  JOIN link lk ON lk.id = law.link
                  JOIN role_ids lt ON lt.id = lk.link_type
                  JOIN work w ON w.id = law.entity1
                ),
                performed AS (
                  SELECT i.id AS artist_id, w.id AS work_id, w.name AS work_name
                  FROM inp i
                  JOIN artist_credit_name acn ON acn.artist = i.id
                  JOIN artist_credit ac ON ac.id = acn.artist_credit
                  JOIN recording r ON r.artist_credit = ac.id
                  JOIN l_recording_work lrw ON lrw.entity0 = r.id
                  JOIN work w ON w.id = lrw.entity1
                )
                SELECT artist_id, work_id, MIN(work_name) AS work_name
                FROM (
                  SELECT * FROM authored
                  UNION ALL
                  SELECT * FROM performed
                ) x
                GROUP BY artist_id, work_id;
                """
            )
            cur.execute("CREATE INDEX ON workset (artist_id, work_id);")
            cur.execute("CREATE INDEX ON workset (work_id);")
            cur.execute("ANALYZE workset;")

        # -------------------------------------------------------------
        # 5. Optional: genres & first release date & collaborators
        # -------------------------------------------------------------
        if include_genres and (use_work_genre or use_work_tag):
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_genres;")
                if use_work_genre:
                    cur.execute(
                        """
                        CREATE UNLOGGED TABLE work_genres AS
                        SELECT w.artist_id,
                               wg.work AS work_id,
                               array_agg(DISTINCT g.name) AS genres
                        FROM workset w
                        JOIN work_genre wg ON wg.work = w.work_id
                        JOIN genre g ON g.id = wg.genre
                        GROUP BY w.artist_id, wg.work;
                        """
                    )
                else:
                    cur.execute(
                        """
                        CREATE UNLOGGED TABLE work_genres AS
                        SELECT w.artist_id,
                               wt.work AS work_id,
                               array_agg(DISTINCT t.name) AS genres
                        FROM workset w
                        JOIN work_tag wt ON wt.work = w.work_id
                        JOIN tag t ON t.id = wt.tag
                        GROUP BY w.artist_id, wt.work;
                        """
                    )
                cur.execute("CREATE INDEX ON work_genres (artist_id, work_id);")
                cur.execute("ANALYZE work_genres;")
        else:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_genres;")
                cur.execute(
                    "CREATE UNLOGGED TABLE work_genres (artist_id int, work_id int, genres text[]);"
                )

        if has_wfrd:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_dates;")
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE work_dates AS
                    SELECT w.artist_id,
                           wfrd.work_id,
                           wfrd.first_date
                    FROM workset w
                    JOIN work_first_release_date wfrd ON wfrd.work_id = w.work_id;
                    """
                )
                cur.execute("CREATE INDEX ON work_dates (artist_id, work_id);")
                cur.execute("ANALYZE work_dates;")
        else:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_dates;")
                cur.execute(
                    "CREATE UNLOGGED TABLE work_dates (artist_id int, work_id int, first_date date);"
                )

        if include_collaborators and has_wcr:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_collab;")
                cur.execute(
                    """
                    CREATE UNLOGGED TABLE work_collab AS
                    SELECT w.artist_id,
                           wcr.work_id,
                           wcr.collaborators
                    FROM workset w
                    JOIN work_contrib_roles wcr ON wcr.work_id = w.work_id;
                    """
                )
                cur.execute("CREATE INDEX ON work_collab (artist_id, work_id);")
                cur.execute("ANALYZE work_collab;")
        else:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS work_collab;")
                cur.execute(
                    "CREATE UNLOGGED TABLE work_collab (artist_id int, work_id int, collaborators jsonb);"
                )

        # -------------------------------------------------------------
        # 6. Aggregate works per artist into JSON
        # -------------------------------------------------------------
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS artist_works;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE artist_works AS
                SELECT
                  w.artist_id,
                  jsonb_agg(
                    jsonb_build_object(
                      'id', w.work_id,
                      'name', w.work_name,
                      'first_release_date',
                        CASE
                          WHEN d.first_date IS NULL THEN NULL
                          ELSE to_char(d.first_date, 'YYYY-MM-DD')
                        END,
                      'genres', COALESCE(g.genres, ARRAY[]::text[]),
                      'collaborators', COALESCE(c.collaborators, '[]'::jsonb)
                    )
                  ) AS works
                FROM workset w
                LEFT JOIN work_dates   d ON d.artist_id = w.artist_id AND d.work_id = w.work_id
                LEFT JOIN work_genres  g ON g.artist_id = w.artist_id AND g.work_id = w.work_id
                LEFT JOIN work_collab  c ON c.artist_id = w.artist_id AND c.work_id = w.work_id
                GROUP BY w.artist_id;
                """
            )
            cur.execute("CREATE INDEX ON artist_works (artist_id);")
            cur.execute("ANALYZE artist_works;")

        # -------------------------------------------------------------
        # 7. Recording + release + label info
        # -------------------------------------------------------------
        with conn.cursor() as cur:
            # raw per-recording/per-release rows
            cur.execute("DROP TABLE IF EXISTS work_rec_rel_raw;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE work_rec_rel_raw AS
                SELECT DISTINCT
                  w.artist_id,
                  w.work_id,
                  r.id        AS rec_id,
                  r.name      AS rec_name,
                  r.length    AS rec_length,
                  rel.id      AS rel_id,
                  rel.name    AS rel_name,
                  rev.rel_date AS rel_date,
                  rg.id       AS rg_id,
                  lb.gid      AS label_gid,
                  lb.name     AS label_name
                FROM workset w
                JOIN l_recording_work lrw ON lrw.entity1 = w.work_id
                JOIN recording r ON r.id = lrw.entity0
                JOIN track t ON t.recording = r.id
                JOIN medium m ON m.id = t.medium
                JOIN release rel ON rel.id = m.release
                LEFT JOIN LATERAL (
                  SELECT
                    CASE
                      WHEN re.date_year IS NULL THEN NULL::date
                      ELSE make_date(
                        re.date_year,
                        COALESCE(re.date_month, 1),
                        COALESCE(re.date_day, 1)
                      )
                    END AS rel_date
                  FROM release_event re
                  WHERE re.release = rel.id
                  ORDER BY re.date_year, re.date_month, re.date_day
                  LIMIT 1
                ) rev ON TRUE
                LEFT JOIN release_group rg ON rg.id = rel.release_group
                LEFT JOIN release_label rl ON rl.release = rel.id
                LEFT JOIN label lb ON lb.id = rl.label;
                """
            )
            cur.execute("CREATE INDEX ON work_rec_rel_raw (artist_id, work_id);")
            cur.execute("CREATE INDEX ON work_rec_rel_raw (artist_id, rec_id);")
            cur.execute("CREATE INDEX ON work_rec_rel_raw (artist_id, rel_id);")
            cur.execute("ANALYZE work_rec_rel_raw;")

            # recordings per artist
            cur.execute("DROP TABLE IF EXISTS artist_recordings;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE artist_recordings AS
                SELECT
                  artist_id,
                  jsonb_agg(
                    DISTINCT jsonb_build_object(
                      'id', rec_id,
                      'name', rec_name,
                      'length_ms', rec_length,
                      'work_id', work_id,
                      'release_id', rel_id
                    )
                  ) AS recordings
                FROM work_rec_rel_raw
                WHERE rec_id IS NOT NULL
                GROUP BY artist_id;
                """
            )
            cur.execute("CREATE INDEX ON artist_recordings (artist_id);")
            cur.execute("ANALYZE artist_recordings;")

            # releases per artist (first flatten to per-release)
            cur.execute("DROP TABLE IF EXISTS artist_release_flat;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE artist_release_flat AS
                SELECT
                  artist_id,
                  rel_id,
                  max(rel_name) AS rel_name,
                  min(rel_date) AS rel_date,
                  max(rg_id)    AS rg_id,
                  array_remove(array_agg(DISTINCT label_gid), NULL)  AS label_ids,
                  array_remove(array_agg(DISTINCT label_name), NULL) AS label_names
                FROM work_rec_rel_raw
                WHERE rel_id IS NOT NULL
                GROUP BY artist_id, rel_id;
                """
            )
            cur.execute("CREATE INDEX ON artist_release_flat (artist_id, rel_id);")
            cur.execute("ANALYZE artist_release_flat;")

            cur.execute("DROP TABLE IF EXISTS artist_releases;")
            cur.execute(
                """
                CREATE UNLOGGED TABLE artist_releases AS
                SELECT
                  artist_id,
                  jsonb_agg(
                    jsonb_build_object(
                      'id', rel_id,
                      'title', rel_name,
                      'date',
                        CASE
                          WHEN rel_date IS NULL THEN NULL
                          ELSE to_char(rel_date, 'YYYY-MM-DD')
                        END,
                      'release_group_id', rg_id,
                      'label_ids', COALESCE(label_ids, ARRAY[]::uuid[]),
                      'label_names', COALESCE(label_names, ARRAY[]::text[])
                    )
                  ) AS releases
                FROM artist_release_flat
                GROUP BY artist_id;
                """
            )
            cur.execute("CREATE INDEX ON artist_releases (artist_id);")
            cur.execute("ANALYZE artist_releases;")

        # -------------------------------------------------------------
        # 8. Final SELECT: one row per artist with JSON blobs
        # -------------------------------------------------------------
        # We pull everything at once (5000 rows max), then yield in cohort order.
        final_sql = """
        SELECT
          i.gid        AS mbid,
          i.name       AS artist_name,
          ar.role_list AS roles_array,
          {country_sel}    AS country_label,
          {rc_sel}         AS region_city_label,
          COALESCE(aw.works, '[]'::jsonb)        AS works_json,
          COALESCE(arcd.recordings, '[]'::jsonb) AS recordings_json,
          COALESCE(arls.releases, '[]'::jsonb)   AS releases_json
        FROM inp i
        LEFT JOIN artist_roles     ar   ON ar.artist_id = i.id
        {country_join}
        {rc_join}
        LEFT JOIN artist_works     aw   ON aw.artist_id = i.id
        LEFT JOIN artist_recordings arcd ON arcd.artist_id = i.id
        LEFT JOIN artist_releases  arls ON arls.artist_id = i.id
        ;
        """

        country_sel = "c.country_label" if include_country else "NULL::text"
        rc_sel = "rc.region_city_label" if include_region_city else "NULL::text"

        country_join = "LEFT JOIN country c ON c.artist_id = i.id" if include_country else ""
        rc_join = "LEFT JOIN region_city rc ON rc.artist_id = i.id" if include_region_city else ""

        final_sql = final_sql.format(
            country_sel=country_sel,
            rc_sel=rc_sel,
            country_join=country_join,
            rc_join=rc_join,
        )

        artists_by_mbid: dict[str, dict] = {}

        with conn.cursor(row_factory=dict_row) as cur:
            for row in cur.stream(final_sql, size=itersize):
                mbid = str(row["mbid"])
                roles = row["roles_array"] or []
                country = row["country_label"]
                region_city = row["region_city_label"]

                works = list(row["works_json"] or [])
                recordings = list(row["recordings_json"] or [])
                releases = list(row["releases_json"] or [])

                artists_by_mbid[mbid] = {
                    "mbid": mbid,
                    "artist_name": row["artist_name"],
                    "roles": ", ".join(roles) if roles else "",
                    "country": country,
                    "region_city": region_city,
                    "works": works,
                    "recordings": recordings,
                    "releases": releases,
                }

        # -------------------------------------------------------------
        # 9. Emit in input order, including "empty" artists
        # -------------------------------------------------------------
        for mbid in cohort:
            data = artists_by_mbid.get(mbid)
            if data is not None:
                yield data
            else:
                yield {
                    "mbid": mbid,
                    "artist_name": None,
                    "roles": "",
                    "country": None,
                    "region_city": None,
                    "works": [],
                    "recordings": [],
                    "releases": [],
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
    """).format(artist_mbid=sql.Literal(artist_mbid))

    with _connect() as conn:
        with conn.cursor() as cset:
            # Optional work_mem tuning
            if work_mem:
                cset.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(work_mem)))

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
    "youtube.com", "youtu.be", "music.youtube.com",
    # SoundCloud
    "soundcloud.com",
    # Spotify
    "open.spotify.com", "spotify.com",
    # Last.fm
    "last.fm",
    # Twitter / X
    "twitter.com", "x.com",
    # Instagram (user list had 'instagramr' — assuming Instagram)
    "instagram.com",
]


def stream_artist_resources(
    mbids: Iterable[str],
    work_mem: str | None = None,
    batch_size: int = 10_000,
    include_empty: bool = False,
) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    """
    Stream (artist_mbid, resources[]) for the given artists.

    resources[] = [{ 'url': str, 'link_type': str }]

    Behavior:
    - Returns *all* URL resources for each artist; no platform substring filtering.
    - If include_empty is False (default), artists with no URL resources are skipped.
    - If include_empty is True, artists are returned even if they have no URLs
      (their resources list will be empty).

    Notes:
    - psycopg3-safe: no % wildcards in the SQL text; all parameters via %s.
    - Uses a server-side cursor to avoid loading all rows into memory.
    - Uses a.gid = ANY(%s) to pass batched MBIDs.
    """

    # No LIKE / platform filters: fetch all URLs for the given artists
    query = """
        SELECT
            a.gid AS artist_mbid,
            lt.name AS link_type,
            u.url AS url
        FROM artist a
        LEFT JOIN l_artist_url lau ON lau.entity0 = a.id
        LEFT JOIN link l ON l.id = lau.link
        LEFT JOIN link_type lt ON lt.id = l.link_type
        LEFT JOIN url u ON u.id = lau.entity1
        WHERE a.gid = ANY(%s)
        ORDER BY a.gid
    """

    with _connect() as conn:
        # SET LOCAL applies within the current transaction
        if work_mem:
            with conn.cursor() as c:
                c.execute("SET LOCAL work_mem = %s", (work_mem,))

        for batch in _batched(mbids, batch_size):
            # Server-side (named) cursor for streaming rows
            with conn.cursor(name="artist_resource_stream") as cur:
                # Only parameter is the array of MBIDs
                cur.execute(query, (batch,))

                current_artist: str | None = None
                current_resources: List[Dict[str, str]] = []

                for artist_mbid, link_type, url in cur:
                    if artist_mbid != current_artist:
                        # emit previous artist if present
                        if current_artist is not None and (
                            include_empty or current_resources
                        ):
                            yield current_artist, current_resources

                        current_artist = artist_mbid
                        current_resources = []

                    # Only add a resource if there's actually a URL
                    if url is not None:
                        current_resources.append(
                            {
                                "link_type": link_type,
                                "url": url,
                            }
                        )

                # emit the final artist from this batch
                if current_artist is not None and (
                    include_empty or current_resources
                ):
                    yield current_artist, current_resources