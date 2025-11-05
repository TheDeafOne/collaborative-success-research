from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Optional
from uuid import UUID
import os

import psycopg
from psycopg.rows import dict_row
from psycopg import sql

AUTHOR_ROLES = ("composer", "lyricist", "writer", "librettist")

def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        return cur.fetchone()[0] is not None

def ensure_perf_indexes() -> None:
    with _connect() as conn, conn.cursor() as cur:
        stmts = [
            "CREATE INDEX IF NOT EXISTS idx_artist_gid ON artist (gid);",
            "CREATE INDEX IF NOT EXISTS idx_l_artist_work_e0_e1 ON l_artist_work (entity0, entity1);",
            "CREATE INDEX IF NOT EXISTS idx_link_link_type ON link (link_type);",
            "CREATE INDEX IF NOT EXISTS idx_l_recording_work_e0_e1 ON l_recording_work (entity0, entity1);",
            "CREATE INDEX IF NOT EXISTS idx_artist_credit_name_artist ON artist_credit_name (artist);",
            "CREATE INDEX IF NOT EXISTS idx_recording_artist_credit ON recording (artist_credit);",
            "CREATE INDEX IF NOT EXISTS idx_track_recording ON track (recording);",
            "CREATE INDEX IF NOT EXISTS idx_medium_release ON medium (release);",
            "CREATE INDEX IF NOT EXISTS idx_release_country_rel ON release_country (release, date_year, date_month, date_day);",
            "CREATE INDEX IF NOT EXISTS idx_release_unk_rel ON release_unknown_country (release, date_year, date_month, date_day);",
            "CREATE INDEX IF NOT EXISTS idx_artist_alias_artist ON artist_alias (artist);",
        ]
        for s in stmts:
            cur.execute(s)

        # Optional: only index work_genre if it exists
        if _table_exists(conn, "work_genre"):
            cur.execute("CREATE INDEX IF NOT EXISTS idx_work_genre_work ON work_genre (work);")

        conn.commit()


def ensure_work_genres_resolved_mv() -> None:
    """
    Build MV work_genres_resolved(work_id, genres text[]), pulling genres/tags from:
    - work_genre/genre or work_tag/tag
    - recording_genre/genre or recording_tag/tag (via work -> recording)
    - release_group_genre/genre or release_group_tag/tag (via work -> rec -> track -> medium -> release -> release_group)
    Only includes the sources that actually exist in your DB.
    """
    with _connect() as conn, conn.cursor() as cur:
        has_work_genre   = _table_exists(conn, "work_genre") and _table_exists(conn, "genre")
        has_work_tag     = _table_exists(conn, "work_tag")   and _table_exists(conn, "tag")
        has_rec_genre    = _table_exists(conn, "recording_genre") and _table_exists(conn, "genre")
        has_rec_tag      = _table_exists(conn, "recording_tag")   and _table_exists(conn, "tag")
        has_rg_genre     = _table_exists(conn, "release_group_genre") and _table_exists(conn, "genre")
        has_rg_tag       = _table_exists(conn, "release_group_tag")   and _table_exists(conn, "tag")

        # Build the UNION body based on what exists
        unions: list[str] = []

        if has_work_genre:
            unions.append("""
            SELECT wg.work AS work_id, g.name AS genre_name
            FROM work_genre wg JOIN genre g ON g.id = wg.genre
            """)

        if has_work_tag:
            unions.append("""
            SELECT wt.work AS work_id, t.name AS genre_name
            FROM work_tag wt JOIN tag t ON t.id = wt.tag
            """)

        if has_rec_genre:
            unions.append("""
            SELECT lrw.entity1 AS work_id, g.name AS genre_name
            FROM l_recording_work lrw
            JOIN recording_genre rg ON rg.recording = lrw.entity0
            JOIN genre g ON g.id = rg.genre
            """)

        if has_rec_tag:
            unions.append("""
            SELECT lrw.entity1 AS work_id, t.name AS genre_name
            FROM l_recording_work lrw
            JOIN recording_tag rt ON rt.recording = lrw.entity0
            JOIN tag t ON t.id = rt.tag
            """)

        if has_rg_genre:
            unions.append("""
            SELECT lrw.entity1 AS work_id, g.name AS genre_name
            FROM l_recording_work lrw
            JOIN track   t  ON t.recording = lrw.entity0
            JOIN medium  m  ON m.id = t.medium
            JOIN release r  ON r.id = m.release
            JOIN release_group rg ON rg.id = r.release_group
            JOIN release_group_genre rgg ON rgg.release_group = rg.id
            JOIN genre g ON g.id = rgg.genre
            """)

        if has_rg_tag:
            unions.append("""
            SELECT lrw.entity1 AS work_id, t2.name AS genre_name
            FROM l_recording_work lrw
            JOIN track   tr  ON tr.recording = lrw.entity0
            JOIN medium  md  ON md.id = tr.medium
            JOIN release rl  ON rl.id = md.release
            JOIN release_group rg ON rg.id = rl.release_group
            JOIN release_group_tag rgt ON rgt.release_group = rg.id
            JOIN tag t2 ON t2.id = rgt.tag
            """)

        if not unions:
            # No genre/tag sources at all -> create a tiny MV that yields empty arrays
            cur.execute("DROP MATERIALIZED VIEW IF EXISTS work_genres_resolved;")
            cur.execute("""
            CREATE MATERIALIZED VIEW work_genres_resolved AS
            SELECT w.id AS work_id, ARRAY[]::text[] AS genres
            FROM work w;
            """)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wgr_work_id ON work_genres_resolved (work_id);")
            conn.commit()
            return

        union_sql = "\nUNION ALL\n".join(unions)

        cur.execute("DROP MATERIALIZED VIEW IF EXISTS work_genres_resolved;")
        cur.execute(f"""
        CREATE MATERIALIZED VIEW work_genres_resolved AS
        WITH raw AS (
          {union_sql}
        ),
        cleaned AS (
          SELECT work_id, lower(trim(genre_name)) AS g
          FROM raw
          WHERE genre_name IS NOT NULL AND genre_name <> ''
        )
        SELECT work_id,
               array_agg(DISTINCT g ORDER BY g) AS genres
        FROM cleaned
        GROUP BY work_id;
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wgr_work_id ON work_genres_resolved (work_id);")
        conn.commit()

# --- materialized views: create + refresh ------------------------------------
def ensure_work_first_release_date_mv() -> None:
    """
    MV: earliest release-event date per Work (via recording→track→medium→release).
    """
    create_mv = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS work_first_release_date AS
    WITH dates AS (
      SELECT lrw.entity1 AS work_id,
             make_date(rc.date_year, COALESCE(rc.date_month,1), COALESCE(rc.date_day,1)) AS d
      FROM l_recording_work lrw
      JOIN track   t ON t.recording = lrw.entity0
      JOIN medium  m ON m.id = t.medium
      JOIN release r ON r.id = m.release
      JOIN release_country rc ON rc.release = r.id
      WHERE rc.date_year IS NOT NULL
      UNION ALL
      SELECT lrw.entity1 AS work_id,
             make_date(ru.date_year, COALESCE(ru.date_month,1), COALESCE(ru.date_day,1)) AS d
      FROM l_recording_work lrw
      JOIN track   t ON t.recording = lrw.entity0
      JOIN medium  m ON m.id = t.medium
      JOIN release r ON r.id = m.release
      JOIN release_unknown_country ru ON ru.release = r.id
      WHERE ru.date_year IS NOT NULL
    )
    SELECT work_id, MIN(d) AS first_date
    FROM dates
    GROUP BY work_id;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(create_mv)
        # UNIQUE index is required for REFRESH CONCURRENTLY
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wfrd_work_id ON work_first_release_date (work_id);")
        conn.commit()

def refresh_work_first_release_date_mv(concurrently: bool = True) -> None:
    with _connect() as conn, conn.cursor() as cur:
        if concurrently:
            # REFRESH CONCURRENTLY must run outside a transaction block
            conn.autocommit = True
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY work_first_release_date;")
        else:
            cur.execute("REFRESH MATERIALIZED VIEW work_first_release_date;")
            conn.commit()

def ensure_work_contrib_roles_mv() -> None:
    """
    MV: collaborators (authors) per Work with their roles, as a prebuilt JSON array.
    Safe to call multiple times.
    """
    create_mv = """
    DROP MATERIALIZED VIEW IF EXISTS work_contrib_roles;

    CREATE MATERIALIZED VIEW work_contrib_roles AS
    WITH role_ids AS (
      SELECT id
      FROM link_type
      WHERE name IN ('composer','lyricist','writer','librettist')
    ),
    roles_per_work AS (
      SELECT
        law.entity1        AS work_id,
        ar.gid             AS artist_mbid,
        ar.name            AS artist_name,
        array_agg(DISTINCT lt_name.name ORDER BY lt_name.name) AS roles
      FROM l_artist_work law
      JOIN link lk            ON lk.id = law.link
      JOIN role_ids rids      ON lk.link_type = rids.id          -- filter by allowed role ids
      JOIN link_type lt_name  ON lt_name.id = lk.link_type       -- fetch the role *names*
      JOIN artist ar          ON ar.id = law.entity0
      GROUP BY law.entity1, ar.gid, ar.name
    )
    SELECT
      work_id,
      jsonb_agg(
        jsonb_build_object(
          'name',  artist_name,
          'mbid',  artist_mbid,
          'roles', roles
        )
        ORDER BY artist_name
      ) AS collaborators
    FROM roles_per_work
    GROUP BY work_id;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(create_mv)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wcr_work_id ON work_contrib_roles (work_id);")
        conn.commit()



def refresh_work_contrib_roles_mv(concurrently: bool = True) -> None:
    with _connect() as conn, conn.cursor() as cur:
        if concurrently:
            conn.autocommit = True
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY work_contrib_roles;")
        else:
            cur.execute("REFRESH MATERIALIZED VIEW work_contrib_roles;")
            conn.commit()

def setup_musicbrainz_speedups() -> None:
    """
    Call once after ingest/refresh: indexes + both MVs.
    """
    # ensure_perf_indexes()
    # ensure_work_first_release_date_mv()
    ensure_work_contrib_roles_mv()


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("MUSICBRAINZ_DB_HOST", "localhost"),
        port=int(os.getenv("MUSICBRAINZ_DB_PORT", "5432")),
        dbname=os.getenv("MUSICBRAINZ_DB_NAME", "musicbrainz_db"),
        user=os.getenv("MUSICBRAINZ_DB_USER", "musicbrainz"),
        password=os.getenv("MUSICBRAINZ_DB_PASSWORD", "musicbrainz"),
    )

def _coerce_uuid(s: Any) -> Optional[str]:
    try:
        s = str(s).strip()
        return str(UUID(s)) if s else None
    except Exception:
        return None

def _dedupe(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        return cur.fetchone()[0] is not None



def stream_artists_songs_by_mbids(
    artist_mbids,
    *,
    include_aliases: bool = False,
    include_country: bool = False,
    include_region_city: bool = False,
    include_perf_roles: bool = False,       # toggles l_artist_recording path in roles only
    include_genres: bool = False,           # needs work_genre/genre or work_tag/tag
    include_collaborators: bool = True,     # needs work_contrib_roles MV
    input_path: str = "temp_table",
    work_mem: str = "512MB",
    disable_jit: bool = True,
    itersize: int = 100_000,
):
    # ---- normalize inputs (same as before) ----
    raw = list(artist_mbids or [])
    if not raw: return
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
                seen.add(x); out.append(x)
        return out
    valid = [u for u in (_coerce_uuid(x) for x in raw) if u]
    cohort = _dedupe(valid)
    if not cohort: return

    from psycopg import sql
    from psycopg.rows import dict_row

    with _connect() as conn:
        with conn.cursor() as cset:
            if work_mem: cset.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(work_mem)))
            if disable_jit: cset.execute("SET LOCAL jit = off")
            cset.execute("SET LOCAL synchronous_commit = off")
            cset.execute("SET LOCAL max_parallel_workers_per_gather = 8")
            cset.execute("SET LOCAL parallel_leader_participation = on")

        # Build inp (UNLOGGED for speed)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS inp;")
            if input_path == "temp_table":
                cur.execute("CREATE TEMP TABLE tmp_input_gid (gid uuid PRIMARY KEY) ON COMMIT DROP;")
                with cur.copy("COPY tmp_input_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cur.execute("""
                    CREATE UNLOGGED TABLE inp AS
                    SELECT a.id, a.gid, a.name, a.area, a.begin_area
                    FROM artist a JOIN tmp_input_gid i ON i.gid = a.gid;
                """)
            else:
                cur.execute("""
                    CREATE UNLOGGED TABLE inp AS
                    SELECT a.id, a.gid, a.name, a.area, a.begin_area
                    FROM artist a
                    JOIN unnest(%s::uuid[]) x(gid) ON a.gid = x.gid;
                """, (cohort,))
            cur.execute("CREATE INDEX ON inp (id); ANALYZE inp;")

        # Detect optional tables/MVs
        def _exists(tab):
            with conn.cursor() as c:
                c.execute("SELECT to_regclass(%s)", (tab,))
                return c.fetchone()[0] is not None
        has_wfrd = _exists("work_first_release_date")
        has_wcr  = _exists("work_contrib_roles")
        use_work_genre = _exists("work_genre") and _exists("genre")
        use_work_tag   = _exists("work_tag")   and _exists("tag")

        # Build the SQL parts conditionally
        role_ids_cte = "WITH role_ids AS (SELECT id, name FROM link_type WHERE name = ANY (%%s::text[]))"

        aliases_cte = """
        ,aliases AS (
          SELECT aa.artist, array_agg(DISTINCT aa.name ORDER BY aa.name) AS alias_names
          FROM artist_alias aa JOIN inp i ON i.id = aa.artist
          GROUP BY aa.artist
        )""" if include_aliases else ""

        country_cte = """
        ,country AS (
          SELECT i.id AS artist_id, a1.name AS country_label
          FROM inp i LEFT JOIN area a1 ON a1.id = i.area
        )""" if include_country else ""

        rc_cte = """
        ,region_city AS (
          SELECT i.id AS artist_id, a_city.name AS region_city_label
          FROM inp i LEFT JOIN area a_city ON a_city.id = i.begin_area
        )""" if include_region_city else ""

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
        perf_roles_cte = """
        ,perf_roles AS (
          SELECT i.id AS artist_id, array_agg(DISTINCT COALESCE(lt2.name,'performer')) AS roles
          FROM inp i
          JOIN l_artist_recording lar ON lar.entity0 = i.id
          JOIN link lk2 ON lk2.id = lar.link
          LEFT JOIN link_type lt2 ON lt2.id = lk2.link_type
          GROUP BY i.id
        )""" if include_perf_roles else ""
        artist_roles_cte = """
        ,artist_roles AS (
          SELECT artist_id, array_agg(DISTINCT role) AS role_list
          FROM (
            SELECT artist_id, unnest(roles) AS role FROM author_roles
            %s
          ) u
          GROUP BY artist_id
        )""" % (("UNION ALL SELECT artist_id, unnest(roles) FROM perf_roles" if include_perf_roles else ""))

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
        date_join  = "LEFT JOIN work_first_release_date wfrd ON wfrd.work_id = d.work_id" if has_wfrd else ""
        collab_join = "LEFT JOIN work_contrib_roles wcr ON wcr.work_id = d.work_id" if (include_collaborators and has_wcr) else ""

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
        aliases_sel   = "al.alias_names" if include_aliases else "NULL::text[]"
        country_sel   = "c.country_label" if include_country else "NULL::text"
        rc_sel        = "rc.region_city_label" if include_region_city else "NULL::text"
        date_sel      = "to_char(wfrd.first_date, 'YYYY-MM-DD')" if has_wfrd else "NULL::text"
        genres_sel    = "wg.genres" if (include_genres and (use_work_genre or use_work_tag)) else "NULL::text[]"
        collab_sel    = "wcr.collaborators" if (include_collaborators and has_wcr) else "'[]'::jsonb"

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
                        "works": []
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
                    b["works"].append({
                        "id": row["work_id"],
                        "name": row["work_name"],
                        "release_date": row["first_date"],
                        "genres": row["genres_array"] or [],
                        "collaborators": list(row["collaborators_json"] or []),
                    })

            # yield in input order
            seen = set()
            for mbid in cohort:
                if mbid in buckets:
                    seen.add(mbid); yield buckets[mbid]
                else:
                    yield {"mbid": mbid, "artist_name": None, "roles": "",
                           "aliases": [], "country": None, "region_city": None, "works": []}