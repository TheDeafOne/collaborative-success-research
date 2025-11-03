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
        cur.execute(f"REFRESH MATERIALIZED VIEW {'CONCURRENTLY' if concurrently else ''} work_contrib_mbids;")
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_work_contrib_mbids_work_id ON work_contrib_mbids (work_id);")
        conn.commit()


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

def _chunked(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def stream_artists_songs_by_mbids_fast(
    artist_mbids: Sequence[Any],
    *,
    input_path: Literal["unnest", "temp_table"] = "unnest",
    set_work_mem: str = "2GB",
    disable_jit: bool = False,
    max_parallel_workers_per_gather: int = 8,
    on_invalid: Literal["empty", "skip", "error"] = "empty",
) -> Iterator[Dict[str, Any]]:
    """
    Fast path using a parallel-friendly query + LATERAL contributor lookup.

    Emits one object per *input token* (invalids can be empty per policy):
    {"artist_mbid": <uuid str>, "works": [{"song_title": str, "contributor_mbids": [uuid str, ...]}, ...]}
    """
    # ---- helpers ----
    def _coerce_uuid_str_or_none(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return str(UUID(s))
        except Exception:
            return None

    def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    # ---- normalize inputs ----
    if not artist_mbids:
        return

    valid_canonical: List[str] = []
    invalid_originals: List[str] = []
    for tok in artist_mbids:
        coerced = _coerce_uuid_str_or_none(tok)
        (valid_canonical if coerced is not None else invalid_originals).append(coerced or str(tok))

    cohort = _dedupe_preserve_order(valid_canonical)
    invalid_originals = _dedupe_preserve_order(invalid_originals)

    if invalid_originals and on_invalid == "error":
        raise ValueError(f"{len(invalid_originals)} invalid MBIDs; examples: {invalid_originals[:5]}")

    if not cohort:
        if on_invalid == "empty":
            for bad in invalid_originals:
                yield {"artist_mbid": bad, "works": []}
        return

    AUTHOR_ROLE_NAMES = ("composer", "lyricist", "writer", "librettist")

    # Build the artist set (ta) via UNNEST or a temp table
    with _connect() as conn:
        with conn.cursor() as cur:
            # session knobs (measure each in your env)
            if set_work_mem:
                cur.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(set_work_mem)))
            cur.execute(sql.SQL("SET LOCAL jit = {}").format(sql.SQL("off" if disable_jit else "on")))
            cur.execute("SET LOCAL synchronous_commit = off")
            if max_parallel_workers_per_gather:
                cur.execute(
                    sql.SQL("SET LOCAL max_parallel_workers_per_gather = {}").format(
                        sql.Literal(int(max_parallel_workers_per_gather))
                    )
                )
                cur.execute("SET LOCAL parallel_leader_participation = on")

            # Build ta
            if input_path == "temp_table":
                cur.execute("CREATE TEMP TABLE tmp_input_gid (gid uuid PRIMARY KEY) ON COMMIT DROP;")
                # Use psycopg COPY for speed
                with cur.copy("COPY tmp_input_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cur.execute("ANALYZE tmp_input_gid;")
                ta_src = "SELECT a.id, a.gid FROM artist a JOIN tmp_input_gid i ON i.gid = a.gid"
                params = {"roles": list(AUTHOR_ROLE_NAMES)}
            else:
                ta_src = "SELECT a.id, a.gid FROM artist a JOIN unnest(%(artist_gids)s::uuid[]) x(gid) ON a.gid = x.gid"
                params = {"roles": list(AUTHOR_ROLE_NAMES), "artist_gids": cohort}

            query = f"""
            WITH role_ids AS (
              SELECT id FROM link_type WHERE name = ANY (%(roles)s::text[])
            ),
            ta AS (
              {ta_src}
            ),
            rw AS (
              -- authored
              SELECT ta.id AS artist_id, w.id AS work_id, w.name AS work_name
              FROM ta
              JOIN l_artist_work law ON law.entity0 = ta.id
              JOIN link lk ON lk.id = law.link AND lk.link_type IN (SELECT id FROM role_ids)
              JOIN work w ON w.id = law.entity1

              UNION ALL

              -- performed via recording
              SELECT ta.id, w.id, w.name
              FROM ta
              JOIN artist_credit_name acn ON acn.artist = ta.id
              JOIN artist_credit ac       ON ac.id = acn.artist_credit
              JOIN recording r            ON r.artist_credit = ac.id
              JOIN l_recording_work lrw   ON lrw.entity0 = r.id
              JOIN work w                 ON w.id = lrw.entity1
            ),
            rw_dedup AS (
              SELECT artist_id, work_id, MIN(work_name) AS work_name
              FROM rw
              GROUP BY artist_id, work_id
            )
            SELECT
              ta.gid AS artist_mbid,
              COALESCE(
                jsonb_agg(
                  jsonb_build_object(
                    'song_title', rw_dedup.work_name,
                    'contributor_mbids', COALESCE(contrib.contributor_mbids, ARRAY[]::uuid[])
                  )
                ) FILTER (WHERE rw_dedup.work_name IS NOT NULL),
                '[]'::jsonb
              ) AS works
            FROM ta
            LEFT JOIN rw_dedup ON rw_dedup.artist_id = ta.id
            LEFT JOIN LATERAL (
              SELECT array_agg(DISTINCT ar.gid) AS contributor_mbids
              FROM l_artist_work law
              JOIN link lk ON lk.id = law.link AND lk.link_type IN (SELECT id FROM role_ids)
              JOIN artist ar ON ar.id = law.entity0
              WHERE law.entity1 = rw_dedup.work_id
            ) contrib ON TRUE
            GROUP BY ta.gid
            """

            cur.execute(query, params)
            rows = cur.fetchall()

        for artist_mbid, works_json in rows:
            if works_json:
                yield {"artist_mbid": str(artist_mbid), "works": works_json}


# ---------------------------------------------
# Core: stream artists & authored works by MBIDs
# ---------------------------------------------
def stream_artists_songs_by_mbids(
    artist_mbids: Sequence[Any],
    *,
    itersize: int = 100_000,
    path: str = "temp_table",           # "temp_table" or "unnest"
    set_work_mem: Optional[str] = "512MB",
    disable_jit: bool = True,
    on_invalid: str = "empty",          # "empty" | "skip" | "error"
) -> Iterator[Dict[str, Any]]:
    """
    Yield exactly one object per input token (after de-dup). Works now include:
      1) authored works (composer/lyricist/writer/librettist), and
      2) works performed by recordings where the artist is in the recording's artist credit.
    """

    # ---- small local helpers so we don't touch other files ----
    import math
    from uuid import UUID

    def _coerce_uuid_str_or_none(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return str(UUID(s))
        except Exception:
            return None

    def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
        seen, out = set(), []
        for x in items:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    # ---- normalize inputs ----
    if not artist_mbids:
        return

    valid_canonical: List[str] = []
    invalid_originals: List[str] = []
    for tok in artist_mbids:
        coerced = _coerce_uuid_str_or_none(tok)
        (valid_canonical if coerced is not None else invalid_originals).append(coerced or str(tok))

    cohort = _dedupe_preserve_order([x for x in valid_canonical])
    invalid_originals = _dedupe_preserve_order(invalid_originals)

    if invalid_originals and on_invalid == "error":
        raise ValueError(f"{len(invalid_originals)} invalid MBIDs; examples: {invalid_originals[:5]}")

    AUTHOR_ROLE_NAMES = ("composer", "lyricist", "writer", "librettist")

    # If no valid UUIDs, just emit empties for invalids (if configured)
    if not cohort:
        if on_invalid == "empty":
            for bad in invalid_originals:
                yield {"artist_mbid": bad, "works": []}
        return

    with _connect() as conn:
        # Session knobs
        with conn.cursor() as cset:
            if set_work_mem:
                cset.execute(sql.SQL("SET LOCAL work_mem = {}").format(sql.Literal(set_work_mem)))
            if disable_jit:
                cset.execute("SET LOCAL jit = off")
            cset.execute("SET LOCAL synchronous_commit = off")

        # Resolve link_type ids once for author roles
        with conn.cursor() as c_lt:
            c_lt.execute("SELECT id FROM link_type WHERE name = ANY(%s)", (list(AUTHOR_ROLE_NAMES),))
            role_ids = [r[0] for r in c_lt.fetchall()]
            if not role_ids:
                raise RuntimeError("Authorship link types not found.")

        # Build tmp_artist from inputs
        if path != "unnest":
            with conn.cursor() as cprep:
                cprep.execute("CREATE TEMP TABLE tmp_input_gid (gid uuid PRIMARY KEY) ON COMMIT DROP")
                with cprep.copy("COPY tmp_input_gid (gid) FROM STDIN WITH (FORMAT text)") as cp:
                    cp.write("\n".join(cohort) + "\n")
                cprep.execute("ANALYZE tmp_input_gid")
                cprep.execute("""
                    CREATE TEMP TABLE tmp_artist AS
                    SELECT a.id, a.gid
                    FROM artist a
                    JOIN tmp_input_gid i ON i.gid = a.gid
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

        # Track which valid MBIDs actually exist in artist (for emitting empties later)
        with conn.cursor() as cgids:
            cgids.execute("SELECT gid FROM tmp_artist")
            present_in_artist = {str(r[0]) for r in cgids.fetchall()}

        # ------------------------------------------------------------------------------
        # KEY FIX: works "related" to the artist = AUTHORED  ∪  PERFORMED-VIA-RECORDING
        # ------------------------------------------------------------------------------

        with conn.cursor() as c1:
            c1.execute("""
                CREATE TEMP TABLE tmp_artist_related_works AS
                -- (A) Authored works
                SELECT DISTINCT
                    ta.id    AS artist_id,
                    ta.gid   AS artist_mbid,
                    w.id     AS work_id,
                    w.name   AS work_name
                FROM tmp_artist ta
                JOIN l_artist_work law ON law.entity0 = ta.id
                JOIN link lk           ON lk.id = law.link AND lk.link_type = ANY (%s::int[])
                JOIN work w            ON w.id = law.entity1

                UNION

                -- (B) Performed works via recordings:
                --     ta.id is in the recording's artist credit, and that recording links to the work.
                SELECT DISTINCT
                    ta.id    AS artist_id,
                    ta.gid   AS artist_mbid,
                    w.id     AS work_id,
                    w.name   AS work_name
                FROM tmp_artist ta
                JOIN artist_credit_name acn ON acn.artist = ta.id
                JOIN artist_credit ac       ON ac.id = acn.artist_credit
                JOIN recording r            ON r.artist_credit = ac.id
                JOIN l_recording_work lrw   ON lrw.entity0 = r.id
                JOIN work w                 ON w.id = lrw.entity1
            """, (role_ids,))
            c1.execute("CREATE INDEX ON tmp_artist_related_works (artist_id, work_id)")
            c1.execute("CREATE INDEX ON tmp_artist_related_works (work_id)")
            c1.execute("ANALYZE tmp_artist_related_works")

        # Contributor MBIDs per work (authors only, as before)
        with conn.cursor() as c2:
            c2.execute("""
                CREATE TEMP TABLE tmp_work_contrib_mbids AS
                SELECT
                    law.entity1 AS work_id,
                    array_agg(DISTINCT ar.gid) AS contributor_mbids
                FROM l_artist_work law
                JOIN link   lk ON lk.id = law.link AND lk.link_type = ANY (%s::int[])
                JOIN artist ar ON ar.id = law.entity0
                JOIN tmp_artist_related_works w ON w.work_id = law.entity1
                GROUP BY law.entity1
            """, (role_ids,))
            c2.execute("CREATE INDEX ON tmp_work_contrib_mbids (work_id)")
            c2.execute("ANALYZE tmp_work_contrib_mbids")

        # Stream: LEFT JOIN from tmp_artist so every present artist yields at least once
        sql_stream = """
            SELECT
                ta.gid AS artist_mbid,
                rw.work_name AS song_title,
                co.contributor_mbids
            FROM tmp_artist ta
            LEFT JOIN tmp_artist_related_works rw ON rw.artist_id = ta.id
            LEFT JOIN tmp_work_contrib_mbids   co ON co.work_id  = rw.work_id
        """

        with conn.cursor(name="mbz_song_stream_grouped", row_factory=dict_row) as cur:
            cur.itersize = itersize
            cur.execute(sql_stream)

            current_artist: Optional[str] = None
            bucket: List[Dict[str, Any]] = []

            for row in cur:
                artist_mbid = str(row["artist_mbid"])
                if current_artist is None:
                    current_artist = artist_mbid
                if artist_mbid != current_artist:
                    yield {"artist_mbid": current_artist, "works": bucket}
                    current_artist = artist_mbid
                    bucket = []

                if row["song_title"] is not None:
                    bucket.append({
                        "song_title": row["song_title"],
                        "contributor_mbids": row["contributor_mbids"] or [],
                    })

            if current_artist is not None:
                yield {"artist_mbid": current_artist, "works": bucket}

    # Emit empties for valid-but-missing artists (not in artist table)
    for mbid in cohort:
        if mbid not in present_in_artist:
            yield {"artist_mbid": mbid, "works": []}

    # Emit empties (or skip) for invalid tokens
    if on_invalid == "empty":
        for bad in invalid_originals:
            yield {"artist_mbid": bad, "works": []}


# --- worker executed in each process/thread ---
def _worker_stream_chunk(chunk: List[Any], stream_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Important: stream function itself now sanitizes inputs inside each worker.
    return list(stream_artists_songs_by_mbids(chunk, **stream_kwargs))


def parallel_stream_artists_songs_by_mbids(
    artist_mbids: Sequence[Any],
    *,
    # parallelism
    mode: str = "process",                 # "process" (best) or "thread" (lighter)
    max_workers: int | None = None,        # default: min(8, cpu_count)
    chunk_size: int = 1000,                # tune based on avg works/artist
    ordered: bool = False,                 # True = preserve input order of shards
    # kwargs forwarded to the single-node function
    itersize: int = 100_000,
    path: str = "temp_table",
    set_work_mem: str | None = "512MB",
    disable_jit: bool = True,
    on_invalid: Literal["empty", "skip", "error"] = "empty",
) -> Iterator[Dict[str, Any]]:
    """
    Parallel wrapper around `stream_artists_songs_by_mbids`.

    Yields exactly one object per input token (after de-dup), with the same invalid handling policy.
    """
    if not artist_mbids:
        return

    # De-dup the *raw* inputs before sharding (cheap, avoids repeated work across shards).
    # We do not validate here — workers handle validation and policy.
    seen_raw: set[str] = set()
    cohort_raw: List[Any] = []
    for x in artist_mbids:
        sx = str(x)
        if sx not in seen_raw:
            seen_raw.add(sx)
            cohort_raw.append(x)

    # choose executor
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)
    Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

    # kwargs forwarded to the inner streaming call
    stream_kwargs = dict(
        itersize=itersize,
        path=path,
        set_work_mem=set_work_mem,
        disable_jit=disable_jit,
        on_invalid=on_invalid,
    )

    # shard input
    shards = [(i, chunk) for i, chunk in enumerate(_chunked(cohort_raw, chunk_size)) if chunk]

    with Executor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_worker_stream_chunk, chunk, stream_kwargs): idx
            for idx, chunk in shards
        }

        if not ordered:
            for fut in as_completed(futures):
                for item in fut.result():
                    yield item
        else:
            next_idx = 0
            buffer: dict[int, List[Dict[str, Any]]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                buffer[idx] = fut.result()
                while next_idx in buffer:
                    for item in buffer.pop(next_idx):
                        yield item
                    next_idx += 1

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
