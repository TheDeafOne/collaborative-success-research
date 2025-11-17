import psycopg

def get_all_artist_mbids():
    
    with psycopg.connect("dbname=musicbrainz_db user=musicbrainz password=musicbrainz host=localhost") as conn:
        with conn.cursor() as cur, open("artist_mbids.txt", "w") as f:
            cur.execute("SELECT gid FROM artist;")
            for (gid,) in cur:
                f.write(str(gid) + "\n")

import json
import time
from pathlib import Path
from datetime import datetime

def dump_artists_works_jsonl(
    mbids,
    out_dir,
    getter,
    *,
    gb_client=None,              # e.g. your mb module
    mb_stream_kwargs=None,       # e.g. {"set_work_mem": "1GB"}
    batch_size=1000,
    shard_max_mb=50,
    prefix=None,
    retries=1,
    backoff=1.5,
):
    """
    Streams artist works for many MBIDs and writes to multiple JSONL shards.
    Logs errors to errors.jsonl and continues.

    Example:
        dump_artists_works_jsonl(top_artists, "data/artists_works")
    """
    if gb_client is None:
        gb_client = globals().get("gb")
        if gb_client is None:
            raise RuntimeError("Provide gb_client or define a global `gb`.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not prefix:
        prefix = f"artists_works_{datetime.utcnow():%Y-%m-%d}"

    shard_max_bytes = shard_max_mb * 1024 * 1024
    part, total_rows = 0, 0
    cur_file, cur_bytes, cur_rows = None, 0, 0
    errors_path = out / "errors.jsonl"

    def open_new_file():
        nonlocal part, cur_file, cur_bytes, cur_rows
        if cur_file:
            cur_file.close()
        part += 1
        path = out / f"{prefix}_part-{part:04d}.jsonl"
        cur_file = open(path, "w", encoding="utf-8")
        cur_bytes, cur_rows = 0, 0
        return path

    def write_obj(obj):
        nonlocal cur_bytes, cur_rows
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        cur_file.write(line)
        cur_bytes += len(line.encode("utf-8"))
        cur_rows += 1

    open_new_file()

    with open(errors_path, "a", encoding="utf-8") as errlog:
        for i in range(0, len(mbids), batch_size):
            errlog.write(json.dumps({
                "artist_number": i,
            }) + "\n")
            errlog.flush()
            batch = mbids[i:i + batch_size]
            for attempt in range(retries + 1):
                try:
                    for obj in getter(batch):
                        if cur_bytes >= shard_max_bytes:
                            open_new_file()
                        write_obj(obj)
                        total_rows += 1
                    break  # success, move to next batch
                except Exception as e:
                    errlog.write(json.dumps({
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "batch_start": i,
                        "batch_size": len(batch),
                        "attempt": attempt,
                        "error": str(e),
                    }) + "\n")
                    errlog.flush()
                    if attempt < retries:
                        time.sleep(backoff * (2 ** attempt))
                    else:
                        break

    if cur_file:
        cur_file.close()

    print(f"Done. {total_rows} rows written across {part} files. Errors logged to {errors_path}")

if __name__ == "__main__":
    from experiments import graph_builder as gb
    with open('./data/all_artist_mbids.txt', 'r') as f:
        artist_mbids = f.read().splitlines()
    
    dump_artists_works_jsonl(artist_mbids, './data/artist_collab_data', gb.stream_artists_songs_by_mbids)