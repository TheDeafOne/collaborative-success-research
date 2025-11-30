#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from itertools import islice
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
sys.path.append(str(ROOT))
from apis import spotify as sp
from apis import musicbrainz as mb
from dotenv import load_dotenv
import os 
load_dotenv()
DATA_DIR = ROOT / "data"
ARTIST_IDS_PATH = DATA_DIR / "all_spotify_filtered_artist_ids.csv"
OUTPUT_DIR = DATA_DIR / "artist_full_data"  # JSONL shards will go here
SHARD_MAX_MB = 300  # cap per jsonl file

def chunked(iterable, size):
    """Yield lists of up to `size` items from `iterable`."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

def main() -> None:
    # Load input mapping
    try:
        artist_ids = pd.read_csv(ARTIST_IDS_PATH).to_numpy()
        mbid_to_spotify_id_map = dict(artist_ids)

    except Exception as exc:
        sys.exit(f"Failed to read {ARTIST_IDS_PATH}: {exc}")

    failures = 0
    processed = 0

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shard_max_bytes = SHARD_MAX_MB * 1024 * 1024
    part = 0
    cur_file = None
    cur_bytes = 0
    cur_rows = 0
    total_rows = 0

    def open_new_file():
        nonlocal part, cur_file, cur_bytes, cur_rows
        if cur_file:
            cur_file.close()
        part += 1
        path = OUTPUT_DIR / f"filtered_artist_data_part-{part:04d}.jsonl"
        cur_file = open(path, "w", encoding="utf-8")
        cur_bytes = 0
        cur_rows = 0
        print(f"Opened new shard: {path}")
        return path

    def write_obj(obj):
        nonlocal cur_bytes, cur_rows, total_rows
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        if cur_bytes + len(encoded) > shard_max_bytes:
            open_new_file()
        cur_file.write(line)
        cur_bytes += len(encoded)
        cur_rows += 1
        total_rows += 1

    # Start first shard
    open_new_file()

    
    all_mbids = list(mbid_to_spotify_id_map.keys())
    total = len(all_mbids)
    BATCH_SIZE = 2_000  # tweak as needed

    print(f"getting artist data for {total} artists (batch_size={BATCH_SIZE})", flush=True)

    processed = 0
    failures = 0

    for batch_idx, mbid_batch in enumerate(chunked(all_mbids, BATCH_SIZE), start=1):
        print(f"\n--- starting batch {batch_idx} with {len(mbid_batch)} artists ---", flush=True)
        try:
            for artist_works_data in mb.stream_artists_songs_by_mbids(mbid_batch):
                processed += 1

                try:
                    # if not artist_works_data.get("works"):
                    #     mbid = artist_works_data["mbid"]
                    #     print("getting data for empty artist:", mbid)

                    #     # fallback to Spotify data
                    #     spotify_id = mbid_to_spotify_id_map.get(mbid)
                    #     if spotify_id is None:
                    #         raise KeyError(f"no spotify id for mbid {mbid}")

                    #     artist_works_data = sp.get_spotify_artist(spotify_id)
                    #     artist_works_data["mbid"] = mbid

                    write_obj(artist_works_data)

                except Exception as exc:
                    failures += 1
                    artist_mbid = artist_works_data.get("mbid")
                    print(f"failed to process artist {artist_mbid}: {exc}")
                    print(f"this was artist index {processed}")
        except Exception as exc:
            print(f"failed to process chunk {batch_idx}: {exc}")
            print(f"this was artist index {processed}")
        if processed % 1000 == 0:
            print(f"processed {processed} artists so far (failures={failures})", flush=True)

    print(f"\nDone. processed={processed}, failures={failures}", flush=True)

    if cur_file:
        cur_file.close()

    print(f"Finished. Processed {processed} artists. Failures: {failures}")
    print(
        f"Wrote {total_rows} records across {part} JSONL files in {OUTPUT_DIR} "
        f"(max {SHARD_MAX_MB} MB per file)."
    )


if __name__ == "__main__":
    main()
