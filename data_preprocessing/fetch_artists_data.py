#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from apis import musicbrainz as mb

DATA_DIR = ROOT / "data"
ARTIST_TO_METRIC_MAP = DATA_DIR / "artist_to_metrics_map_old.json"
OUTPUT_DIR = DATA_DIR / "filtered_artist_data"  # JSONL shards will go here
SHARD_MAX_MB = 300  # cap per jsonl file


def main() -> None:
    # Load input mapping
    try:
        with open(ARTIST_TO_METRIC_MAP) as f:
            artist_to_metrics_map = json.load(f)
    except Exception as exc:
        sys.exit(f"Failed to read {ARTIST_TO_METRIC_MAP}: {exc}")

    failures = 0
    processed = 0

    # This is your filtered list of artist MBIDs
    filtered_artist_mbids = list(artist_to_metrics_map.keys())

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

    # Stream from MusicBrainz client and write to sharded JSONL
    for artist_works_data in mb.stream_artists_songs_by_mbids(filtered_artist_mbids):
        processed += 1
        try:
            write_obj(artist_works_data)
        except Exception as exc:
            failures += 1
            artist_mbid = artist_works_data.get("mbid")
            print(f"failed to process artist {artist_mbid}: {exc}")
            print(f"this was artist index {processed}")

        if processed % 1000 == 0:
            print(f"processed {processed} artists so far (failures={failures})")

    if cur_file:
        cur_file.close()

    print(f"Finished. Processed {processed} artists. Failures: {failures}")
    print(
        f"Wrote {total_rows} records across {part} JSONL files in {OUTPUT_DIR} "
        f"(max {SHARD_MAX_MB} MB per file)."
    )


if __name__ == "__main__":
    main()
