#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dotenv import load_dotenv

from apis import spotify as sp

load_dotenv()

DATA_DIR = ROOT / "data"
ARTIST_TO_METRIC_MAP = DATA_DIR / "artist_to_metric_sources_map.json"
OUTPUT_FILE = DATA_DIR / "artist_to_metrics_map.jsonl"  # JSONL now


def main() -> None:
    # Load input mapping
    try:
        with open(ARTIST_TO_METRIC_MAP) as f:
            artist_to_metrics_map = json.load(f)
    except Exception as exc:
        sys.exit(f"Failed to read {ARTIST_TO_METRIC_MAP}: {exc}")

    failures = 0
    processed = 0

    spotify_id_length = 22
    filtered_artist_mbids = [
        (artist_mbid, source_data["spotify"].split("/")[-1])
        for artist_mbid, source_data in list(artist_to_metrics_map.items())
        if "spotify" in source_data
    ]
    filtered_artist_mbids = list(
        filter(lambda x: len(x[1]) == spotify_id_length, filtered_artist_mbids)
    )

    chunk_size = 50
    chunked_list = [
        filtered_artist_mbids[i : i + chunk_size]
        for i in range(0, len(filtered_artist_mbids), chunk_size)
    ]

    spotify_id_to_mbid_map = {
        id_pair[1]: id_pair[0] for id_pair in filtered_artist_mbids
    }

    # Open output JSONL in append mode so script can resume safely
    with open(OUTPUT_FILE, "a") as out:
        for artist_id_chunk in chunked_list:
            mbids = [id_pair[0] for id_pair in artist_id_chunk]
            spotify_ids = [id_pair[1] for id_pair in artist_id_chunk]

            try:
                multiple_artist_data = sp.get_multiple_artist_success_metrics(
                    spotify_ids
                )
                for mbid, artist_data in zip(mbids, multiple_artist_data):
                    artist_data["artist_mbid"] = mbid
                    # Write one JSON object per line
                    out.write(json.dumps(artist_data) + "\n")
                out.flush()  # ensure data is written immediately
                processed += 50
            except Exception as exc:
                failures += 1
                print(f"failed to process: {exc}")
                print(f"this was at {processed}")

            if processed % 1000 == 0:
                print(f"processed {processed} artists so far")

    print(f"Finished. Failures: {failures}")
    print(f"Results written incrementally to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
