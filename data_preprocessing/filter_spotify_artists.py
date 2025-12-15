#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from apis import musicbrainz as mb
from tqdm import tqdm
DATA_DIR = ROOT / "data"
MBIDS_FILE = DATA_DIR / "all_artist_mbids.txt"
OUTPUT_FILE = DATA_DIR / "artist_to_metric_sources_map.json"

def main() -> None:
    try:
        mbids = [line.strip() for line in MBIDS_FILE.read_text().splitlines() if line.strip()]
    except Exception as exc:
        sys.exit(f"Failed to read {MBIDS_FILE}: {exc}")
    mapping = {}
    failures = 0
    with tqdm(total=len(mbids), desc="Artists", unit="artist") as pbar:
        for artist_mbid, resources in mb.stream_artist_resources(mbids, include_empty=True):
            try:
                valid = {}
                for resource in resources:
                    url = resource.get("url", "")
                    lowered = url.lower()
                    if "spotify" in lowered:
                        valid["spotify"] = url
                if valid:
                    mapping[str(artist_mbid)] = valid
            except Exception as exc:
                failures += 1
                print(f"failed to process {artist_mbid}: {exc}", file=sys.stderr)
            finally:
                pbar.update(1)
    OUTPUT_FILE.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"wrote {len(mapping)} artists to {OUTPUT_FILE} (failures: {failures})")

if __name__ == "__main__":
    main()
