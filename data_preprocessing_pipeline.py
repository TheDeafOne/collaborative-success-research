from __future__ import annotations

from pathlib import Path

from apis import musicbrainz as mb
from data_preprocessing import (
    fetch_artist_metrics,
    fetch_artists_data,
    filter_artists_with_works,
    filter_spotify_artists,
)


def run_pipeline(
    run_collect_all_artist_mbids: bool = False,
    run_filter_spotify_artists: bool = False,
    run_fetch_artist_metrics: bool = False,
    run_fetch_artist_data: bool = False,
    run_filter_artist_works: bool = True,
) -> dict[str, Path]:
    """
    Straightforward, step-by-step data-prep pipeline mirroring the
    network_construction/network_pipeline style. Toggle steps on/off by
    setting the boolean flags; leave them False to reuse existing artifacts.
    """
    all_mbids_path = filter_spotify_artists.MBIDS_FILE
    metric_map_path = Path(filter_spotify_artists.OUTPUT_FILE)
    artist_metrics_path = Path(fetch_artist_metrics.OUTPUT_FILE)
    artist_full_data_dir = Path(fetch_artists_data.OUTPUT_DIR)
    filtered_artist_dir = Path(filter_artists_with_works.OUTPUT_DIR)
    base_features_path = filter_artists_with_works.DATA_DIR / "artist_base_features_5_years_only_collab.csv"

    step_total = 5
    step = 1

    do_collect_all_mbids = run_collect_all_artist_mbids or (
        run_filter_spotify_artists and not all_mbids_path.exists()
    )
    if do_collect_all_mbids:
        print(f"[{step}/{step_total}] Dumping all artist MBIDs from local MusicBrainz")
        mb.dump_all_artist_mbids(all_mbids_path)
        print(f"      mbids -> {all_mbids_path}")
    else:
        print(f"[{step}/{step_total}] Skipping MBID dump (reuse {all_mbids_path})")
    step += 1

    if run_filter_spotify_artists:
        print(f"[{step}/{step_total}] Filtering artists with Spotify links from MusicBrainz")
        filter_spotify_artists.main()
        print(f"      wrote map -> {metric_map_path}")
    else:
        print(f"[{step}/{step_total}] Skipping filter_spotify_artists (reuse {metric_map_path})")
    step += 1

    if run_fetch_artist_metrics:
        print(f"[{step}/{step_total}] Fetching artist metrics from Spotify")
        fetch_artist_metrics.main()
        print(f"      metrics jsonl -> {artist_metrics_path}")
    else:
        print(f"[{step}/{step_total}] Skipping fetch_artist_metrics (reuse {artist_metrics_path})")
    step += 1

    if run_fetch_artist_data:
        print(f"[{step}/{step_total}] Fetching artist works/recordings shards")
        fetch_artists_data.main()
        print(f"      shards -> {artist_full_data_dir}")
    else:
        print(f"[{step}/{step_total}] Skipping fetch_artists_data (reuse {artist_full_data_dir})")
    step += 1

    if run_filter_artist_works:
        print(f"[{step}/{step_total}] Filtering artists with early works and computing base features")
        filter_artists_with_works.main()
        print(f"      filtered artists -> {filtered_artist_dir}")
        print(f"      base features -> {base_features_path}")
    else:
        print(f"[{step}/{step_total}] Skipping filter_artists_with_works (expected output {filtered_artist_dir})")

    print("Pipeline complete")

    return {
        "all_artist_mbids": all_mbids_path,
        "metric_source_map": metric_map_path,
        "artist_metrics": artist_metrics_path,
        "artist_full_data_dir": artist_full_data_dir,
        "artist_filtered_dir": filtered_artist_dir,
        "base_features": base_features_path,
    }


if __name__ == "__main__":
    run_pipeline(False, True, False, False, False)
