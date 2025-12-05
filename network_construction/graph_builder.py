from __future__ import annotations

import json
from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _parse_date_to_ordinal(value: Optional[str]) -> Optional[int]:
    """Parse a loose date string (YYYY[-MM[-DD]]) to an ordinal integer."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None

    from datetime import datetime

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).date().toordinal()
        except ValueError:
            continue
    return None


def _compute_work_release_ordinals(artist: dict) -> Dict[int, Optional[int]]:
    """Return earliest known release date per work as ordinals."""
    release_dates: Dict[int, Optional[int]] = {}
    for release in artist.get("releases") or []:
        rid = release.get("id")
        if rid is None:
            continue
        release_dates[rid] = _parse_date_to_ordinal(release.get("date"))

    work_dates: Dict[int, Optional[int]] = {}
    for rec in artist.get("recordings") or []:
        wid = rec.get("work_id")
        rid = rec.get("release_id")
        if wid is None or rid is None:
            continue
        rel_ord = release_dates.get(rid)
        if rel_ord is None:
            continue
        cur = work_dates.get(wid)
        if cur is None or rel_ord < cur:
            work_dates[wid] = rel_ord

    for work in artist.get("works") or []:
        wid = work.get("id")
        if wid is None:
            continue
        first_ord = _parse_date_to_ordinal(work.get("first_release_date"))
        cur = work_dates.get(wid)
        if first_ord is not None and (cur is None or first_ord < cur):
            work_dates[wid] = first_ord
        elif wid not in work_dates:
            work_dates[wid] = None
    return work_dates


def _load_artist_info(features_df: pd.DataFrame) -> Dict[str, Dict[str, Optional[int]]]:
    """Map artist MBID -> metadata needed for window checking."""
    info: Dict[str, Dict[str, Optional[int]]] = {}
    for _, row in features_df.iterrows():
        mbid = row.get("artist_mbid")
        if not isinstance(mbid, str):
            continue
        info[mbid] = {
            "debut": _parse_date_to_ordinal(row.get("debut_date")),
            "cutoff": _parse_date_to_ordinal(row.get("window_cutoff_date")),
        }
    return info


def _process_artist_file(
    file_path: Path,
    artist_info: Dict[str, Dict[str, Optional[int]]],
) -> Dict[Tuple[str, str], int]:
    """Process one JSONL file and return pair counts found within."""
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                artist = json.loads(line)
            except json.JSONDecodeError:
                continue

            artist_mbid = artist.get("mbid")
            if artist_mbid not in artist_info:
                continue

            work_dates = _compute_work_release_ordinals(artist)
            works = artist.get("works") or []

            for work in works:
                wid = work.get("id")
                if wid is None:
                    continue
                work_ord = work_dates.get(wid)
                if work_ord is None:
                    continue

                participant_mbids = {artist_mbid}
                for collab in work.get("collaborators") or []:
                    mbid = collab.get("mbid")
                    if mbid:
                        participant_mbids.add(mbid)

                # participants_in_set = []
                # for mbid in participant_mbids:
                #     info = artist_info.get(mbid)
                #     if info is None:
                #         continue
                #     debut = info["debut"]
                #     cutoff = info["cutoff"]
                #     if debut is not None and work_ord < debut:
                #         continue
                #     if cutoff is not None and work_ord > cutoff:
                #         continue
                #     participants_in_set.append(mbid)

                participants_in_set = list(participant_mbids)

                if len(participants_in_set) < 2:
                    continue

                participants_in_set.sort()
                for i in range(len(participants_in_set)):
                    for j in range(i + 1, len(participants_in_set)):
                        u, v = participants_in_set[i], participants_in_set[j]
                        pair_counts[(u, v)] += 1

    return pair_counts


def build_cowrite_graph_simple(
    features_df: pd.DataFrame,
    artist_dir: Path | str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a minimalist co-write graph.

    - Nodes: exactly the artists listed in `features_df`.
    - Edge between artists u and v exists if they co-authored at least one work
      whose release date falls within the first five years of BOTH artists'
      observed careers (based on debut/cutoff in `features_df`).
    """
    artist_dir = Path(artist_dir)
    nodes_df = features_df.copy().reset_index(drop=True)
    if "artist_mbid" not in nodes_df.columns:
        raise ValueError("features_df must contain 'artist_mbid'.")

    artist_info = _load_artist_info(nodes_df)
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    files = sorted(artist_dir.glob("*.jsonl"))
    max_workers = max(1, min(8, os.cpu_count() or 1))

    def _run_pool(executor_cls):
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_artist_file, path, artist_info): path
                for path in files
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Artist files",
                unit="file",
            ):
                local_counts = future.result()
                for key, value in local_counts.items():
                    pair_counts[key] += value

    try:
        _run_pool(ProcessPoolExecutor)
    except PermissionError:
        _run_pool(ThreadPoolExecutor)

    edges_rows = [
        {"u": u, "v": v, "weight": weight}
        for (u, v), weight in pair_counts.items()
    ]
    edges_df = pd.DataFrame(edges_rows)
    return edges_df, nodes_df


def main():
    features = pd.read_csv("./data/artist_base_features_5_years_only_collab.csv")
    edges_df, nodes_df = build_cowrite_graph_simple(
        features_df=features,
        artist_dir=Path("./data/artist_filtered_data"),
    )
    out_dir = Path("./data/graphs/only_connected")
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_dir / "edges.csv", index=False)
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)


if __name__ == "__main__":
    main()
