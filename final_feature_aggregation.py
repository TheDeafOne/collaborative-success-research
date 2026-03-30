#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_BASE_FEATURES = Path("data/artist_base_features_5_years_only_collab.csv")
DEFAULT_NODE_FEATURES = Path("data/graphs/better_graph/node_features.csv")
DEFAULT_NODE_EMBEDDINGS = Path("data/graphs/only_connected/node_embeddings.csv")
DEFAULT_METRICS_JSONL = Path("data/artist_to_metrics_map.jsonl")
DEFAULT_OUTPUT = Path("data/final_df_with_ids.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate base + graph + embedding + metrics features into a single final "
            "dataset while preserving artist identifiers."
        )
    )
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--node-features", type=Path, default=DEFAULT_NODE_FEATURES)
    parser.add_argument("--node-embeddings", type=Path, default=DEFAULT_NODE_EMBEDDINGS)
    parser.add_argument("--metrics-jsonl", type=Path, default=DEFAULT_METRICS_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_metrics_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["artist_mbid", "artist_id", "followers", "popularity"])
    metrics = pd.DataFrame(rows)
    if "artist_mbid" not in metrics.columns:
        raise ValueError("metrics JSONL must include 'artist_mbid'.")
    metrics["artist_mbid"] = metrics["artist_mbid"].astype(str).str.strip()
    # Script may be rerun in append mode upstream; keep latest record per mbid.
    metrics = metrics.drop_duplicates(subset=["artist_mbid"], keep="last")
    return metrics


def _normalize_node_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "eigen_centrality": "eigencent",
        "participation_coef": "community_participation",
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if cols_to_rename:
        df = df.rename(columns=cols_to_rename)
    return df


def build_final_dataframe(
    base_features_path: Path,
    node_features_path: Path,
    node_embeddings_path: Path,
    metrics_jsonl_path: Path,
) -> pd.DataFrame:
    base = pd.read_csv(base_features_path)
    node_features = pd.read_csv(node_features_path)
    node_embeddings = pd.read_csv(node_embeddings_path)
    metrics = _read_metrics_jsonl(metrics_jsonl_path)

    required_base_cols = {"artist_mbid", "artist_name"}
    missing_base = required_base_cols - set(base.columns)
    if missing_base:
        raise ValueError(f"base features missing required columns: {sorted(missing_base)}")

    if "node" not in node_features.columns:
        raise ValueError("node features must include 'node' column (artist MBID).")
    if "artist_mbid" not in node_embeddings.columns:
        raise ValueError("node embeddings must include 'artist_mbid' column.")

    base["artist_mbid"] = base["artist_mbid"].astype(str).str.strip()
    node_features["node"] = node_features["node"].astype(str).str.strip()
    node_embeddings["artist_mbid"] = node_embeddings["artist_mbid"].astype(str).str.strip()

    node_features = _normalize_node_feature_columns(node_features)
    node_embeddings = node_embeddings.drop_duplicates(subset=["artist_mbid"], keep="first")

    network = node_features.merge(
        node_embeddings,
        how="left",
        left_on="node",
        right_on="artist_mbid",
    )
    network = network.merge(
        metrics,
        how="left",
        left_on="node",
        right_on="artist_mbid",
    )

    # Keep graph rows that map to the modeling artist universe from base features.
    network = network[network["node"].isin(set(base["artist_mbid"]))].copy()
    network["artist_mbid"] = network["node"]

    drop_cols = [c for c in ["node", "artist_mbid_x", "artist_mbid_y"] if c in network.columns]
    if drop_cols:
        network = network.drop(columns=drop_cols)

    network = network.drop_duplicates(subset=["artist_mbid"], keep="last")

    final_df = base.merge(network, how="left", on="artist_mbid")

    # Align with prior notebook cleanup, but keep the requested identifiers.
    notebook_drop_cols = [
        "window_years",
        "debut_date",
        "window_cutoff_date",
        "release_velocity_releases_per_day",
        "unique_collaborator_count",
        "all_labels_str",
        "all_genres_str",
        "all_roles_str",
        "debut_decade",
        "recency_index",
        "popularity",
        "debut_year",
    ]
    final_df = final_df.drop(columns=[c for c in notebook_drop_cols if c in final_df.columns])

    # Ensure key identifiers are present in output.
    ordered_front = ["artist_mbid", "artist_name", "artist_id"]
    present_front = [c for c in ordered_front if c in final_df.columns]
    rest = [c for c in final_df.columns if c not in present_front]
    final_df = final_df.loc[:, present_front + rest]

    return final_df


def main() -> None:
    args = parse_args()
    final_df = build_final_dataframe(
        base_features_path=args.base_features,
        node_features_path=args.node_features,
        node_embeddings_path=args.node_embeddings,
        metrics_jsonl_path=args.metrics_jsonl,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"Wrote {len(final_df):,} rows and {len(final_df.columns)} columns to {args.output}")
    keep_cols = [c for c in ["artist_mbid", "artist_name", "artist_id"] if c in final_df.columns]
    print(f"Kept identifier columns: {keep_cols}")


if __name__ == "__main__":
    main()
