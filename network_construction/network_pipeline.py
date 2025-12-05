from __future__ import annotations

from pathlib import Path

import pandas as pd

from network_construction.graph_builder import build_cowrite_graph_simple
from network_construction.network_features import compute_network_features


def run_pipeline(
    base_features_path: Path | str = "./data/artist_base_features_5_years_only_collab.csv",
    artist_dir: Path | str = "./data/artist_filtered_data",
    output_dir: Path | str = "./data/graphs/only_connected",
):
    base_features_path = Path(base_features_path)
    artist_dir = Path(artist_dir)
    output_dir = Path(output_dir)

    print(f"[1/5] Loading base features from {base_features_path}")
    features_df = pd.read_csv(base_features_path)
    print(f"      Loaded {len(features_df):,} artists")

    print(f"[2/5] Building co-write graph from artist files in {artist_dir}")
    edges_df, nodes_df = build_cowrite_graph_simple(features_df=features_df, artist_dir=artist_dir)
    print(f"      Graph has {len(nodes_df):,} nodes and {len(edges_df):,} edges")

    print(f"[3/5] Ensuring output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_path = output_dir / "edges.csv"
    nodes_path = output_dir / "nodes.csv"
    print(f"[4/5] Writing graph artifacts:\n        edges -> {edges_path}\n        nodes -> {nodes_path}")
    edges_df.to_csv(edges_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)

    print("[5/5] Computing node-level network features")
    node_features = compute_network_features(nodes_df, edges_df)
    features_path = output_dir / "node_features.csv"
    node_features.to_csv(features_path, index=False)
    print(f"      Wrote {len(node_features):,} rows to {features_path}")

    return {
        "edges": edges_df,
        "nodes": nodes_df,
        "node_features": node_features,
        "paths": {"edges": edges_path, "nodes": nodes_path, "node_features": features_path},
    }


if __name__ == "__main__":
    run_pipeline()
