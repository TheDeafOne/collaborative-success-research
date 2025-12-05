from __future__ import annotations

from pathlib import Path

import pandas as pd

from network_construction.graph_builder import build_cowrite_graph_simple
from network_construction.network_features import compute_network_features
from network_construction.node_embeddings import compute_node2vec_embeddings


def run_pipeline(
    base_features_path: Path | str = "./data/artist_base_features_5_years_only_collab.csv",
    artist_dir: Path | str = "./data/artist_filtered_data",
    output_dir: Path | str = "./data/graphs/only_connected",
):
    """
    Straightforward, step-by-step pipeline.
    Comment out steps if you want to skip them.
    """
    # base_features_path = Path(base_features_path)
    artist_dir = Path(artist_dir)
    output_dir = Path(output_dir)

    # print("[1/6] Loading base features")
    # features_df = pd.read_csv(base_features_path)
    # print(f"      Loaded {len(features_df):,} artists from {base_features_path}")

    # print("[2/6] Building co-write graph")
    # edges_df, nodes_df = build_cowrite_graph_simple(features_df=features_df, artist_dir=artist_dir)
    # print(f"      Graph has {len(nodes_df):,} nodes and {len(edges_df):,} edges")

    # print("[3/6] Saving graph artifacts")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # edges_path = output_dir / "edges.csv"
    # nodes_path = output_dir / "nodes.csv"
    # edges_df.to_csv(edges_path, index=False)
    # nodes_df.to_csv(nodes_path, index=False)
    # print(f"      edges -> {edges_path}")
    # print(f"      nodes -> {nodes_path}")

    nodes_df = pd.read_csv(output_dir / 'nodes.csv')
    edges_df = pd.read_csv(output_dir / 'edges.csv')

    # print("[4/6] Computing node-level network features")
    # node_features = compute_network_features(nodes_df, edges_df)
    # features_path = output_dir / "node_features.csv"
    # node_features.to_csv(features_path, index=False)
    # print(f"      node_features -> {features_path}")

    print("[5/6] Computing node2vec embeddings")
    node_embeddings = compute_node2vec_embeddings(nodes_df, edges_df)
    embeds_path = output_dir / "node_embeddings.csv"
    print(node_embeddings.head())
    node_embeddings.to_csv(embeds_path, index=False)
    print(f"      node_embeddings -> {embeds_path}")

    print("[6/6] Done")

    return {
        "edges": edges_df,
        "nodes": nodes_df,
        # "node_features": node_features,
        "node_embeddings": node_embeddings,
        # "paths": {
        #     "edges": edges_path,
        #     "nodes": nodes_path,
        #     "node_features": features_path,
        #     "node_embeddings": embeds_path,
        # },
    }


if __name__ == "__main__":
    run_pipeline()
