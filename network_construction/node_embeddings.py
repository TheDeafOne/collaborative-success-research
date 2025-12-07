from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from tqdm import tqdm
import multiprocessing

n_cpus = multiprocessing.cpu_count()

def compute_node2vec_embeddings(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    dimensions: int = 32,
    walk_length: int = 10,
    num_walks: int = 8,
    p: float = 1.0,
    q: float = 1.0,
    workers: int = 1,
    weight_column: str = "weight",
    progress: bool = True,
) -> pd.DataFrame:
    """Compute node2vec embeddings and return a DataFrame with mbid + embedding columns."""
    if "mbid" in nodes.columns:
        mbid_col = "mbid"
    elif "artist_mbid" in nodes.columns:
        mbid_col = "artist_mbid"
    else:
        raise ValueError("nodes must contain either 'mbid' or 'artist_mbid'")

    if "u" not in edges or "v" not in edges:
        raise ValueError("edges must contain 'u' and 'v'")

    mbids = nodes[mbid_col].astype(str)
    nodes_df = nodes.copy()
    nodes_df["mbid"] = mbids

    edges_df = edges.copy()
    edges_df["u"] = edges_df["u"].astype(str)
    edges_df["v"] = edges_df["v"].astype(str)

    G = nx.Graph()

    if progress:
        print("[node2vec] adding nodes")
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), disable=not progress, desc="[node2vec] nodes"):
        attrs = row.to_dict()
        node_id = attrs.pop("mbid")
        G.add_node(node_id, **attrs)

    if progress:
        print("[node2vec] adding edges")
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), disable=not progress, desc="[node2vec] edges"):
        u = str(row["u"])
        v = str(row["v"])
        weight = row.get(weight_column, 1.0)
        if pd.isna(weight):
            weight = 1.0
        G.add_edge(u, v, weight=weight)

    if progress:
        print(f"[node2vec] graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print("[node2vec] running node2vec walks")

    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        weight_key="weight",
        quiet=not progress,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    if progress:
        print(f"[node2vec] finished node2vec fit")
        print("[node2vec] running mapping")

    embeddings = []
    for node in G.nodes():
        key = str(node)
        if key in model.wv:
            vec = model.wv[key]
            embeddings.append((node, vec))

    

    embed_df = pd.DataFrame(
        [[node] + vec.tolist() for node, vec in embeddings],
        columns=["artist_mbid"] + [f"n2v_{i}" for i in range(model.vector_size)],
    )
    return embed_df

__all__ = ["compute_node2vec_embeddings"]
