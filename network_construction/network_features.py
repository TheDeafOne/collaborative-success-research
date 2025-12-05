import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import sparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def _prepare_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[nx.Graph, List[str]]:
    """Build an undirected graph from edges and align node attributes."""
    if "u" not in edges or "v" not in edges:
        raise ValueError("edges must contain columns 'u' and 'v'")
    if "artist_mbid" not in nodes:
        raise ValueError("nodes must contain column 'artist_mbid'")

    nodes_df = nodes.copy()
    edges_df = edges.copy()

    nodes_df["mbid"] = nodes_df["artist_mbid"].astype(str).str.strip()
    edges_df["u"] = edges_df["u"].astype(str).str.strip()
    edges_df["v"] = edges_df["v"].astype(str).str.strip()
    edges_df = edges_df.loc[
        edges_df["u"].notna() & edges_df["v"].notna() & (edges_df["u"] != "") & (edges_df["v"] != "")
    ]

    g = nx.Graph()
    g.add_edges_from(edges_df[["u", "v"]].itertuples(index=False, name=None))
    node_order = list(g.nodes())

    nodes_lookup = nodes_df.set_index("mbid")
    aligned = nodes_lookup.reindex(node_order)
    for col in aligned.columns:
        if col in {"mbid", "name"}:
            continue
        values = aligned[col]
        values = values.where(pd.notna(values), None)
        nx.set_node_attributes(g, values.to_dict(), col)

    nx.set_node_attributes(g, {n: n for n in node_order}, "mbid")
    return g, node_order


def _edge_shared_partners_matrix(g: nx.Graph, node_order: Sequence[str]) -> sparse.spmatrix:
    adj = nx.to_scipy_sparse_array(g, nodelist=node_order, dtype=np.int64, format="csr")
    return adj @ adj


def _enumerate_triangles(g: nx.Graph, node_to_idx: Dict[str, int]) -> Iterable[Tuple[str, str, str]]:
    """Yield each triangle once using index ordering to avoid duplicates."""
    for u in g:
        u_idx = node_to_idx[u]
        for v in g[u]:
            v_idx = node_to_idx[v]
            if v_idx <= u_idx:
                continue
            common = set(g[u]) & set(g[v])
            for w in common:
                w_idx = node_to_idx[w]
                if w_idx <= v_idx:
                    continue
                yield (u, v, w)


def _prop_same_attr(g: nx.Graph, node_order: Sequence[str], attr: str, neighbors: Dict[str, set]) -> List[float]:
    vals = {n: g.nodes[n].get(attr) for n in node_order}
    result = []
    for n in node_order:
        neigh = neighbors[n]
        if not neigh:
            result.append(0.0)
            continue
        match = sum(1 for nb in neigh if vals.get(nb) == vals[n])
        result.append(match / len(neigh))
    return result


def _chunked(seq: Sequence, size: int) -> List[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _count_triangles_chunk(tri_chunk, attr_vals, mode: str) -> Dict[int, int]:
    """
    mode: 'same' (all attributes equal) or 'all_diff' (all attributes distinct)
    attr_vals: list aligned to node indices.
    """
    counts = defaultdict(int)
    for u_idx, v_idx, w_idx in tri_chunk:
        vals = (attr_vals[u_idx], attr_vals[v_idx], attr_vals[w_idx])
        uniq = set(vals)
        if mode == "same":
            ok = len(uniq) == 1
        else:
            ok = len(uniq) == 3
        if not ok:
            continue
        counts[u_idx] += 1
        counts[v_idx] += 1
        counts[w_idx] += 1
    return counts


def _approximate_closeness(
    g: nx.Graph,
    node_order: Sequence[str],
    node_to_idx: Dict[str, int],
    k: int,
    seed: int | None,
    show_progress: bool,
) -> np.ndarray:
    """Approximate closeness by sampling k source nodes."""
    n = len(node_order)
    if k is None or k >= n:
        samples = list(node_order)
    else:
        rng = default_rng(seed)
        samples = rng.choice(node_order, size=k, replace=False).tolist()

    total_dist = np.zeros(n, dtype=float)
    reached = np.zeros(n, dtype=int)

    iterator = samples
    if show_progress:
        iterator = tqdm(samples, desc="[features] closeness BFS", leave=False)

    for s in iterator:
        lengths = nx.single_source_shortest_path_length(g, s)
        for node, dist in lengths.items():
            idx = node_to_idx[node]
            total_dist[idx] += dist
            reached[idx] += 1

    closeness = np.zeros(n, dtype=float)
    for i in range(n):
        if total_dist[i] == 0.0:
            closeness[i] = 0.0
        else:
            # Approximate: average distance to sampled nodes, scaled by how many samples reached this node.
            closeness[i] = reached[i] / total_dist[i]
    return closeness


def compute_network_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_k_hubs: int = 5,
    gwesp_decay: float = 0.5,
    *,
    max_workers: int | None = None,
    progress: bool = True,
    betweenness_k: int | None = 256,
    betweenness_seed: int | None = 0,
    closeness_k: int | None = 256,
    eigen_max_iter: int = 200,
    eigen_tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Reproduce the network features engineered in ergm_new_features.ipynb.

    Parameters
    ----------
    nodes : pd.DataFrame
        Must contain at least 'artist_mbid' (used as node id) and, optionally,
        attributes like primary_genre/primary_label/primary_role.
    edges : pd.DataFrame
        Must contain 'u' and 'v' columns (artist_mbid endpoints).
    top_k_hubs : int
        Number of highest-eigenvector-centrality nodes used for distance to hub.
    gwesp_decay : float
        Decay parameter for the gwesp-style node score.
    """
    g, node_order = _prepare_graph(nodes, edges)
    if g.number_of_nodes() == 0:
        raise ValueError("graph is empty after filtering edges with missing endpoints")

    node_to_idx = {n: i for i, n in enumerate(node_order)}
    n = len(node_order)

    neighbors = {n: set(g.neighbors(n)) for n in node_order}
    deg_arr = np.array([len(neighbors[n]) for n in node_order], dtype=float)
    log_deg = np.log1p(deg_arr)

    # Weak ties and gwesp-style score share the shared-partner matrix.
    if progress:
        print("[features] computing shared-partner matrix")
    sp_mat = _edge_shared_partners_matrix(g, node_order)
    gwesp_score = np.zeros(n, dtype=float)
    weak_tie_count = np.zeros(n, dtype=float)
    edge_iter = g.edges()
    if progress:
        edge_iter = tqdm(edge_iter, total=g.number_of_edges(), desc="[features] scoring edges")
    for u, v in edge_iter:
        i, j = node_to_idx[u], node_to_idx[v]
        shared = sp_mat[i, j]
        weight = 1 - math.exp(-gwesp_decay * shared)
        gwesp_score[i] += weight
        gwesp_score[j] += weight
        if shared <= 1:
            weak_tie_count[i] += 1.0
            weak_tie_count[j] += 1.0
    weak_tie_frac = np.where(deg_arr > 0, weak_tie_count / deg_arr, 0.0)

    if progress:
        print("[features] clustering")
    clustering = nx.clustering(g, nodes=node_order)
    clust_local = np.array([clustering[n] for n in node_order], dtype=float)
    if progress:
        print("[features] triangles")
    triangles = nx.triangles(g)
    triangles_per_node = np.array([triangles[n] for n in node_order], dtype=float)

    if progress:
        print("[features] wedges")
    open_wedges = np.where(deg_arr >= 2, deg_arr * (deg_arr - 1) / 2 - triangles_per_node, 0.0)

    if progress:
        print(
            "[features] centralities (eigen/core/betweenness/closeness)"
        )
    eigencent = nx.eigenvector_centrality(g, max_iter=eigen_max_iter, tol=eigen_tol)
    eig_cen = np.array([eigencent[n] for n in node_order], dtype=float)

    core_numbers = nx.core_number(g)
    kcore = np.array([core_numbers[n] for n in node_order], dtype=float)

    betweenness_sample = None if betweenness_k is None or betweenness_k >= n else min(betweenness_k, n)
    if progress:
        print(f"[features] betweenness k={betweenness_sample or 'all'}")
    betw = nx.betweenness_centrality(
        g,
        normalized=True,
        k=betweenness_sample,
        seed=betweenness_seed,
    )
    betweenness = np.array([betw[n] for n in node_order], dtype=float)

    closeness_sample = None if closeness_k is None or closeness_k >= n else min(closeness_k, n)
    if progress:
        print(f"[features] closeness k={closeness_sample or 'all'}")
    if closeness_sample:
        closeness = _approximate_closeness(
            g,
            node_order,
            node_to_idx,
            k=closeness_sample,
            seed=betweenness_seed,
            show_progress=progress,
        )
    else:
        close = nx.closeness_centrality(g, wf_improved=True)
        closeness = np.array([close[n] for n in node_order], dtype=float)

    # Distance to hubs (top eigenvector centrality nodes).
    if progress:
        print("[features] distance to hubs")
    hub_idx = np.argsort(-eig_cen)[: min(top_k_hubs, n)]
    hub_nodes = [node_order[i] for i in hub_idx]
    dist_map = nx.multi_source_dijkstra_path_length(g, hub_nodes)
    dist_to_hubs = np.array([dist_map.get(n, math.inf) for n in node_order], dtype=float)

    # Community participation via Louvain partition.
    if progress:
        print("[features] louvain communities")
    communities = nx.algorithms.community.louvain_communities(g, seed=0)
    comm_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = cid
    
    if progress:
        print("[features] participation")
    participation_coef = []
    for node_id in node_order:
        neigh = neighbors[node_id]
        if not neigh:
            participation_coef.append(0.0)
            continue
        different = sum(1 for nb in neigh if comm_map.get(nb) != comm_map.get(node_id))
        participation_coef.append(different / len(neigh))

    # Homophily.
    if progress:
        print("[features] homophily shares")
    same_genre_share = _prop_same_attr(g, node_order, "primary_genre", neighbors)
    same_label_share = _prop_same_attr(g, node_order, "primary_label", neighbors)
    same_role_share = _prop_same_attr(g, node_order, "primary_role", neighbors)

    # Triangle compositions.
    if progress:
        print("[features] enumerating triangles")
    triangle_list = list(_enumerate_triangles(g, node_to_idx))

    def _triangle_counts_parallel(mode: str, attr: str):
        attr_vals = [g.nodes[n].get(attr) for n in node_order]
        # chunk by ~50k triangles to balance overhead
        chunk_size = int(5e4)
        chunks = list(_chunked([(node_to_idx[u], node_to_idx[v], node_to_idx[w]) for u, v, w in triangle_list], chunk_size))
        counts_total = defaultdict(int)
        if not chunks:
            return [0] * n
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_count_triangles_chunk, chunk, attr_vals, mode)
                for chunk in chunks
            ]
            for fut in tqdm(futures, desc=f"[features] triangles ({mode} {attr})", disable=not progress):
                for idx, val in fut.result().items():
                    counts_total[idx] += val
        return [counts_total[i] for i in range(n)]

    tri_within_genre = _triangle_counts_parallel("same", "primary_genre")
    tri_within_label = _triangle_counts_parallel("same", "primary_label")
    tri_within_role = _triangle_counts_parallel("same", "primary_role")

    tri_cross_genre = _triangle_counts_parallel("all_diff", "primary_genre")
    tri_cross_label = _triangle_counts_parallel("all_diff", "primary_label")
    tri_cross_role = _triangle_counts_parallel("all_diff", "primary_role")

    # Component size.
    comp_size_map = {}
    for comp in nx.connected_components(g):
        size = len(comp)
        for node in comp:
            comp_size_map[node] = size
    component_size = np.array([comp_size_map[n] for n in node_order], dtype=float)

    # Open dyad opportunities.
    open_dyads = []
    if progress:
        print("[features] component sizes & open dyads")
    for node_id in node_order:
        onehop = neighbors[node_id]
        twohop = set(onehop)
        for nb in onehop:
            twohop.update(neighbors.get(nb, set()))
        twohop.add(node_id)
        twohop_unique = len(twohop) - 1
        open_dyads.append(max(twohop_unique - len(onehop), 0))

    df = pd.DataFrame(
        {
          "node": node_order,
          "clust_local": clust_local,
          "triangles": triangles_per_node,
          "open_wedges": open_wedges,
          "gwesp_score": gwesp_score,
          "degree": deg_arr,
          "log_degree": log_deg,
          "eigencent": eig_cen,
          "kcore": kcore,
          "betweenness": betweenness,
          "closeness": closeness,
          "dist_to_hub_min": dist_to_hubs,
          "same_genre_share": same_genre_share,
          "same_label_share": same_label_share,
          "same_role_share": same_role_share,
          "tri_within_genre": tri_within_genre,
          "tri_within_label": tri_within_label,
          "tri_within_role": tri_within_role,
          "tri_cross_genre": tri_cross_genre,
          "tri_cross_label": tri_cross_label,
          "tri_cross_role": tri_cross_role,
          "component_size": component_size,
          "open_dyads": open_dyads,
          "weak_tie_frac": weak_tie_frac,
          "community_participation": participation_coef,
        }
    )
    return df


__all__ = ["compute_network_features"]
