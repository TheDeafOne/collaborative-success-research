import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


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


def _triangles_with_same_attr(
    g: nx.Graph, node_order: Sequence[str], triangles: Iterable[Tuple[str, str, str]], attr: str
) -> List[int]:
    vals = {n: g.nodes[n].get(attr) for n in node_order}
    counts = defaultdict(int)
    for u, v, w in triangles:
        if len({vals[u], vals[v], vals[w]}) == 1:
            counts[u] += 1
            counts[v] += 1
            counts[w] += 1
    return [counts[n] for n in node_order]


def _triangles_with_all_diff_attr(
    g: nx.Graph, node_order: Sequence[str], triangles: Iterable[Tuple[str, str, str]], attr: str
) -> List[int]:
    vals = {n: g.nodes[n].get(attr) for n in node_order}
    counts = defaultdict(int)
    for u, v, w in triangles:
        if len({vals[u], vals[v], vals[w]}) == 3:
            counts[u] += 1
            counts[v] += 1
            counts[w] += 1
    return [counts[n] for n in node_order]


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


def compute_network_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, top_k_hubs: int = 5, gwesp_decay: float = 0.5
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
    sp_mat = _edge_shared_partners_matrix(g, node_order)
    gwesp_score = np.zeros(n, dtype=float)
    weak_tie_count = np.zeros(n, dtype=float)
    for u, v in g.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        shared = sp_mat[i, j]
        weight = 1 - math.exp(-gwesp_decay * shared)
        gwesp_score[i] += weight
        gwesp_score[j] += weight
        if shared <= 1:
            weak_tie_count[i] += 1.0
            weak_tie_count[j] += 1.0
    weak_tie_frac = np.where(deg_arr > 0, weak_tie_count / deg_arr, 0.0)

    clustering = nx.clustering(g, nodes=node_order)
    clust_local = np.array([clustering[n] for n in node_order], dtype=float)

    triangles = nx.triangles(g)
    triangles_per_node = np.array([triangles[n] for n in node_order], dtype=float)

    open_wedges = np.where(deg_arr >= 2, deg_arr * (deg_arr - 1) / 2 - triangles_per_node, 0.0)

    eigencent = nx.eigenvector_centrality(g, max_iter=1000)
    eig_cen = np.array([eigencent[n] for n in node_order], dtype=float)

    core_numbers = nx.core_number(g)
    kcore = np.array([core_numbers[n] for n in node_order], dtype=float)

    betw = nx.betweenness_centrality(g, normalized=True)
    betweenness = np.array([betw[n] for n in node_order], dtype=float)

    close = nx.closeness_centrality(g, wf_improved=True)
    closeness = np.array([close[n] for n in node_order], dtype=float)

    # Distance to hubs (top eigenvector centrality nodes).
    hub_idx = np.argsort(-eig_cen)[: min(top_k_hubs, n)]
    hub_nodes = [node_order[i] for i in hub_idx]
    dist_map = nx.multi_source_dijkstra_path_length(g, hub_nodes)
    dist_to_hubs = np.array([dist_map.get(n, math.inf) for n in node_order], dtype=float)

    # Community participation via Louvain partition.
    communities = nx.algorithms.community.louvain_communities(g, seed=0)
    comm_map = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = cid
    participation_coef = []
    for n in node_order:
        neigh = neighbors[n]
        if not neigh:
            participation_coef.append(0.0)
            continue
        different = sum(1 for nb in neigh if comm_map.get(nb) != comm_map.get(n))
        participation_coef.append(different / len(neigh))

    # Homophily.
    same_genre_share = _prop_same_attr(g, node_order, "primary_genre", neighbors)
    same_label_share = _prop_same_attr(g, node_order, "primary_label", neighbors)
    same_role_share = _prop_same_attr(g, node_order, "primary_role", neighbors)

    # Triangle compositions.
    triangle_list = list(_enumerate_triangles(g, node_to_idx))
    tri_within_genre = _triangles_with_same_attr(g, node_order, triangle_list, "primary_genre")
    tri_within_label = _triangles_with_same_attr(g, node_order, triangle_list, "primary_label")
    tri_within_role = _triangles_with_same_attr(g, node_order, triangle_list, "primary_role")

    tri_cross_genre = _triangles_with_all_diff_attr(g, node_order, triangle_list, "primary_genre")
    tri_cross_label = _triangles_with_all_diff_attr(g, node_order, triangle_list, "primary_label")
    tri_cross_role = _triangles_with_all_diff_attr(g, node_order, triangle_list, "primary_role")

    # Component size.
    comp_size_map = {}
    for comp in nx.connected_components(g):
        size = len(comp)
        for node in comp:
            comp_size_map[node] = size
    component_size = np.array([comp_size_map[n] for n in node_order], dtype=float)

    # Open dyad opportunities.
    open_dyads = []
    for n in node_order:
        onehop = neighbors[n]
        twohop = set(onehop)
        for nb in onehop:
            twohop.update(neighbors.get(nb, set()))
        twohop.add(n)
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
