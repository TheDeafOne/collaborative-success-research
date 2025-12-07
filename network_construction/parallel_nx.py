import networkx as nx
import pandas as pd
from tqdm import tqdm
import logging
import multiprocessing as mp
import networkx as nx
import numpy as np
from time import time
import networkit as nk

# Optional: Louvain
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


# -------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def log_step(step_name):
    logging.info(f"⬆️ START: {step_name}")
    start = time()
    return start

def log_end(step_name, start):
    end = time()
    logging.info(f"⬇️ END: {step_name} (elapsed {end - start:.2f}s)\n")


# -------------------------------------------------------------
# Parallel degree summary
# -------------------------------------------------------------
def degree_worker(sub_nodes, G):
    """Worker: compute degrees for a subset of nodes."""
    return [G.degree(n) for n in sub_nodes]

def degree_distribution_parallel(G, workers=mp.cpu_count()):
    """Compute degree summary in parallel."""
    step = "Degree distribution summary"
    t0 = log_step(step)

    nodes = list(G.nodes())
    chunks = np.array_split(nodes, workers)

    with mp.Pool(processes=workers) as pool:
        results = pool.starmap(degree_worker, [(chunk, G) for chunk in chunks])

    degrees = np.concatenate(results)

    summary = {
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)),
        "var_degree": float(np.var(degrees))
    }

    log_end(step, t0)
    return summary


# -------------------------------------------------------------
# Fast approximate diameter (double sweep BFS)
# -------------------------------------------------------------
def bfs_farthest(G, source):
    """Return farthest node from `source` using BFS."""
    dist = nx.single_source_shortest_path_length(G, source)
    farthest_node = max(dist, key=dist.get)
    farthest_dist = dist[farthest_node]
    return farthest_node, farthest_dist


def approximate_diameter(G):
    """
    Pseudo-diameter using double sweep method.
    Very fast and widely used for large graphs.
    """
    step = "Approximate diameter (double sweep BFS)"
    t0 = log_step(step)

    # Start from an arbitrary node in LCC
    start_node = next(iter(G.nodes()))

    # BFS 1
    v, _ = bfs_farthest(G, start_node)

    # BFS 2
    w, dist = bfs_farthest(G, v)

    log_end(step, t0)
    return dist  # approximate diameter


# -------------------------------------------------------------
# Largest connected component
# -------------------------------------------------------------
def largest_cc(G):
    step = "Largest connected component"
    t0 = log_step(step)

    largest = max(nx.connected_components(G), key=len)
    LCC = G.subgraph(largest).copy()

    log_end(step, t0)
    return LCC


# -------------------------------------------------------------
# Global clustering (transitivity)
# -------------------------------------------------------------
def global_clustering(G):
    step = "Global clustering coefficient"
    t0 = log_step(step)

    c = nx.transitivity(G)

    log_end(step, t0)
    return c


# -------------------------------------------------------------
# Modularity
# -------------------------------------------------------------
# def modularity_score(G):
#     step = "Modularity"
#     t0 = log_step(step)

#     comms = nx.algorithms.community.greedy_modularity_communities(G)
#     Q = nx.algorithms.community.modularity(G, comms)
#     method = "Greedy"
#     num_comms = len(comms)

#     log_end(step, t0)
#     return Q, method, num_comms


def modularity_score(G):
    step = "Modularity (NetworKit PLM Louvain)"
    t0 = log_step(step)

    # ----------------------------------------
    # Convert NetworkX → NetworKit
    # ----------------------------------------
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G_nk = nk.graph.Graph(n=len(mapping), weighted=False, directed=False)

    for u, v in tqdm(G.edges(), desc="Adding edges for modularity", unit="edge", leave=False):
        G_nk.addEdge(mapping[u], mapping[v])

    # ----------------------------------------
    # Run PLM (Parallel Louvain)
    # ----------------------------------------
    plm = nk.community.PLM(G_nk)
    result = plm.run()

    # Partition object
    part = plm.getPartition()

    # Modularity score
    Q = nk.community.Modularity().getQuality(part, G_nk)

    # Number of detected communities
    num_comms = part.numberOfSubsets()

    log_end(step, t0)

    return Q, "PLM-Louvain", num_comms
# -------------------------------------------------------------
# Main orchestrating function
# -------------------------------------------------------------
def compute_all_network_metrics(G):
    logging.info("========== BEGIN NETWORK ANALYSIS ==========")

    # |V| and |E|
    # t0 = log_step("Count nodes and edges")
    # n, m = G.number_of_nodes(), G.number_of_edges()
    # log_end("Count nodes and edges", t0)
    # logging.info(f"|V| = {n}, |E| = {m}")

    # # Density
    # t0 = log_step("Density")
    # density = nx.density(G)
    # log_end("Density", t0)
    # logging.info(f"Density = {density:.6f}")

    # # Largest connected component
    # LCC = largest_cc(G)
    # logging.info(f"LCC size = {LCC.number_of_nodes()} nodes")

    # # Approximate diameter
    # approx_diam = approximate_diameter(LCC)
    # logging.info(f"Approximate diameter = {approx_diam}")

    # # Average path length (still expensive!)
    # # t0 = log_step("Average shortest path length (LCC)")
    # # avg_path = approx_average_shortest_path_length(G)
    # # log_end("Average shortest path length (LCC)", t0)
    # # logging.info(f"Average shortest path length = {avg_path:.4f}")

    # # Parallel degree distribution
    # deg_summary = degree_distribution_parallel(G)
    # logging.info("Degree summary:")
    # for k, v in deg_summary.items():
    #     logging.info(f"  {k}: {v}")

    # # Global clustering
    # gcc = global_clustering(G)
    # logging.info(f"Global clustering coefficient = {gcc:.6f}")

    # Modularity
    Q, method, k = modularity_score(G)
    logging.info(f"Modularity ({method}) = {Q:.5f} across {k} communities")

    logging.info("========== END NETWORK ANALYSIS ==========\n")

    return {
        # "num_nodes": n,
        # "num_edges": m,
        # "density": density,
        # "LCC_size": LCC.number_of_nodes(),
        # "approx_diameter": approx_diam,
        # # "avg_shortest_path": avg_path,
        # "degree_summary": deg_summary,
        # "global_clustering": gcc,
        "modularity": Q,
        "modularity_method": method,
        "num_communities": k
    }



def approx_average_shortest_path_length(G):
    """
    Fast, scalable approximation of average shortest path length using NetworKit.
    Accepts a NetworkX graph, converts it internally.

    Works well for graphs with 100k–10M nodes.
    """

    # ---------------------------
    # Convert NetworkX → NetworKit
    # ---------------------------
    logging.info("⬆️ START: Convert NetworkX → NetworKit (for ASPL)")
    t0 = time()

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G_nk = nk.graph.Graph(n=len(mapping), weighted=False, directed=False)

    for u, v in tqdm(G.edges(), desc="Adding edges", unit="edge"):
        G_nk.addEdge(mapping[u], mapping[v])

    logging.info(f"⬇️ END: Conversion complete (elapsed {time() - t0:.2f}s)\n")

    # ---------------------------
    # Approximate APSP
    # ---------------------------
    logging.info("⬆️ START: Approximate average shortest path length (NetworKit)")
    t0 = time()

    approx = nk.distance.ApproximateAPSP(G_nk).run()
    distances = approx.getDistances()

    # Remove unreachable (inf) distances
    finite = [d for d in distances if d < float("inf")]
    avg = sum(finite) / len(finite)

    logging.info(
        f"⬇️ END: ASPL = {avg:.6f} (elapsed {time() - t0:.2f}s)\n"
    )

    return avg

def get_network(nodes: pd.DataFrame,
    edges: pd.DataFrame):
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

    print("[node2vec] adding nodes")
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), disable=True, desc="[node2vec] nodes"):
        attrs = row.to_dict()
        node_id = attrs.pop("mbid")
        G.add_node(node_id, **attrs)

    print("[node2vec] adding edges")
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), disable=True, desc="[node2vec] edges"):
        u = str(row["u"])
        v = str(row["v"])
        weight = row.get('weight', 1.0)
        if pd.isna(weight):
            weight = 1.0
        G.add_edge(u, v, weight=weight)
    return G

# -----------------------------
# Main orchestrating function
# -----------------------------

def main():
    print("=== Network Summary ===")
    nodes = pd.read_csv('../data/graphs/only_connected/nodes.csv')
    edges = pd.read_csv('../data/graphs/only_connected/edges.csv')
    G = get_network(nodes, edges)

    d = compute_all_network_metrics(G)
    print(d)


if __name__ == "__main__":
    main()

# Example usage:
# G = nx.read_edgelist("my_graph.edgelist")
# compute_all_network_metrics(G)


# |V| = 243233, |E| = 1426829
# Density = 0.000048
# LCC size = 223031 nodes
# Approximate diameter = 19
# Degree summary:
#   mean_degree: 11.732199167053812
#   median_degree: 4.0
#   max_degree: 7557
#   var_degree: 1716.715190740333
# Global clustering coefficient = 0.044976
# {'modularity': 0.705152051173847, 'modularity_method': 'PLM-Louvain', 'num_communities': 7137}