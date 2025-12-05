import pandas as pd
import networkx as nx
from node2vec import Node2Vec  # from the node2vec package

# ---------------------------------------------------
# 1. Load your data (if you already have DataFrames, skip this part)
# ---------------------------------------------------
nodes_df = pd.read_csv("nodes.csv")  # has 'mbid', 'name', etc.
edges_df = pd.read_csv("edges.csv")  # has 'u', 'v', 'weight_size_adj', ...

# ---------------------------------------------------
# 2. Build a graph from edges_df
# ---------------------------------------------------
# Use an undirected graph – change to nx.DiGraph() if direction matters
G = nx.Graph()

# Optionally, you can add node attributes from nodes_df
for _, row in nodes_df.iterrows():
    node_id = row["mbid"]
    # Add node with attributes (all other columns become attributes)
    attrs = row.to_dict()
    G.add_node(node_id, **attrs)

# Add edges with weights (you can choose any edge feature as weight)
# Here we use weight_size_adj; fall back to 1.0 if NaN
for _, row in edges_df.iterrows():
    u = row["u"]
    v = row["v"]
    weight = row.get("weight_size_adj", 1.0)
    if pd.isna(weight):
        weight = 1.0
    G.add_edge(u, v, weight=weight)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# ---------------------------------------------------
# 3. Run node2vec
# ---------------------------------------------------
# Key hyperparameters:
#   dimensions: size of embedding vector
#   walk_length, num_walks: control context / training data size
#   p, q: return/in-out parameters (p=1,q=1 -> plain DeepWalk)
node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=30,
    num_walks=200,
    p=1.0,
    q=1.0,
    workers=4,           # adjust to your CPU
    weight_key="weight", # use the edge attribute 'weight'
    quiet=True
)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# ---------------------------------------------------
# 4. Build a node embedding DataFrame
# ---------------------------------------------------
# model.wv.index_to_key are the node IDs as strings; convert if needed
embeddings = []
for node in G.nodes():
    # node2vec model stores node keys as strings by default
    key = str(node)
    if key in model.wv:
        vec = model.wv[key]
        embeddings.append((node, vec))

# Convert to DataFrame: one row per node, columns: ['mbid', 'n2v_0', 'n2v_1', ...]
embed_df = pd.DataFrame(
    [
        [node] + vec.tolist()
        for node, vec in embeddings
    ],
    columns=["mbid"] + [f"n2v_{i}" for i in range(model.vector_size)]
)

# ---------------------------------------------------
# 5. Join embeddings back to your original node features
# ---------------------------------------------------
nodes_with_embed = nodes_df.merge(embed_df, on="mbid", how="left")

print(nodes_with_embed.head())
