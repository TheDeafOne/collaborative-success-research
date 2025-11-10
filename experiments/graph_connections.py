#!/usr/bin/env python3
"""
Build a weighted co-writer graph from a JSONL of recordings.

Each line in the JSONL looks like:
{
  "recording_id": "...",
  "recording_title": "...",
  "authors": [
     {"name": "...", "artist_mbid": "...", "relation_type": "writer", ...},
     ...
  ],
  "works": [...]
}

Edges connect writers who co-wrote a song; edge weight = # songs together.

Outputs:
- <prefix>_writer_collab_graph.png
- <prefix>_writer_collaborations_edges.csv
- <prefix>_writer_collaborations_nodes.csv
- (optional) <prefix>_writer_collab_graph.html  (interactive, with --html)
"""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def load_records(jsonl_path: Path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def person_key(author: dict) -> str:
    """Prefer stable MBID; fall back to name."""
    return author.get("artist_mbid") or author.get("name")


def build_graph(records):
    """Return (Graph, nodes_df, edges_df)."""
    # Collect song-level sets of unique writers
    songs = []
    for rec in records:
        writers = [a for a in rec.get("authors", []) if a.get("relation_type") == "writer"]
        dedup = {}
        for a in writers:
            dedup[person_key(a)] = a.get("name")  # keep display name
        if dedup:
            songs.append({
                "recording_id": rec.get("recording_id"),
                "recording_title": rec.get("recording_title"),
                "authors": dedup,  # dict: key -> display name
            })

    # Count co-occurrences across songs
    edge_weights = Counter()
    pair_examples = defaultdict(list)
    for s in songs:
        keys = sorted(s["authors"].keys())
        for u, v in itertools.combinations(keys, 2):
            edge = (u, v)
            edge_weights[edge] += 1
            pair_examples[edge].append(s["recording_title"])

    # Build graph
    G = nx.Graph()
    # Add nodes
    for s in songs:
        for k, name in s["authors"].items():
            if k not in G:
                G.add_node(k, label=name)
    # Add edges (weighted)
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w, songs=sorted(set(pair_examples[(u, v)])))

    # Node metrics
    weighted_degree = dict(G.degree(weight="weight"))
    unique_collabs = {n: G.degree(n) for n in G.nodes()}

    nodes_df = pd.DataFrame({
        "id": list(G.nodes()),
        "name": [G.nodes[n]["label"] for n in G.nodes()],
        "weighted_degree": [weighted_degree.get(n, 0) for n in G.nodes()],
        "unique_collaborators": [unique_collabs.get(n, 0) for n in G.nodes()],
        "betweenness": pd.Series(nx.betweenness_centrality(G, weight=None))  # simple unweighted betweenness
    }).sort_values(
        ["weighted_degree", "unique_collaborators", "betweenness", "name"],
        ascending=[False, False, False, True]
    )

    edges_df = pd.DataFrame([
        {
            "source_id": u,
            "source": G.nodes[u]["label"],
            "target_id": v,
            "target": G.nodes[v]["label"],
            "weight": data["weight"],
            "example_songs": "; ".join(data.get("songs", [])),
        }
        for u, v, data in G.edges(data=True)
    ]).sort_values(["weight", "source", "target"], ascending=[False, True, True])

    return G, nodes_df, edges_df


def draw_png(G: nx.Graph, out_png: Path, title="Co-writer Network (edge weight = # songs together)"):
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.7)

    # Node sizes scaled by weighted degree
    weighted_degree = dict(G.degree(weight="weight"))
    deg_series = pd.Series(weighted_degree)
    min_size, max_size = 500, 2500
    if len(deg_series) and deg_series.max() != deg_series.min():
        node_sizes = [
            min_size + (deg_series[n] - deg_series.min()) /
            (deg_series.max() - deg_series.min()) * (max_size - min_size)
            for n in G.nodes()
        ]
    else:
        node_sizes = [min_size for _ in G.nodes()]

    # Edge widths scaled by weight
    edge_widths = [1 + G[u][v]["weight"] * 2 for u, v in G.edges()]

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.75)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=9)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def export_csv(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, out_nodes: Path, out_edges: Path):
    nodes_df.to_csv(out_nodes, index=False)
    edges_df.to_csv(out_edges, index=False)


def export_html_interactive(G: nx.Graph, out_html: Path, title="Co-writer Network"):
    """
    Export an interactive HTML with PyVis, supplying our own Jinja2 template
    to avoid the 'template is None' error on some installs.
    """
    try:
        from pyvis.network import Network
        from jinja2 import Template
        import json
    except ImportError:
        print("PyVis/Jinja2 not installed. Run: pip install pyvis jinja2")
        return

    # Build PyVis network
    net = Network(height="750px", width="100%", notebook=False, bgcolor="#ffffff", font_color="#222")
    net.barnes_hut()

    weighted_degree = dict(G.degree(weight="weight"))

    for n, data in G.nodes(data=True):
        net.add_node(
            n,
            label=data.get("label", n),
            title=f"{data.get('label', n)}\nWeighted degree: {weighted_degree.get(n, 0)}",
            value=weighted_degree.get(n, 0),
        )

    for u, v, data in G.edges(data=True):
        title_txt = f"Weight: {data.get('weight', 1)}"
        songs = data.get("songs")
        if songs:
            title_txt += "\\nSongs: " + "; ".join(songs)
        net.add_edge(u, v, value=data.get("weight", 1), title=title_txt)

    # Safe JSON options
    options = {
        "edges": {"smooth": {"type": "dynamic"}},
        "physics": {
            "stabilization": {"iterations": 150},
            "barnesHut": {"gravitationalConstant": -4500}
        },
        "nodes": {"shape": "dot", "scaling": {"min": 5, "max": 40}},
        "interaction": {"tooltipDelay": 100}
    }
    options_json = json.dumps(options)

    # Provide our own minimal Jinja2 template to avoid None template
    html_template = Template(r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    #mynetwork { width: 100%; height: 750px; border: 1px solid #eee; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji", sans-serif; }
  </style>
</head>
<body>
  <h2 style="margin:16px 0 8px 0;">{{ title }}</h2>
  <div id="mynetwork"></div>
  <script type="text/javascript">
    const nodes = new vis.DataSet({{ nodes | safe }});
    const edges = new vis.DataSet({{ edges | safe }});
    const container = document.getElementById('mynetwork');
    const data = { nodes: nodes, edges: edges };
    const options = {{ options | safe }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>
""")

    # Attach our template and write HTML
    net.template = html_template
    # Ensure PyVis has the serialized node/edge/options it expects:
    net.write_html(out_html.as_posix(), open_browser=False, notebook=False,
                   nodes=True, edges=True, options=options_json, title=title)
    print(f"Wrote HTML: {out_html}")


def export_html_standalone(G: nx.Graph, out_html: Path, title="Co-writer Network"):
    """
    Write a standalone interactive HTML using vis-network (no PyVis dependency).
    Nodes are sized by weighted degree; edges weighted by # songs together.
    """
    weighted_degree = dict(G.degree(weight="weight"))

    # Build serializable node/edge lists for vis-network
    nodes = []
    for n, data in G.nodes(data=True):
        nodes.append({
            "id": n,
            "label": data.get("label", n),
            "title": f"{data.get('label', n)}\nWeighted degree: {weighted_degree.get(n, 0)}",
            "value": weighted_degree.get(n, 0)  # controls node size
        })

    edges = []
    for u, v, data in G.edges(data=True):
        title_txt = f"Weight: {data.get('weight', 1)}"
        songs = data.get("songs")
        if songs:
            title_txt += "\\nSongs: " + "; ".join(songs)
        edges.append({
            "from": u, "to": v,
            "value": data.get("weight", 1),   # controls edge thickness
            "title": title_txt
        })

    options = {
        "edges": {"smooth": {"type": "dynamic"}},
        "physics": {
            "stabilization": {"iterations": 150},
            "barnesHut": {"gravitationalConstant": -4500}
        },
        "nodes": {"shape": "dot", "scaling": {"min": 5, "max": 40}},
        "interaction": {"tooltipDelay": 100}
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    #mynetwork {{ width: 100%; height: 750px; border: 1px solid #eee; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji", sans-serif; margin: 16px; }}
    h2 {{ margin-top: 0; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div id="mynetwork"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const container = document.getElementById('mynetwork');
    const data = {{ nodes, edges }};
    const options = {json.dumps(options)};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {out_html}")

def main():
    ap = argparse.ArgumentParser(description="Build a weighted co-writer graph from JSONL.")
    ap.add_argument("jsonl", type=Path, help="Path to input JSONL")
    ap.add_argument("--prefix", type=str, default=None,
                    help="Output filename prefix (default: input stem)")
    ap.add_argument("--html", action="store_true",
                    help="Also export an interactive HTML (requires pyvis)")
    args = ap.parse_args()

    jsonl_path: Path = args.jsonl
    prefix = args.prefix or jsonl_path.stem
    out_dir = jsonl_path.parent

    # Load and process
    records = load_records(jsonl_path)
    G, nodes_df, edges_df = build_graph(records)

    # Outputs
    png_path = out_dir / f"{prefix}_writer_collab_graph.png"
    nodes_csv = out_dir / f"{prefix}_writer_collaborations_nodes.csv"
    edges_csv = out_dir / f"{prefix}_writer_collaborations_edges.csv"

    draw_png(G, png_path)
    export_csv(nodes_df, edges_df, nodes_csv, edges_csv)
    print(f"Wrote PNG:  {png_path}")
    print(f"Wrote CSVs: {nodes_csv} , {edges_csv}")

    if args.html:
        html_path = out_dir / f"{prefix}_writer_collab_graph.html"
        export_html_standalone(G, html_path)


if __name__ == "__main__":
    main()
