from __future__ import annotations

import json
from math import exp, log
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def _parse_date_to_ordinal(d: Optional[str]) -> Optional[int]:
    """Parse a date string to a `date.toordinal()` integer, or None."""
    if not d or not isinstance(d, str):
        return None
    d = d.strip()
    if not d:
        return None
    from datetime import datetime

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(d, fmt)
            return dt.date().toordinal()
        except ValueError:
            continue
    return None


def _to_roles(x) -> List[str]:
    """Normalize roles to a lowercased list of strings."""
    if not x:
        return []
    if isinstance(x, str):
        parts = x.split(",")
    else:
        parts = x
    return [str(r).strip().lower() for r in parts if str(r).strip()]


def _compute_work_release_ordinals(artist: dict) -> Dict[int, Optional[int]]:
    """
    For a single artist dict, compute earliest known release date per work_id
    as an ordinal. Uses both work.first_release_date and releases via recordings.
    """
    # release_id -> date_ordinal
    release_date_by_id: Dict[int, Optional[int]] = {}
    for rel in artist.get("releases", []) or []:
        rid = rel.get("id")
        if rid is None:
            continue
        release_date_by_id[rid] = _parse_date_to_ordinal(rel.get("date"))

    # work_id -> earliest release date
    work_date: Dict[int, Optional[int]] = {}
    for rec in artist.get("recordings", []) or []:
        wid = rec.get("work_id")
        rid = rec.get("release_id")
        if wid is None or rid is None:
            continue
        rel_ord = release_date_by_id.get(rid)
        if rel_ord is None:
            continue
        cur = work_date.get(wid)
        if cur is None or rel_ord < cur:
            work_date[wid] = rel_ord

    # combine with works.first_release_date
    for w in artist.get("works", []) or []:
        wid = w.get("id")
        if wid is None:
            continue
        first_ord = _parse_date_to_ordinal(w.get("first_release_date"))
        cur = work_date.get(wid)
        if first_ord is not None and (cur is None or first_ord < cur):
            work_date[wid] = first_ord
        elif wid not in work_date:
            work_date[wid] = None

    return work_date


def build_cowrite_graph_streaming(
    features_df: pd.DataFrame,
    artist_dir: Path | str,
    recency_half_life_years: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Streaming co-write graph builder.

    Parameters
    ----------
    features_df : pd.DataFrame
        Per-artist features. Must contain at least:
            - 'artist_mbid' (unique id, used as node key)
            - 'debut_date' (optional, YYYY-MM-DD or datetime)
            - 'window_cutoff_date' (YYYY-MM-DD or datetime, used as end of window)
            - 'primary_genre', 'artist_country', 'artist_region_city',
              'primary_label'
            - 'all_roles_str', 'all_genres_str', 'all_labels_str'
              (comma-separated lists; used for Jaccard overlaps)
    artist_dir : Path | str
        Directory containing .jsonl files with artist work data.
    recency_half_life_years : float
        Half-life for exponential recency decay on edges.

    Returns
    -------
    edges_df : pd.DataFrame
        Columns:
            ['u', 'v',
             'weight_raw', 'weight_size_adj',
             'first_collab_date', 'last_collab_date',
             'collab_span_years', 'recency_weight',
             'same_primary_genre', 'same_country',
             'same_region_city', 'same_primary_label',
             'roles_overlap', 'genres_overlap', 'labels_overlap',
             'mean_team_size_cowrite_joint']
    nodes_df : pd.DataFrame
        Exactly `features_df` (copied), with no added graph-level features.
    """
    artist_dir = Path(artist_dir)

    # ---- 1) Node indexing & window boundaries --------------------------------
    nodes_df = features_df.copy().reset_index(drop=True)
    if "artist_mbid" not in nodes_df.columns:
        raise ValueError("features_df must have an 'artist_mbid' column.")

    n_artists = len(nodes_df)

    mbids = nodes_df["artist_mbid"].astype(str).tolist()
    mbid_to_idx = {mbid: i for i, mbid in enumerate(mbids)}
    idx_to_mbid = mbids

    # debut / cutoff ordinals
    debut_ord: List[Optional[int]] = [None] * n_artists
    cutoff_ord: List[Optional[int]] = [None] * n_artists

    if "debut_date" in nodes_df.columns:
        debut_series = pd.to_datetime(nodes_df["debut_date"], errors="coerce")
    else:
        debut_series = pd.Series([pd.NaT] * n_artists)

    if "window_cutoff_date" in nodes_df.columns:
        cutoff_series = pd.to_datetime(nodes_df["window_cutoff_date"], errors="coerce")
    else:
        cutoff_series = pd.Series([pd.NaT] * n_artists)

    for i, (d, c) in enumerate(zip(debut_series, cutoff_series)):
        debut_ord[i] = d.date().toordinal() if pd.notna(d) else None
        cutoff_ord[i] = c.date().toordinal() if pd.notna(c) else None

    def in_window(idx: int, work_ord: int) -> bool:
        """Check if a work date is within the artist's window."""
        c = cutoff_ord[idx]
        if c is not None and work_ord > c:
            return False
        # Optional: enforce work_ord >= debut_ord[idx]
        return True

    # Node metadata for edge covariates
    country = nodes_df.get("artist_country", pd.Series([None] * n_artists)).tolist()
    region = nodes_df.get("artist_region_city", pd.Series([None] * n_artists)).tolist()
    prim_genre = nodes_df.get("primary_genre", pd.Series([None] * n_artists)).tolist()
    prim_label = nodes_df.get("primary_label", pd.Series([None] * n_artists)).tolist()

    # Parse comma-separated role/genre/label strings into sets for Jaccard
    def _series_to_set_list(col: str) -> List[set]:
        if col not in nodes_df.columns:
            return [set() for _ in range(n_artists)]
        s = nodes_df[col].fillna("")
        out: List[set] = []
        for v in s:
            if not isinstance(v, str) or not v.strip():
                out.append(set())
            else:
                tokens = {p.strip().lower() for p in v.split(",") if p.strip()}
                out.append(tokens)
        return out

    roles_sets = _series_to_set_list("all_roles_str")
    genres_sets = _series_to_set_list("all_genres_str")
    labels_sets = _series_to_set_list("all_labels_str")

    # ---- 2) Edge stats containers -------------------------------------------
    # pair_stats[(u_idx, v_idx)] = [weight_raw, weight_size_adj,
    #                               first_ord, last_ord, sum_team_size, joint_count]
    pair_stats: Dict[Tuple[int, int], List] = {}

    # Per-artist last release ordinal in-window (for recency reference)
    last_ord: List[Optional[int]] = [None] * n_artists

    LAMBDA = log(2.0) / recency_half_life_years if recency_half_life_years > 0 else 0.0

    COWRITE_ROLES = {
        "writer", "composer", "lyricist", "librettist", "scriptwriter", "translator",
        "arranger", "instrument arranger", "orchestrator", "vocal arranger",
        "adapter", "revised by", "reconstructed by",
    }

    # ---- 3) Streaming pass over all artist JSONL files ----------------------
    files = sorted(artist_dir.glob("*.jsonl"))
    outer_bar = tqdm(files, desc="Artist files", unit="file")

    for path in outer_bar:
        try:
            with path.open("r", encoding="utf-8") as f:
                inner_bar = tqdm(
                    f,
                    desc=f"{path.name}",
                    unit="artist",
                    leave=False,
                )

                for line in inner_bar:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        artist = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    a_mbid = artist.get("mbid")
                    if not a_mbid:
                        continue

                    a_roles = set(_to_roles(artist.get("roles")))
                    work_date_ord = _compute_work_release_ordinals(artist)
                    works = artist.get("works", []) or []

                    for w in works:
                        wid = w.get("id")
                        if wid is None:
                            continue
                        work_ord = work_date_ord.get(wid)
                        if work_ord is None:
                            continue

                        # Collect participants and their roles
                        participants_roles: Dict[str, set] = {}
                        participants_roles[a_mbid] = set(a_roles)

                        for c in w.get("collaborators", []) or []:
                            cm = c.get("mbid")
                            if not cm:
                                continue
                            cr = set(_to_roles(c.get("roles")))
                            if not cr:
                                continue
                            if cm in participants_roles:
                                participants_roles[cm].update(cr)
                            else:
                                participants_roles[cm] = cr

                        if not participants_roles:
                            continue

                        # Dedup: only lexicographically smallest mbid owns this work
                        if a_mbid != min(participants_roles.keys()):
                            continue

                        # Build team
                        cowwriters_all_mbids: List[str] = []
                        cowwriters_idx: List[int] = []  # <-- correctly defined here

                        for mbid, roles in participants_roles.items():
                            if not (roles & COWRITE_ROLES):
                                continue

                            cowwriters_all_mbids.append(mbid)

                            idx = mbid_to_idx.get(mbid)
                            if idx is None:
                                continue  # collaborator not in our node set

                            if not in_window(idx, work_ord):
                                continue

                            cowwriters_idx.append(idx)

                            # Track last in-window date per artist
                            if last_ord[idx] is None or work_ord > last_ord[idx]:
                                last_ord[idx] = work_ord

                        n_full = len(cowwriters_all_mbids)
                        if n_full < 2:
                            continue
                        if len(cowwriters_idx) < 2:
                            continue

                        denom = n_full * (n_full - 1) / 2.0
                        adj_inc = 1.0 / denom if denom > 0 else 0.0

                        for i in range(len(cowwriters_idx)):
                            ui = cowwriters_idx[i]
                            for j in range(i + 1, len(cowwriters_idx)):
                                vi = cowwriters_idx[j]
                                if ui == vi:
                                    continue
                                u_idx, v_idx = (ui, vi) if ui < vi else (vi, ui)
                                key = (u_idx, v_idx)

                                stats = pair_stats.get(key)
                                if stats is None:
                                    stats = [0, 0.0, work_ord, work_ord, 0, 0]
                                    pair_stats[key] = stats

                                stats[0] += 1
                                stats[1] += adj_inc
                                if work_ord < stats[2]:
                                    stats[2] = work_ord
                                if work_ord > stats[3]:
                                    stats[3] = work_ord
                                stats[4] += n_full
                                stats[5] += 1

                inner_bar.close()
        except Exception:
            # If a file blows up, keep going; tqdm will still move on.
            continue

    outer_bar.close()

    # ---- 4) Convert pair_stats to edges_df ----------------------------------
    def years_between(o1: int, o2: int) -> float:
        return abs(o2 - o1) / 365.25

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        u = a | b
        if not u:
            return 0.0
        return len(a & b) / len(u)

    edges_rows: List[dict] = []
    total_pairs = len(pair_stats)

    for (u_idx, v_idx), (w_raw, w_adj, first_c, last_c, sum_team, joint_count) in tqdm(
        pair_stats.items(),
        desc="Aggregating edges",
        total=total_pairs,
        unit="edge",
    ):
        collab_span_years = (
            years_between(first_c, last_c)
            if first_c is not None and last_c is not None
            else 0.0
        )

        lu = last_ord[u_idx]
        lv = last_ord[v_idx]
        if LAMBDA and lu is not None and lv is not None:
            ref_edge = min(lu, lv)
            if ref_edge >= last_c:
                years_since = years_between(last_c, ref_edge)
                recency_weight = float(exp(-LAMBDA * years_since))
            else:
                recency_weight = 0.0
        else:
            recency_weight = 0.0

        cu, cv = country[u_idx], country[v_idx]
        ru, rv = region[u_idx], region[v_idx]
        gu, gv = prim_genre[u_idx], prim_genre[v_idx]
        lu_label, lv_label = prim_label[u_idx], prim_label[v_idx]

        same_country = int(bool(cu and cv and cu == cv))
        same_region = int(bool(ru and rv and ru == rv))
        same_genre = int(bool(gu and gv and gu == gv))
        same_label = int(bool(lu_label and lv_label and lu_label == lv_label))

        mean_team_size = float(sum_team / joint_count) if joint_count > 0 else 0.0

        # Jaccard overlaps from precomputed string columns
        roles_overlap = jaccard(roles_sets[u_idx], roles_sets[v_idx])
        genres_overlap = jaccard(genres_sets[u_idx], genres_sets[v_idx])
        labels_overlap = jaccard(labels_sets[u_idx], labels_sets[v_idx])

        edges_rows.append({
            "u": idx_to_mbid[u_idx],
            "v": idx_to_mbid[v_idx],
            "weight_raw": int(w_raw),
            "weight_size_adj": float(w_adj),
            "first_collab_date": first_c,
            "last_collab_date": last_c,
            "collab_span_years": round(collab_span_years, 6),
            "recency_weight": round(recency_weight, 6),
            "same_primary_genre": same_genre,
            "same_country": same_country,
            "same_region_city": same_region,
            "same_primary_label": same_label,
            "roles_overlap": round(roles_overlap, 6),
            "genres_overlap": round(genres_overlap, 6),
            "labels_overlap": round(labels_overlap, 6),
            "mean_team_size_cowrite_joint": round(mean_team_size, 3),
        })

    edges_df = pd.DataFrame(edges_rows)

    # Convert ordinals back to ISO for readability
    from datetime import date as _date

    def ord_to_iso(o: Optional[int]) -> Optional[str]:
        return _date.fromordinal(o).isoformat() if o is not None else None

    if not edges_df.empty:
        edges_df["first_collab_date"] = edges_df["first_collab_date"].map(ord_to_iso)
        edges_df["last_collab_date"] = edges_df["last_collab_date"].map(ord_to_iso)

        int_cols = [
            "weight_raw",
            "same_primary_genre",
            "same_country",
            "same_region_city",
            "same_primary_label",
        ]
        for col in int_cols:
            edges_df[col] = edges_df[col].astype("Int64")

    # nodes_df is just your features_df, unchanged (apart from reset index)
    return edges_df, nodes_df

def main():
    from pathlib import Path
    import pandas as pd
    import os

    features = pd.read_csv("./data/artist_base_features.csv")  # your 40MB file
    edges_df, nodes_df = build_cowrite_graph_streaming(
        features_df=features,
        artist_dir=Path("./data/artist_full_data"),
        recency_half_life_years=3.0,
    )
    os.makedirs('./data/graphs/better_graph', exist_ok=True)

    edges_df.to_csv('./data/graphs/better_graph/edges.csv', index=False)

if __name__ == "__main__":
    main()