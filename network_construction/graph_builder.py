import pandas as pd
from itertools import combinations
from collections import defaultdict, Counter
from datetime import datetime, date
import math

def build_graph_with_bipartite(
    artists,
    first_n_songs=None,
    first_m_months=None,          # NEW: months-based window
    recency_half_life_years=3.0,  # for recency_weight via last-collab
):
    """
    Build:
      1) artist_song_df (bipartite credits, one row per credit)
      2) edges_df (artist–artist projection with weights & covariates)
      3) nodes_df (artist attributes within the per-artist window)

    Windowing (choose ONE):
      - first_n_songs: keep the first N works (by release date when available) per focal artist
      - first_m_months: keep works released within m months from the artist's first dated release

    Returns
    -------
    artist_song_df : pd.DataFrame
        [artist_mbid, artist_name, song_id, song_name, credit_role, release_date,
         team_size, team_size_cowrite, genre_tags]
    edges_df : pd.DataFrame
        [u, v, weight_raw, weight_size_adj, first_collab_date, last_collab_date,
         recency_weight, roles_overlap, genre_overlap]
         # recency_weight uses per-edge reference = min(last_date_u, last_date_v)
    nodes_df : pd.DataFrame
        [mbid, name, all_roles, all_genres, first_release_date_in_window,
         last_release_date_in_window, num_songs_in_window, num_collaborators_in_window,
         time_in_network_years]
    """

    # -----------------------------
    # 0) Validate window arguments
    # -----------------------------
    if (first_n_songs is not None) and (first_m_months is not None):
        raise ValueError("Provide only one of first_n_songs OR first_m_months.")
    if (first_n_songs is None) and (first_m_months is None):
        # allowed: means use all works (no truncation)
        pass

    # -----------------------------
    # 1) Role settings & utilities
    # -----------------------------
    CREDIT_ROLES = {
        "writer", "composer", "lyricist", "librettist", "scriptwriter", "translator",
        "arranger", "instrument arranger", "orchestrator", "vocal arranger",
        "adapter", "revised by", "reconstructed by",
        "producer", "engineer", "artist", "singer", "vocal"
    }
    COWRITE_ROLES = {
        "writer", "composer", "lyricist", "librettist", "scriptwriter", "translator",
        "arranger", "instrument arranger", "orchestrator", "vocal arranger",
        "adapter", "revised by", "reconstructed by",
    }

    def to_roles(x):
        if not x:
            return []
        if isinstance(x, str):
            return [r.strip().lower() for r in x.split(",") if r.strip()]
        return [str(r).strip().lower() for r in x if str(r).strip()]

    def parse_date(d):
        if not d:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(d, fmt)
            except ValueError:
                continue
        return None

    def ydiff(d1, d2):
        return abs((d2 - d1).days) / 365.25

    def nC2(n):
        return (n * (n - 1)) / 2 if n and n >= 2 else 0

    def add_months(dt: date, m: int) -> date:
        """Add m months to a date, capping day to end-of-month when needed."""
        y = dt.year + (dt.month - 1 + m) // 12
        mo = (dt.month - 1 + m) % 12 + 1
        # end-of-month handling
        last_day = [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                    31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]
        day = min(dt.day, last_day)
        return date(y, mo, day)

    # Lambda for recency decay
    LAMBDA = math.log(2) / recency_half_life_years if recency_half_life_years and recency_half_life_years > 0 else 0.0

    # ---------------------------------------------------
    # 2) Per-artist windows (N songs OR M months)
    # ---------------------------------------------------
    artist_windows = {}  # mbid -> {"works_used": list, "window_end_date": datetime|None}
    all_release_dates = []  # for internal checks/diagnostics if needed

    for a in artists:
        a_mbid = a.get("mbid")
        works = a.get("works", []) or []

        # Sort by date (undated go last)
        def work_key(w):
            dt = parse_date(w.get("release_date"))
            return (dt or datetime.max)

        works_sorted = sorted(works, key=work_key)

        if first_n_songs and first_n_songs > 0:
            works_used = works_sorted[:first_n_songs]

        elif first_m_months and first_m_months > 0:
            # anchor on first dated release
            first_dated = None
            for w in works_sorted:
                dt = parse_date(w.get("release_date"))
                if dt:
                    first_dated = dt
                    break
            if first_dated:
                cutoff = add_months(first_dated.date(), first_m_months)
                # include only items with a parsable date <= cutoff
                works_used = [
                    w for w in works_sorted
                    if (parse_date(w.get("release_date")) and parse_date(w.get("release_date")).date() <= cutoff)
                ]
            else:
                works_used = []  # no dated releases -> empty window for months rule

        else:
            works_used = works_sorted  # no truncation

        # window_end_date = date of the last kept work (if any)
        end_dt = None
        if works_used:
            last_dt = parse_date(works_used[-1].get("release_date"))
            end_dt = last_dt
            if last_dt:
                all_release_dates.append(last_dt)

        artist_windows[a_mbid] = {
            "works_used": works_used,
            "window_end_date": end_dt
        }

        # gather all valid dates for diagnostics
        for w in works_used:
            dt = parse_date(w.get("release_date"))
            if dt:
                all_release_dates.append(dt)

    # ----------------------------------------------------
    # 3) Build bipartite table: artist_song_df (credits)
    # ----------------------------------------------------
    song_credit_rows = []
    song_to_credit_people = defaultdict(set)
    song_to_cowrite_people = defaultdict(set)

    person_roles = defaultdict(set)
    person_genres = defaultdict(Counter)
    person_first_date = {}
    person_last_date = {}

    for a in artists:
        a_mbid = a.get("mbid")
        a_name = a.get("artist_name")
        a_roles = to_roles(a.get("roles"))

        works_used = artist_windows[a_mbid]["works_used"]

        for w in works_used:
            song_id = w.get("id")
            song_name = w.get("name")
            release_dt = parse_date(w.get("release_date"))
            genre_tags = [str(g).lower() for g in (w.get("genres") or [])]

            # participants: focal + collaborators
            participants = []
            if a_roles:
                participants.append((a_mbid, a_name, a_roles))
            for c in (w.get("collaborators") or []):
                cm, cn, cr = c.get("mbid"), c.get("name"), to_roles(c.get("roles"))
                participants.append((cm, cn, cr))

            # team-size sets
            for (pm, pn, pr) in participants:
                if any(r in CREDIT_ROLES for r in pr):
                    song_to_credit_people[song_id].add(pm)
                if any(r in COWRITE_ROLES for r in pr):
                    song_to_cowrite_people[song_id].add(pm)

            # expand to one row per CREDIT role
            for (pm, pn, pr) in participants:
                for role in pr:
                    if role in CREDIT_ROLES:
                        song_credit_rows.append({
                            "artist_mbid": pm,
                            "artist_name": pn,
                            "song_id": song_id,
                            "song_name": song_name,
                            "credit_role": role,
                            "release_date": release_dt.date().isoformat() if release_dt else None,
                            "genre_tags": ";".join(sorted(set(genre_tags))) if genre_tags else "",
                            "team_size": None,
                            "team_size_cowrite": None,
                        })
                # accumulate roles/genres per person within window
                person_roles[pm].update(pr)
                for g in genre_tags:
                    person_genres[pm][g] += 1
                if release_dt:
                    if (pm not in person_first_date) or (release_dt < person_first_date[pm]):
                        person_first_date[pm] = release_dt
                    if (pm not in person_last_date) or (release_dt > person_last_date[pm]):
                        person_last_date[pm] = release_dt

    # Fill team sizes
    for row in song_credit_rows:
        sid = row["song_id"]
        row["team_size"] = len(song_to_credit_people[sid]) if sid in song_to_credit_people else 0
        row["team_size_cowrite"] = len(song_to_cowrite_people[sid]) if sid in song_to_cowrite_people else 0

    artist_song_df = pd.DataFrame(song_credit_rows)

    # ----------------------------------------------------
    # 4) Build projection: edges with covariates
    # ----------------------------------------------------
    pair_stats = defaultdict(lambda: {
        "weight_raw": 0,
        "weight_size_adj": 0.0,
        "first_collab_date": None,
        "last_collab_date": None,
        "song_ids": set(),
    })

    song_cowriters = {}
    song_release_dt = {}

    for a in artists:
        a_mbid = a.get("mbid")
        works_used = artist_windows[a_mbid]["works_used"]
        for w in works_used:
            sid = w.get("id")
            rdt = parse_date(w.get("release_date"))
            song_release_dt[sid] = rdt

            roster = []
            a_roles = to_roles(a.get("roles"))
            if any(r in COWRITE_ROLES for r in a_roles):
                roster.append(a_mbid)
            for c in (w.get("collaborators") or []):
                cm, cr = c.get("mbid"), to_roles(c.get("roles"))
                if any(r in COWRITE_ROLES for r in cr):
                    roster.append(cm)

            song_cowriters[sid] = list(sorted(set(roster)))

    for sid, members in song_cowriters.items():
        if len(members) < 2:
            continue
        rdt = song_release_dt.get(sid)
        k = len(members)
        denom = nC2(k)
        adj_inc = (1.0 / denom) if denom > 0 else 0.0
        for u, v in combinations(members, 2):
            a_, b_ = (u, v) if u < v else (v, u)
            S = pair_stats[(a_, b_)]
            S["weight_raw"] += 1
            S["weight_size_adj"] += adj_inc
            S["song_ids"].add(sid)
            if rdt:
                if (S["first_collab_date"] is None) or (rdt < S["first_collab_date"]):
                    S["first_collab_date"] = rdt
                if (S["last_collab_date"] is None) or (rdt > S["last_collab_date"]):
                    S["last_collab_date"] = rdt

    # Node rows (from accumulated per-person stats)
    nodes_rows = []
    all_people = set(list(person_roles.keys())) | set(person_genres.keys())
    for mbid in all_people:
        roles_list = sorted([r for r in person_roles[mbid]]) if mbid in person_roles else []
        genres_list = sorted(list(person_genres[mbid].keys())) if mbid in person_genres else []
        first_dt = person_first_date.get(mbid)
        last_dt = person_last_date.get(mbid)
        time_years = ydiff(first_dt, last_dt) if first_dt and last_dt else 0.0
        nodes_rows.append({
            "mbid": mbid,
            "name": None,  # filled from bipartite (if present)
            "all_roles": ";".join(roles_list),
            "all_genres": ";".join(genres_list),
            "first_release_date_in_window": first_dt.date().isoformat() if first_dt else None,
            "last_release_date_in_window": last_dt.date().isoformat() if last_dt else None,
            "num_songs_in_window": None,               # filled next
            "num_collaborators_in_window": 0,          # filled after edges
            "time_in_network_years": round(time_years, 3),
        })
    nodes_df = pd.DataFrame(nodes_rows)

    # Attach names and num_songs_in_window from bipartite
    if not artist_song_df.empty:
        names_map = artist_song_df.groupby("artist_mbid")["artist_name"].agg(lambda x: next((v for v in x if v), None))
        nodes_df["name"] = nodes_df["mbid"].map(names_map).where(nodes_df["name"].isna(), nodes_df["name"])

        songs_per_artist = artist_song_df.groupby("artist_mbid")["song_id"].nunique()
        nodes_df["num_songs_in_window"] = nodes_df["mbid"].map(songs_per_artist).fillna(0).astype(int)

    # For recency reference per artist (last observed song date in window)
    last_date_by_artist = {
        row["mbid"]: (datetime.strptime(row["last_release_date_in_window"], "%Y-%m-%d") if row["last_release_date_in_window"] else None)
        for _, row in nodes_df.iterrows()
    }

    # Prepare sets for overlaps
    roles_sets = {r["mbid"]: set(r["all_roles"].split(";")) if r["all_roles"] else set()
                  for _, r in nodes_df.iterrows()}
    genres_sets = {r["mbid"]: set(r["all_genres"].split(";")) if r["all_genres"] else set()
                   for _, r in nodes_df.iterrows()}

    # Build edges_df with overlaps and **artist-window** recency
    edge_rows = []
    for (u, v), S in pair_stats.items():
        last_dt = S["last_collab_date"]

        # Reference = earlier of the two artists' last-song dates in their windows
        ref_u = last_date_by_artist.get(u)
        ref_v = last_date_by_artist.get(v)
        ref_edge = None
        if ref_u and ref_v:
            ref_edge = min(ref_u, ref_v)

        if LAMBDA and last_dt and ref_edge and (ref_edge >= last_dt):
            years_since = ydiff(last_dt, ref_edge)
            recency_weight = math.exp(-LAMBDA * years_since)
        else:
            recency_weight = 0.0  # cannot compute or window ends before last collab

        # Jaccard overlaps
        ru, rv = roles_sets.get(u, set()), roles_sets.get(v, set())
        gu, gv = genres_sets.get(u, set()), genres_sets.get(v, set())

        def jaccard(a, b):
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b) if (a or b) else 0.0

        edge_rows.append({
            "u": u,
            "v": v,
            "weight_raw": int(S["weight_raw"]),
            "weight_size_adj": round(S["weight_size_adj"], 6),
            "first_collab_date": S["first_collab_date"].date().isoformat() if S["first_collab_date"] else None,
            "last_collab_date": last_dt.date().isoformat() if last_dt else None,
            "recency_weight": round(recency_weight, 6),
            "roles_overlap": round(jaccard(ru, rv), 6),
            "genre_overlap": round(jaccard(gu, gv), 6),
        })

    edges_df = pd.DataFrame(edge_rows)

    # Degree for nodes
    if not edges_df.empty:
        deg = defaultdict(int)
        for _, r in edges_df.iterrows():
            if r["weight_raw"] > 0:
                deg[r["u"]] += 1
                deg[r["v"]] += 1
        nodes_df["num_collaborators_in_window"] = nodes_df["mbid"].map(deg).fillna(0).astype(int)

    # Tidy dtypes
    for col in ["team_size", "team_size_cowrite"]:
        if col in artist_song_df.columns:
            artist_song_df[col] = artist_song_df[col].astype("Int64")

    return artist_song_df, edges_df, nodes_df


from datetime import datetime, timedelta
from statistics import mean

def avg_songs_in_first_m_months(artists, m):
    """
    Calculate the average number of songs released by artists 
    in their first m months of activity.

    Parameters:
    -----------
    artists : list of dict
        Each dict should have a 'works' key with list of works,
        and each work should include a 'release_date' (YYYY-MM-DD).
    m : int
        Number of months defining the "early career" window.

    Returns:
    --------
    float
        Average number of songs released per artist in their first m months.
    """
    
    def parse_date(d):
        """Parse a date string safely."""
        try:
            return datetime.strptime(d, "%Y-%m-%d")
        except Exception:
            return None

    m_window = timedelta(days=30 * m)
    song_counts = []

    for artist in artists:
        works = artist.get("works", [])
        # Parse and sort by release date
        dates = [parse_date(w.get("release_date")) for w in works if parse_date(w.get("release_date"))]
        if not dates:
            continue  # Skip if no valid dates
        
        debut = min(dates)
        cutoff = debut + m_window
        
        # Count works released within first m months
        n_songs = sum(1 for d in dates if d <= cutoff)
        song_counts.append(n_songs)

    if not song_counts:
        return 0.0
    
    return mean(song_counts)