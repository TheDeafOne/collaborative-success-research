import datetime as dt
import math
from collections import Counter
from statistics import median, pstdev
from typing import Any, Dict, List, Optional, Tuple


# ---------------- generic helpers ----------------

def _parse_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def _least_squares_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def _entropy(counter: Counter[str]) -> Optional[float]:
    total = sum(counter.values())
    if total <= 0:
        return None
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def _pct(num: int, denom: int) -> Optional[float]:
    return num / denom if denom > 0 else None


# ---------------- window + base extraction ----------------

def _compute_debut_and_cutoff(
    works: List[Dict[str, Any]],
    releases: List[Dict[str, Any]],
    years: Optional[int],
) -> Tuple[Optional[dt.date], Optional[dt.date], List[dt.date]]:
    """
    Returns:
        debut_date: earliest activity date (or None if unknown)
        cutoff: debut_date + years if years is not None, otherwise None
        all_dates: list of all valid activity dates (possibly empty)
    """
    release_dates_all = [
        d for d in (_parse_date(r.get("date")) for r in releases) if d is not None
    ]
    work_dates_all = [
        d for d in (_parse_date(w.get("first_release_date")) for w in works) if d is not None
    ]

    all_dates = release_dates_all or work_dates_all
    if not all_dates:
        # No temporal information at all
        return None, None, []

    debut_date = min(all_dates)

    if years is None:
        # No fixed early-career window: aggregate over entire career
        cutoff = None
    else:
        try:
            cutoff = debut_date.replace(year=debut_date.year + years)
        except ValueError:
            # Feb 29 etc., fallback
            cutoff = dt.date(debut_date.year + years, debut_date.month, min(
                debut_date.day,
                dt.date(debut_date.year + years, debut_date.month, 1).replace(day=28).day,
            ))

    return debut_date, cutoff, all_dates

def _filter_early_window(
    works: List[Dict[str, Any]],
    recordings: List[Dict[str, Any]],
    releases: List[Dict[str, Any]],
    cutoff: Optional[dt.date],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, Optional[dt.date]],
]:
    """
    Filter entities to the early-career window if cutoff is provided.
    If cutoff is None (years=None), include all dated activity (no upper bound).
    If no debut/cutoff at all, you'll usually pass cutoff=None but then also
    have debut_date=None, in which case cadence/temporal features become None.
    """
    # releases in window or whole career (depending on cutoff)
    releases_win: List[Dict[str, Any]] = []
    for r in releases:
        d = _parse_date(r.get("date"))
        if d is None:
            continue
        if cutoff is not None and d >= cutoff:
            continue
        releases_win.append(r)

    # map release_id -> date
    rel_date_by_id = {
        r["id"]: _parse_date(r.get("date"))
        for r in releases_win
        if _parse_date(r.get("date")) is not None
    }

    # recordings: keep those linked to a dated release within optional cutoff
    recordings_win: List[Dict[str, Any]] = []
    for rec in recordings:
        rid = rec.get("release_id")
        if rid is None:
            continue
        rd = rel_date_by_id.get(rid)
        if rd is None:
            continue
        recordings_win.append(rec)

    # work_date_by_id: first_release_date, possibly updated using recording release dates
    work_date_by_id: Dict[int, Optional[dt.date]] = {
        w["id"]: _parse_date(w.get("first_release_date")) for w in works
    }

    for rec in recordings:
        w_id = rec.get("work_id")
        if w_id is None:
            continue
        rid = rec.get("release_id")
        if rid is None:
            continue
        rd = rel_date_by_id.get(rid)
        if rd is None:
            continue
        existing = work_date_by_id.get(w_id)
        if existing is None or rd < existing:
            work_date_by_id[w_id] = rd

    works_win = []
    for w in works:
        d = work_date_by_id.get(w["id"])
        if d is None:
            continue
        if cutoff is not None and d >= cutoff:
            continue
        works_win.append(w)

    return releases_win, recordings_win, works_win, work_date_by_id


# ---------------- cadence & timing ----------------

def _compute_cadence_features(
    releases_win: List[Dict[str, Any]],
    debut_date: Optional[dt.date],
    cutoff: Optional[dt.date],
) -> Dict[str, Any]:
    if debut_date is None:
        # No temporal info at all – return null-ish cadence
        return {
            "releases_total": len(releases_win),
            "releases_per_year": None,
            "avg_days_between_releases": None,
            "release_velocity_releases_per_day": None,
            "release_velocity_releases_per_year": None,
            "gap_median_days": None,
            "gap_std_days": None,
            "max_dry_spell_days": None,
            "front_loading_index": None,
        }

    release_dates_win = [
        _parse_date(r.get("date")) for r in releases_win
        if _parse_date(r.get("date")) is not None
    ]
    release_dates_win.sort()

    gaps = [
        (release_dates_win[i] - release_dates_win[i - 1]).days
        for i in range(1, len(release_dates_win))
    ]
    avg_gap_days = (sum(gaps) / len(gaps)) if gaps else None

    # For rates, if cutoff is None, just use today as upper bound
    end_for_rate = cutoff if cutoff is not None else dt.date.today()
    elapsed_days = max(0, (min(end_for_rate, dt.date.today()) - debut_date).days)
    releases_per_year = (
        len(release_dates_win) / (elapsed_days / 365.25) if elapsed_days > 0 else None
    )

    if release_dates_win:
        xs = [(d - debut_date).days for d in release_dates_win]
        ys = list(range(1, len(xs) + 1))
        slope = _least_squares_slope(xs, ys)
        release_velocity_per_day = slope
        release_velocity_releases_per_year = slope * 365.25 if slope is not None else None
    else:
        release_velocity_per_day = None
        release_velocity_releases_per_year = None

    if len(release_dates_win) >= 2:
        gap_median = float(median(gaps))
        gap_std = float(pstdev(gaps)) if len(gaps) > 1 else 0.0
        max_dry = max(gaps)
    else:
        gap_median = None
        gap_std = None
        max_dry = None

    # front-loading index
    if release_dates_win:
        # If cutoff is None, define the "window" as debut -> last release
        end_for_front = cutoff if cutoff is not None else release_dates_win[-1]
        mid = debut_date + (end_for_front - debut_date) / 2
        first_half = sum(1 for d in release_dates_win if d < mid)
        front_loading_index = first_half / len(release_dates_win)
    else:
        front_loading_index = None

    return {
        "releases_total": len(releases_win),
        "releases_per_year": releases_per_year,
        "avg_days_between_releases": avg_gap_days,
        "release_velocity_releases_per_day": release_velocity_per_day,
        "release_velocity_releases_per_year": release_velocity_releases_per_year,
        "gap_median_days": gap_median,
        "gap_std_days": gap_std,
        "max_dry_spell_days": max_dry,
        "front_loading_index": front_loading_index,
    }


# ---------------- collaboration / network ----------------

def _compute_collaboration_features(
    works_win: List[Dict[str, Any]],
    recordings_win: List[Dict[str, Any]],
) -> Dict[str, Any]:
    work_is_collab: Dict[int, bool] = {}
    collaborator_keys: set = set()

    for w in works_win:
        collabs = w.get("collaborators") or []
        work_is_collab[w["id"]] = len(collabs) > 1
        for c in collabs:
            collaborator_keys.add(str(c))

    collab_tracks = 0
    for rec in recordings_win:
        w_id = rec.get("work_id")
        if w_id is None:
            continue
        if work_is_collab.get(w_id, False):
            collab_tracks += 1

    tracks_total = len(recordings_win)
    collab_track_rate = _pct(collab_tracks, tracks_total)
    unique_collaborator_count = len(collaborator_keys)

    return {
        "tracks_total": tracks_total,
        "collab_track_rate": collab_track_rate,
        "unique_collaborator_count": unique_collaborator_count,
    }


# ---------------- labels ----------------
def _compute_label_features(
    releases_win: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label_names_all: List[str] = []
    labels_by_release: List[str] = []

    for r in releases_win:
        names = (r.get("label_names") or [])[:]
        ids = (r.get("label_ids") or [])[:]

        label_names_all.extend(names)

        if names:
            label_for_churn = names[0]
        elif ids:
            label_for_churn = ids[0]
        else:
            label_for_churn = ""  # explicit "unknown" bucket

        labels_by_release.append(label_for_churn)

    # ---------------------
    # Primary label + HHI
    # ---------------------
    label_diversity_count = len({ln for ln in label_names_all if ln})
    lbls_nonempty = [l for l in labels_by_release if l]

    if lbls_nonempty:
        c = Counter(lbls_nonempty)

        primary_label = sorted(
            c.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]

        total_lbl = sum(c.values())
        label_hhi = sum((n / total_lbl) ** 2 for n in c.values())

        churn = sum(
            1 for i in range(1, len(lbls_nonempty))
            if lbls_nonempty[i] != lbls_nonempty[i - 1]
        )
    else:
        primary_label = None
        churn = 0
        label_hhi = None

    # ---------------------
    # NEW: Comma-separated list of ALL label names
    # ---------------------
    # Normalize: lower, strip, dedupe while preserving order
    labels_clean = []
    seen = set()
    for name in label_names_all:
        normalized = name.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            labels_clean.append(normalized)

    all_labels_str = ", ".join(labels_clean) if labels_clean else ""

    return {
        "label_diversity_count": label_diversity_count,
        "label_churn": churn,
        "label_hhi": label_hhi,
        "primary_label": primary_label,
        "all_labels_str": all_labels_str,     # NEW
    }


# ---------------- duration & title patterns ----------------

def _compute_duration_and_title_features(
    recordings_win: List[Dict[str, Any]],
) -> Dict[str, Any]:
    lengths = [
        rec["length_ms"] for rec in recordings_win
        if rec.get("length_ms") is not None
    ]
    titles = [rec.get("name") or "" for rec in recordings_win]

    if lengths:
        s = sorted(lengths)
        mid = len(s) // 2
        if len(s) % 2:
            dur_median = float(s[mid])
        else:
            dur_median = float((s[mid - 1] + s[mid]) / 2)
        duration_ms_mean = sum(s) / len(s)
        duration_ms_median = dur_median
        duration_ms_min = s[0]
        duration_ms_max = s[-1]
    else:
        duration_ms_mean = None
        duration_ms_median = None
        duration_ms_min = None
        duration_ms_max = None

    remix_count = sum(1 for t in titles if "remix" in t.lower())
    acoustic_count = sum(1 for t in titles if "acoustic" in t.lower())

    remix_rate = _pct(remix_count, len(titles))
    acoustic_rate = _pct(acoustic_count, len(titles))

    return {
        "duration_ms_mean": duration_ms_mean,
        "duration_ms_median": duration_ms_median,
        "duration_ms_min": duration_ms_min,
        "duration_ms_max": duration_ms_max,
        "remix_rate": remix_rate,
        "acoustic_rate": acoustic_rate,
    }


# ---------------- genres ----------------
def _compute_genre_features(
    works_win: List[Dict[str, Any]],
) -> Dict[str, Any]:
    genre_counter = Counter()
    genre_list_raw: List[str] = []

    for w in works_win:
        for g in w.get("genres") or []:
            if g:
                genre_counter[g] += 1
                genre_list_raw.append(g)

    if genre_counter:
        primary_genre = sorted(
            genre_counter.items(),
            key=lambda kv: (-kv[1], kv[0])
        )[0][0]
    else:
        primary_genre = None

    # ---------------------
    # NEW: comma-separated list of ALL genres
    # ---------------------
    genres_clean = []
    seen = set()
    for g in genre_list_raw:
        normalized = g.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            genres_clean.append(normalized)

    all_genres_str = ", ".join(genres_clean) if genres_clean else ""

    return {
        "genre_count": len(genre_counter),
        "genre_entropy": _entropy(genre_counter),
        "primary_genre": primary_genre,
        "all_genres_str": all_genres_str,     # NEW
    }



# ---------------- location ----------------

def _compute_location_features(
    artist: Dict[str, Any],
) -> Dict[str, Any]:
    country = artist.get("country")
    region_city = artist.get("region_city")

    return {
        "artist_country": country,
        "artist_region_city": region_city,
    }


# ---------------- temporal controls (era / exposure / recency) ----------------

def _compute_temporal_control_features(
    debut_date: Optional[dt.date],
    all_dates: List[dt.date],
    now: Optional[dt.date] = None,
    lambda_per_month: float = 0.1,
) -> Dict[str, Any]:
    if now is None:
        now = dt.date.today()

    if debut_date is None:
        # No valid dates at all
        return {
            "years_active": None,
            "debut_year": None,
            "debut_decade": None,
            "recency_index": None,
        }

    days_active = max(0, (now - debut_date).days)
    years_active = days_active / 365.25

    debut_year = debut_date.year
    debut_decade = debut_year - (debut_year % 10)

    if all_dates:
        last_date = max(all_dates)
        days_since_last = max(0, (now - last_date).days)
        months_since_last = days_since_last / 30.44
        recency_index = math.exp(-lambda_per_month * months_since_last)
    else:
        recency_index = None

    return {
        "years_active": years_active,
        "debut_year": debut_year,
        "debut_decade": debut_decade,
        "recency_index": recency_index,
    }

def _compute_role_features(artist: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute:
    - primary_role: most frequent normalized role
    - all_roles_str: comma-separated list of unique roles for downstream use
    """
    raw = artist.get("roles") or ""
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]

    if not parts:
        return {
            "primary_role": "unknown",
            "all_roles_str": "",
        }

    c = Counter(parts)
    primary_role = sorted(
        c.items(), key=lambda kv: (-kv[1], kv[0])
    )[0][0]

    # ---------------------
    # NEW: comma-separated list of ALL roles
    # ---------------------
    # Make deterministic: uniqueness + alphabetical order or preserve order
    # We'll preserve input order while deduping:
    roles_clean = []
    seen = set()
    for role in parts:
        if role not in seen:
            seen.add(role)
            roles_clean.append(role)

    all_roles_str = ", ".join(roles_clean)

    return {
        "primary_role": primary_role,
        "all_roles_str": all_roles_str,    # NEW
    }

# ---------------- main entrypoint ----------------

def compute_mb_artist_early_features(
    artist: Dict[str, Any],
    years: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute early-career, artist-level features using ONLY MusicBrainz-style data.

    This function summarizes an artist’s early career output—restricted to their
    first `years` of activity—into a flat feature dict suitable for predictive
    modeling (e.g., forecasting long-run success such as follower counts or
    streaming popularity). It also includes coarse controls for era, exposure
    time, and recency so that models can separate “how long they’ve been around”
    from “how successful they became given their starting conditions.”

    --------------------------------------------------------------------------
    Input schema
    --------------------------------------------------------------------------
    The function expects a single artist record with the following structure:

        artist : dict
            {
                "mbid": str,
                "artist_name": str | None,
                "roles": str,              # comma-separated artist roles (not used directly here)
                "country": str | None,     # e.g. "US" or "United States"
                "region_city": str | None, # e.g. "New York, NY" or "Berlin"

                "works": [
                    {
                        "id": int,
                        "name": str,
                        "first_release_date": "YYYY-MM-DD" | None,
                        "genres": list[str],              # tags for this work
                        "collaborators": list[json],      # arbitrary collaborator entries
                    },
                    ...
                ],

                "recordings": [
                    {
                        "id": int,
                        "name": str,
                        "length_ms": int | None,
                        "work_id": int | None,           # link to works.id
                        "release_id": int | None,        # link to releases.id
                    },
                    ...
                ],

                "releases": [
                    {
                        "id": int,
                        "title": str,
                        "date": "YYYY-MM-DD" | None,     # earliest known date for this release
                        "release_group_id": int | None,
                        "label_ids": list[str],
                        "label_names": list[str],
                    },
                    ...
                ],
            }

    Assumptions:
    - `releases.date` is the earliest known release date for that release.
    - `works.first_release_date` is the first-known appearance of that work.
    - `recordings.release_id` links a recording to a release; that release’s
      date is used to place the recording in time.
    - Dates may be partially missing; only valid ISO dates are used.

    The “early-career window” is defined as:
        [debut_date, debut_date + years)
    where `debut_date` is the earliest date we can infer from releases/works.


    --------------------------------------------------------------------------
    Output schema (feature dictionary)
    --------------------------------------------------------------------------
    Returns
    -------
    features : dict
        A flat dict of artist-level features restricted to the early-career
        window, plus global temporal/era/location controls. All keys are
        present; values may be None where data is insufficient.

    Identification / window definition
    ----------------------------------
    - artist_mbid : str
        MusicBrainz artist ID (stable global identifier). Used for joins and
        de-duplication.

    - artist_name : str | None
        Human-readable artist name for debugging, display, and exploratory work.

    - window_years : int
        The size of the early-career window in years (the `years` parameter).

    - debut_date : "YYYY-MM-DD"
        Earliest known activity date for the artist, defined as the minimum of:
        - all release dates in `releases`
        - all `works.first_release_date` values
        This anchors the start of the career and the early-career window.

    - window_cutoff_date : "YYYY-MM-DD"
        The end date of the early-career window: debut_date + window_years.

    Why these matter:
        These fields define the temporal frame over which “early-career”
        behavior is measured. Downstream models can explicitly condition on
        the window length and align features across artists.


    Era / exposure / recency controls
    ---------------------------------
    These features help control for macro-era differences and simple exposure
    time, so that models don't mistakenly treat older artists as “better” merely
    because they had more time to accumulate listeners.

    - years_active : float
        Approximate years from debut_date to today:
            years_active ≈ (today - debut_date).days / 365.25
        This captures cumulative exposure time—how long the artist has been
        “eligible” to gain listeners or streams.

    - debut_year : int
        Calendar year of the debut (e.g. 1998, 2015).

    - debut_decade : int
        Decade bucket for the debut year (e.g. 1990, 2000, 2010). Computed as:
            debut_decade = debut_year - (debut_year % 10)

    - recency_index : float | None
        A recency weight based on how long it has been since the artist's last
        observed activity (release or work), using an exponential decay:
            recency_index = exp(-λ * months_since_last_release)
        where:
            months_since_last_release ≈ (today - last_activity_date).days / 30.44
            λ is a fixed per-month decay rate (e.g., 0.1).
        If no valid dates exist, this is None.

    Why these matter:
        - years_active: controls for longevity / exposure; an artist active for
          40 years simply has more time to accumulate followers than one active
          for 3 years.
        - debut_year / debut_decade: control for cohort effects and macro-era
          conditions (e.g., pre- vs. post-streaming, platform growth, changes
          in music consumption).
        - recency_index: captures whether the artist is currently active.
          On streaming platforms, recent releases heavily influence follower
          counts and algorithmic exposure; this helps separate “legacy” acts
          from actively promoted ones.


    Cadence & timing (release-based, early window only)
    ---------------------------------------------------
    Using only releases with dates inside the early window:

    - releases_total : int
        Count of releases in the early window.

    - releases_per_year : float | None
        Average number of releases per year within the early window:
            releases_per_year = releases_total / elapsed_years
        where elapsed_years is time from debut_date to either window_cutoff_date
        or today (whichever comes first). None if elapsed time is zero.

    - avg_days_between_releases : float | None
        Mean gap (in days) between consecutive releases in the early window.

    - release_velocity_releases_per_day : float | None
        Slope of cumulative release count vs. days since debut, estimated via
        least squares on (time_since_debut, cumulative_release_index). Higher
        slopes indicate faster early output.

    - release_velocity_releases_per_year : float | None
        Same as above, scaled by 365.25 for interpretability.

    - gap_median_days : float | None
        Median gap between consecutive releases. More robust than the mean to
        outliers (e.g., a single long hiatus).

    - gap_std_days : float | None
        Standard deviation of gaps between releases. Higher values indicate a
        “bursty” release pattern; lower values indicate steadier, more regular
        output.

    - max_dry_spell_days : int | None
        Longest gap in days between releases during the early window.

    - front_loading_index : float | None
        Share of early-window releases that occur in the first half of the
        early window:
            front_loading_index = (# releases in first half) / releases_total
        Values near 1.0 mean strongly front-loaded early careers; values near
        0.5 indicate a more even spread.

    Why these matter:
        Cadence features proxy effort, productivity, and support. Artists who
        release frequently and steadily early on often have more label backing,
        stronger work ethic, or better access to studio resources—all of which
        are predictive of downstream success. Burstiness and dry spells may
        signal instability, experimentation, or label conflicts.


    Collaboration / network features (works & recordings, early window)
    -------------------------------------------------------------------
    Collaboration is inferred from works and recordings inside the early window.

    - tracks_total : int
        Number of recordings in the early window (recordings linked to releases
        in the window).

    - collab_track_rate : float | None
        Fraction of early-window recordings whose underlying work is marked as
        collaborative:
            collab_track_rate = (# collaborative recordings) / tracks_total
        A work is treated as collaborative if `len(collaborators) > 1`.

    - unique_collaborator_count : int
        Number of distinct collaborator entries across early-window works
        (each collaborator entry is converted to a string key and de-duplicated).

    Why these matter:
        Collaboration features approximate early network breadth and depth.
        Artists who work with many collaborators early on may tap into more
        scenes, genres, and audiences, and may benefit from network effects in
        discovery and promotion.


    Label features (early window)
    -----------------------------
    Using label_names / label_ids on early-window releases:

    - label_diversity_count : int
        Number of distinct non-empty label names across early-window releases.
        Higher values indicate more label relationships (or compilations),
        while low values suggest a focused relationship with one main label.

    - label_churn : int
        Number of times the canonical label (first non-empty name or ID on a
        release) changes between consecutive releases, ignoring unknown labels.

    - label_hhi : float | None
        Herfindahl-Hirschman Index (HHI) over the distribution of canonical
        labels:
            label_hhi = sum((n_i / N)^2)
        where n_i is the count of releases on label i and N is the total number
        of labeled releases. Values near 1 imply a single dominant label; lower
        values indicate a more balanced spread.

    - primary_label : str | None
        The most frequently observed canonical label in the early window.
        Ties are broken alphabetically. None if no labels are available.

    Why these matter:
        Labels proxy access to capital, marketing, radio/playlist placement,
        and professional networks. A dominant major label or stable label
        relationship often signals better promotion and infrastructure. High
        label churn can reflect instability, contract issues, or searching for
        a better fit.


    Duration & title-pattern features (early window)
    -----------------------------------------------
    Based on recordings whose releases fall inside the early window:

    - duration_ms_mean : float | None
        Mean track duration in milliseconds.

    - duration_ms_median : float | None
        Median track duration in milliseconds.

    - duration_ms_min : int | None
        Shortest track duration in the early window.

    - duration_ms_max : int | None
        Longest track duration in the early window.

    - remix_rate : float | None
        Fraction of early-window recordings whose title contains "remix"
        (case-insensitive).

    - acoustic_rate : float | None
        Fraction of early-window recordings whose title contains "acoustic"
        (case-insensitive).

    Why these matter:
        Duration features capture stylistic and format choices (e.g., radio-
        friendly 3-minute tracks vs. long-form compositions). Title-pattern
        features such as remix/acoustic rates reflect catalog strategy and
        marketing behavior—e.g., high remix rates can indicate focus on club /
        dance markets or heavy re-packaging to drive streams.


    Genre features (works, early window)
    ------------------------------------
    Derived from `genres` attached to early-window works:

    - genre_count : int
        Number of distinct genre labels used in early-window works.

    - genre_entropy : float | None
        Entropy of the genre distribution:
            genre_entropy = -sum(p_g * log(p_g))
        where p_g is the relative frequency of genre g.

    - primary_genre : str | None
        The genre with the highest count in early-window works; ties broken
        alphabetically. None if no genres are present.

    Why these matter:
        Genres align artists with particular scenes and audience segments.
        Genre diversity (high count, high entropy) suggests experimentation or
        cross-genre positioning, which may affect risk, upside, and the kinds
        of success pathways available. A clear primary genre helps control for
        differences in typical success distributions across genres.


    Location features
    -----------------
    Based solely on top-level artist metadata (no external geocoding):

    - artist_country : str | None
        Raw country code or name from the artist object (e.g., "US").

    - artist_region_city : str | None
        Raw region/city string.

    - location_country_known : bool
        True if `artist_country` is non-empty.

    - location_region_city_known : bool
        True if `artist_region_city` is non-empty.

    Why these matter:
        Geography relates to infrastructure (studios, labels, touring circuits),
        market size, and local scenes. Simple location flags enable models to
        control for systematic differences between, for example, artists from
        major music hubs vs. smaller markets, without over-committing to any
        particular geo ontology.


    Modeling intent
    ---------------
    Together, these features aim to capture:

    - **Effort and productivity**: release cadence, gaps, and velocity.
    - **Network and support**: collaboration breadth and label structure.
    - **Stylistic and packaging choices**: durations, remixes, acoustic versions.
    - **Scene and audience context**: genres and location.
    - **Exposure controls**: years_active, debut cohort, recency.

    When used in predictive models (e.g., regression or tree-based methods)
    for downstream success outcomes (followers, streams, chart performance),
    they help disentangle:
        - “How long has this artist been around?” (exposure)
        - “When and where did they debut?” (macro-era / market conditions)
        - “What did they do with their early years?” (behavioral choices)
        - “Under what structural conditions?” (labels, genres, geography)

    This allows the model to better isolate the contribution of early-career
    patterns from confounding factors like longevity or debut era.
    """
    works = artist.get("works") or []
    recordings = artist.get("recordings") or []
    releases = artist.get("releases") or []

    # 1) debut + window (+ all_dates for temporal controls)
    debut_date, cutoff, all_dates = _compute_debut_and_cutoff(works, releases, years)

    # 2) filter entities to window (if cutoff is None, this means "no upper bound")
    releases_win, recordings_win, works_win, _ = _filter_early_window(
        works, recordings, releases, cutoff
    )

    # 3) section-wise feature blocks
    cadence_feats = _compute_cadence_features(releases_win, debut_date, cutoff)
    collab_feats = _compute_collaboration_features(works_win, recordings_win)
    label_feats = _compute_label_features(releases_win)
    duration_title_feats = _compute_duration_and_title_features(recordings_win)
    genre_feats = _compute_genre_features(works_win)
    location_feats = _compute_location_features(artist)
    temporal_control_feats = _compute_temporal_control_features(debut_date, all_dates)
    role_feats = _compute_role_features(artist)

    # 4) assemble final feature dict
    features: Dict[str, Any] = {
        "artist_mbid": artist.get("mbid"),
        "artist_name": artist.get("artist_name"),
        "window_years": years,  # None means "full-career aggregation"
        "debut_date": debut_date.isoformat() if debut_date is not None else None,
        "window_cutoff_date": cutoff.isoformat() if cutoff is not None else None,
    }

    features.update(cadence_feats)
    features.update(collab_feats)
    features.update(label_feats)
    features.update(duration_title_feats)
    features.update(genre_feats)
    features.update(location_feats)
    features.update(temporal_control_feats)
    features.update(role_feats)

    return features