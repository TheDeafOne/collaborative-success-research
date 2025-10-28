#!/usr/bin/env python3
"""
get_artist_by_mbid.py

Given a MusicBrainz Artist ID (MBID), find the corresponding Wikidata item
and return basic info (QID, label, description). Optionally fetch richer
fields (aliases, genres, occupations, website, etc.) from the Wikidata API.

Usage:
    python get_artist_by_mbid.py 5b11f4ce-a62d-471e-81fc-a69a8278c7da
"""

import re
import sys
import json
import requests
from typing import Dict, Any, Optional

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Be a good API citizen: set a descriptive User-Agent for Wikidata services.
HEADERS = {
    "User-Agent": "mbid-to-wikidata/1.0 (contact: your.email@example.com)"
}

# Simple UUID v4-ish pattern (MBIDs are UUIDs). Looser than strict RFC 4122 on purpose.
MBID_RE = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")


def get_wikidata_item_by_mbid(mbid: str, language: str = "en") -> Optional[Dict[str, Any]]:
    """
    Find the Wikidata item for a MusicBrainz artist ID (MBID) using SPARQL.

    Returns a dict like:
        {
            "qid": "QXXXX",
            "label": "...",
            "description": "..."
        }
    or None if not found.
    """
    # Normalize mbid to lowercase with hyphens
    mbid = mbid.lower()
    if not MBID_RE.match(mbid):
        raise ValueError(f"Invalid MBID format: {mbid}")

    # P434 = MusicBrainz artist ID
    query = f"""
    SELECT ?artist ?artistLabel ?artistDescription WHERE {{
      ?artist wdt:P434 "{mbid}" .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language},en".
      }}
    }}
    LIMIT 1
    """

    resp = requests.get(
        SPARQL_ENDPOINT,
        headers=HEADERS,
        params={"query": query, "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    b = bindings[0]
    uri = b["artist"]["value"]  # e.g., "http://www.wikidata.org/entity/QXXX"
    qid = uri.rsplit("/", 1)[-1]
    label = b.get("artistLabel", {}).get("value")
    description = b.get("artistDescription", {}).get("value")

    return {"qid": qid, "label": label, "description": description}


def fetch_richer_details(qid: str, language: str = "en") -> Dict[str, Any]:
    """
    Fetch additional details for a Wikidata item via the Wikidata API (wbgetentities).
    Pulls aliases, sitelinks, and a handful of useful claims mapped to readable values when possible.
    """
    # Get the entity
    resp = requests.get(
        WIKIDATA_API,
        headers=HEADERS,
        params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels|descriptions|aliases|claims|sitelinks",
            "languages": language,
            "format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    entity = resp.json().get("entities", {}).get(qid, {})
    out: Dict[str, Any] = {"qid": qid}

    # Labels / descriptions / aliases
    out["label"] = (entity.get("labels", {}).get(language) or entity.get("labels", {}).get("en") or {}).get("value")
    out["description"] = (entity.get("descriptions", {}).get(language) or entity.get("descriptions", {}).get("en") or {}).get("value")
    out["aliases"] = [a["value"] for a in entity.get("aliases", {}).get(language, [])]

    # Useful sitelinks (e.g., English Wikipedia)
    enwiki = entity.get("sitelinks", {}).get("enwiki", {})
    out["wikipedia_title_en"] = enwiki.get("title")

    # Helper to extract string/URL/time from simple claim values
    def extract_datavalue_snak(snak: Dict[str, Any]) -> Any:
        if not snak or snak.get("snaktype") != "value":
            return None
        dv = snak.get("datavalue", {})
        v = dv.get("value")
        t = dv.get("type")
        if t == "string":
            return v
        if t == "time" and isinstance(v, dict):
            # Return ISO-ish time string
            return v.get("time")
        if t == "wikibase-entityid" and isinstance(v, dict):
            return f"Q{v.get('numeric-id')}"
        if t == "monolingualtext" and isinstance(v, dict):
            return v.get("text")
        if t == "globecoordinate" and isinstance(v, dict):
            return {"lat": v.get("latitude"), "lon": v.get("longitude")}
        if t == "url":
            return v
        return v

    # Claims of interest:
    # P434 MusicBrainz artist ID, P136 genre, P106 occupation, P19 place of birth, P27 country of citizenship,
    # P571 inception (for groups), P569 date of birth (for humans), P570 date of death, P856 official website
    claims = entity.get("claims", {})

    def get_claim_values(pid: str):
        vals = []
        for c in claims.get(pid, []):
            m = extract_datavalue_snak(c.get("mainsnak"))
            if m is not None:
                vals.append(m)
        return vals

    out["musicbrainz_artist_id"] = get_claim_values("P434")
    out["genres"] = get_claim_values("P136")
    out["occupations"] = get_claim_values("P106")
    out["date_of_birth"] = get_claim_values("P569")
    out["date_of_death"] = get_claim_values("P570")
    out["inception"] = get_claim_values("P571")
    out["country_of_citizenship"] = get_claim_values("P27")
    out["official_website"] = get_claim_values("P856")

    return out


def get_artist_from_mbid(mbid: str, language: str = "en", rich: bool = True) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper: find the Wikidata item by MBID, then (optionally) enrich.
    """
    base = get_wikidata_item_by_mbid(mbid, language=language)
    if not base:
        return None
    if not rich:
        return base
    detailed = fetch_richer_details(base["qid"], language=language)
    # ensure base label/description present if API didn’t have them for requested language
    for k in ("label", "description"):
        if not detailed.get(k):
            detailed[k] = base.get(k)
    return detailed


def get_artist_family_musicians(mbid: str, language: str = "en") -> List[Dict[str, Any]]:
    """
    For a MusicBrainz Artist ID (MBID), return family members who are also musical artists.
    'Musical artist' is defined as:
      - has a MusicBrainz artist ID (P434), OR
      - has an occupation (P106) that is a subclass of 'musician' (Q639669).

    Relations covered: spouse (P26), partner (P451), child (P40),
                       father (P22), mother (P25), sibling (P3373).

    Returns a list of dicts like:
      {
        "relation_property": "P26",
        "relation": "spouse",
        "qid": "QXXXX",
        "label": "...",
        "description": "...",
        "occupations": ["singer", "songwriter"],
        "has_mbid": true,
        "relative_mbid": "...."  # if present
      }
    """
    base = get_wikidata_item_by_mbid(mbid, language=language)
    if not base:
        return []

    qid = base["qid"]

    # Relation label mapping for readability
    RELATION_LABELS = {
        "P26": "spouse",
        "P451": "partner",
        "P40": "child",
        "P22": "father",
        "P25": "mother",
        "P3373": "sibling",
    }

    # SPARQL: gather relatives via selected family properties, then keep only those
    # who either have a MusicBrainz artist ID OR whose occupation is (subclass of) musician.
    query = f"""
    SELECT ?relProp ?relative ?relativeLabel ?relativeDescription ?occLabel ?mbidRel WHERE {{
      VALUES ?relProp {{ wdt:P26 wdt:P451 wdt:P40 wdt:P22 wdt:P25 wdt:P3373 }}
      wd:{qid} ?relProp ?relative .

      # Keep only relatives who are clearly musical artists
      {{
        ?relative wdt:P434 ?mbidRel .
      }} UNION {{
        ?relative wdt:P106/wdt:P279* wd:Q639669 .
      }}

      # Gather occupation labels (optional; can be many)
      OPTIONAL {{
        ?relative wdt:P106 ?occ .
        ?occ rdfs:label ?occLabelRaw .
        FILTER(LANGMATCHES(LANG(?occLabelRaw), "{language}") || LANGMATCHES(LANG(?occLabelRaw), "en"))
      }}

      # Descriptions/labels in requested language, fallback to en
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language},en".
        ?relative rdfs:label ?relativeLabel .
        ?relative schema:description ?relativeDescription .
      }}
    }}
    """

    resp = requests.get(
        SPARQL_ENDPOINT,
        headers=HEADERS,
        params={"query": query, "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Aggregate rows by relative
    rels: Dict[str, Dict[str, Any]] = {}
    for b in data.get("results", {}).get("bindings", []):
        rel_uri = b["relative"]["value"]
        rel_qid = rel_uri.rsplit("/", 1)[-1]
        rel_prop_uri = b["relProp"]["value"]  # e.g. http://www.wikidata.org/prop/direct/P26
        rel_prop = rel_prop_uri.rsplit("/", 1)[-1]

        entry = rels.setdefault(rel_qid, {
            "relation_property": rel_prop,
            "relation": RELATION_LABELS.get(rel_prop, rel_prop),
            "qid": rel_qid,
            "label": b.get("relativeLabel", {}).get("value"),
            "description": b.get("relativeDescription", {}).get("value"),
            "occupations": [],
            "has_mbid": False,
            "relative_mbid": None,
        })

        occ = b.get("occLabel", {}).get("value")
        if occ and occ not in entry["occupations"]:
            entry["occupations"].append(occ)

        mbid_rel = b.get("mbidRel", {}).get("value")
        if mbid_rel:
            entry["has_mbid"] = True
            entry["relative_mbid"] = mbid_rel

        # If multiple relation types appear (rare), keep the most "specific" in a simple priority
        # Priority order: spouse/partner > child > sibling > parent
        priority = {"spouse": 5, "partner": 4, "child": 3, "sibling": 2, "father": 1, "mother": 1}
        curr = entry["relation"]
        new_rel = RELATION_LABELS.get(rel_prop, rel_prop)
        if priority.get(new_rel, 0) > priority.get(curr, 0):
            entry["relation"] = new_rel
            entry["relation_property"] = rel_prop

    # Return as a list
    return list(rels.values())

if __name__ == "__main__":
    print(get_artist_family_musicians('c8b03190-306c-4120-bb0b-6f2ebfc06ea9'))
