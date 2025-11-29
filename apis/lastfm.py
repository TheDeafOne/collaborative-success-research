import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LASTFM_API_KEY")  # Replace with your Last.fm API key


def get_artist_stats(mbid: str):
    """
    Fetch artist info (listeners & playcount) from Last.fm using their MusicBrainz ID (MBID).
    Docs: https://www.last.fm/api/show/artist.getInfo
    """
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getInfo",
        "api_key": API_KEY,
        "mbid": mbid,
        "format": "json",
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise Exception(f"Last.fm API error {data['error']}: {data.get('message', '')}")

    artist = data.get("artist", {})
    stats = artist.get("stats", {})

    return {
        "artist_name": artist.get("name"),
        "listeners": int(stats.get("listeners", 0)),
        "playcount": int(stats.get("playcount", 0)),
    }
