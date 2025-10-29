import psycopg

with psycopg.connect("dbname=musicbrainz_db user=musicbrainz password=musicbrainz host=localhost") as conn:
    with conn.cursor() as cur, open("artist_mbids.txt", "w") as f:
        cur.execute("SELECT gid FROM artist;")
        for (gid,) in cur:
            f.write(str(gid) + "\n")
