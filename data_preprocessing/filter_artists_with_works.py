import json
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import os

import data_preprocessing.features_calculator as fc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta  

load_dotenv()


DATA_DIR = Path('./data')
ARTIST_FULL_DATA = DATA_DIR / "artist_full_data"
OUTPUT_DIR = DATA_DIR / "artist_filtered_data" 

EXCLUDE_LIST = {'[unknown]'}
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_file(file_path: Path):
    """Top-level function, importable by worker processes."""
    failures = 0
    processed = 0
    raw_conversions = []

    out_path = OUTPUT_DIR / file_path.name

    try:
        with file_path.open() as f_in, out_path.open('w') as f_out:
            for raw_artist_work_data in f_in:
                try:
                    if raw_artist_work_data == '\n':
                        continue

                    processed += 1
                    artist_work_data = json.loads(raw_artist_work_data)

                    if artist_work_data["artist_name"] in EXCLUDE_LIST:
                        continue

                    if not artist_work_data['works']:
                        continue

                    work_dates = [
                        datetime.fromisoformat(w['first_release_date']).date()
                        for w in artist_work_data['works']
                        if w.get('first_release_date')
                    ]
                    work_dates.sort()

                    # If we can't date any works, we can't define a career window
                    if not work_dates:
                        continue

                    # Check gaps between first->second and second->third (if available)
                    if len(work_dates) >= 3:
                        # Take the first two consecutive pairs: (0,1) and (1,2)
                        pairs = list(zip(work_dates, work_dates[1:]))[:2]
                        too_large_gap = any(
                            (b - a).days > 365 * 20
                            for a, b in pairs
                        )
                        if too_large_gap:
                            continue

                    career_start = work_dates[0]
                    career_end = career_start + timedelta(days=5 * 365)

                    has_collaborated = False
                    for work in artist_work_data['works']:
                        frd = work.get('first_release_date')
                        if not frd:
                            continue

                        work_date = datetime.fromisoformat(frd).date()
                        # Only consider collaborations in the first 5 years
                        if work_date > career_end:
                            continue

                        filtered_collaborators = [
                            collab
                            for collab in work['collaborators']
                            if artist_work_data['mbid'] != collab['mbid']
                        ]
                        if filtered_collaborators:
                            has_collaborated = True
                            break

                    if not has_collaborated:
                        continue

                    calculated_metrics = fc.compute_mb_artist_early_features(
                        artist_work_data, years=5
                    )
                    raw_conversions.append(calculated_metrics)
                    f_out.write(raw_artist_work_data)

                except Exception:
                    failures += 1
    except Exception:
        failures += 1
    
    return {
        "file": str(file_path),
        "processed": processed,
        "failures": failures,
        "raw_conversions": raw_conversions,
    }

def main():
    files = [f for f in ARTIST_FULL_DATA.iterdir() if f.suffix == ".jsonl"]

    total_processed = 0
    total_failures = 0
    all_raw_conversions = []

    max_workers = None  # or int

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Files", unit="file"):
            result = future.result()
            total_processed += result["processed"]
            total_failures += result["failures"]
            all_raw_conversions.extend(result["raw_conversions"])

    print("Finished")
    print("Total processed:", total_processed)
    print("Total failures:", total_failures)

    pd.DataFrame(all_raw_conversions).to_csv(DATA_DIR / 'artist_base_features_5_years_only_collab.csv', index=False)


if __name__ == "__main__":
    main()