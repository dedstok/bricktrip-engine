import os
import csv
import gzip
import io
import requests
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

# Rebrickable's official bulk Sets download
SETS_URL = "https://cdn.rebrickable.com/media/downloads/sets.csv.gz"

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

print("Downloading Rebrickable bulk sets file...")

response = requests.get(SETS_URL, timeout=120)
response.raise_for_status()

print(f"Downloaded {len(response.content):,} bytes.")

# Decompress the .gz file in memory
with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
    text_stream = io.TextIOWrapper(gz, encoding="utf-8-sig")
    reader = csv.DictReader(text_stream)

    batch = []
    processed = 0

    for item in reader:
        row = {
            "set_num": item["set_num"],
            "lego_number": item["set_num"].split("-")[0],
            "name": item["name"],
            "year": int(item["year"]) if item["year"] else None,
            "piece_count": int(item["num_parts"]) if item["num_parts"] else None,
            "image_url": item.get("img_url") or None,
            "source": "rebrickable",
        }

        batch.append(row)

        # Send records to Supabase in manageable chunks
        if len(batch) >= 500:
            supabase.table("sets").upsert(
                batch,
                on_conflict="set_num"
            ).execute()

            processed += len(batch)
            print(f"Imported {processed:,} sets...")
            batch = []

    # Import whatever remains after the last full batch
    if batch:
        supabase.table("sets").upsert(
            batch,
            on_conflict="set_num"
        ).execute()
        processed += len(batch)

# Record the completed job
supabase.table("pipeline_runs").insert({
    "job_name": "sync_rebrickable_bulk_sets",
    "status": "success",
    "records_processed": processed,
    "records_created": processed
}).execute()

print(f"SUCCESS: BrickTrip now contains {processed:,} Rebrickable sets.")
