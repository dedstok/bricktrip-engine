import os
import csv
import re
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

CSV_FILE = "Brickset-instructions.csv"


def get_document_number(url):
    match = re.search(r"/(\d+)\.pdf", url)
    return match.group(1) if match else url


batch = []
processed = 0
skipped = 0

print("Reading Brickset instruction index...")

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for item in reader:
        set_num = item["SetNumber"].strip()
        url = item["URL"].strip()

        if not set_num or not url:
            skipped += 1
            continue

        row = {
            "set_num": set_num,
            "document_number": get_document_number(url),
            "source": "brickset-lego",
            "source_url": url,
        }

        batch.append(row)

        if len(batch) >= 500:
            supabase.table("instruction_documents").upsert(
                batch,
                on_conflict="set_num,document_number"
            ).execute()

            processed += len(batch)
            print(f"Imported {processed:,} instruction records...")
            batch = []

if batch:
    supabase.table("instruction_documents").upsert(
        batch,
        on_conflict="set_num,document_number"
    ).execute()
    processed += len(batch)

supabase.table("pipeline_runs").insert({
    "job_name": "import_brickset_instructions",
    "status": "success",
    "records_processed": processed,
    "records_created": processed,
}).execute()

print(f"SUCCESS: Imported {processed:,} instruction records.")
print(f"Skipped {skipped:,} incomplete rows.")
