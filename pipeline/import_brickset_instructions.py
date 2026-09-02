import os
import csv
import re
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

CSV_FILE = "Brickset-instructions.csv"

processed = 0
skipped = 0


def get_document_number(url):
    match = re.search(r"/(\d+)\.pdf", url)
    return match.group(1) if match else url


def import_batch(batch):
    global processed, skipped

    if not batch:
        return

    set_numbers = list({row["set_num"] for row in batch})

    existing_sets = set()

    # Check smaller chunks so the database request does not get too large.
    for i in range(0, len(set_numbers), 100):
        chunk = set_numbers[i:i + 100]

        response = (
            supabase
            .table("sets")
            .select("set_num")
            .in_("set_num", chunk)
            .execute()
        )

        existing_sets.update(row["set_num"] for row in response.data)

    valid_rows = [
        row for row in batch
        if row["set_num"] in existing_sets
    ]

    missing_rows = len(batch) - len(valid_rows)
    skipped += missing_rows

    if valid_rows:
        (
            supabase
            .table("instruction_documents")
            .upsert(
                valid_rows,
                on_conflict="set_num,document_number"
            )
            .execute()
        )

        processed += len(valid_rows)

    print(
        f"Imported {processed:,} instructions "
        f"| skipped {skipped:,} unmatched records"
    )


print("Reading Brickset instruction index...")

batch = []

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for item in reader:
        set_num = item["SetNumber"].strip()
        url = item["URL"].strip()

        if not set_num or not url:
            skipped += 1
            continue

                batch.append({
            "set_num": set_num,
            "document_number": get_document_number(url),
            "source": "brickset-lego",
            "source_url": url,
            "description": item["Description"].strip() or None,
        })

        if len(batch) >= 500:
            import_batch(batch)
            batch = []

if batch:
    import_batch(batch)

supabase.table("pipeline_runs").insert({
    "job_name": "import_brickset_instructions",
    "status": "success",
    "records_processed": processed,
    "records_created": processed,
}).execute()

print("")
print("SUCCESS")
print(f"Imported: {processed:,}")
print(f"Skipped because set was not in BrickTrip: {skipped:,}")
