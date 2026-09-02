import os
import csv
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

CSV_FILE = "bricklink_redesigned_sets.csv"

processed = 0
skipped = 0

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        set_num = row["set_num"].strip()

        # Only import sets that exist in BrickTrip's set catalog.
        existing = (
            supabase
            .table("sets")
            .select("set_num")
            .eq("set_num", set_num)
            .execute()
        )

        if not existing.data:
            skipped += 1
            print(f"Skipped {set_num}: not in BrickTrip catalog")
            continue

        (
            supabase
            .table("revision_benchmarks")
            .upsert({
                "set_num": set_num,
                "source": row["source"].strip(),
                "source_url": row["source_url"].strip(),
                "source_label": row["source_label"].strip(),
            }, on_conflict="set_num,source")
            .execute()
        )

        processed += 1

print("")
print("SUCCESS")
print(f"Imported: {processed}")
print(f"Skipped: {skipped}")
