import os
import requests
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]
REBRICKABLE_API_KEY = os.environ["REBRICKABLE_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

headers = {
    "Authorization": f"key {REBRICKABLE_API_KEY}"
}

url = "https://rebrickable.com/api/v3/lego/sets/"
params = {
    "page_size": 100,
    "ordering": "-year"
}

response = requests.get(url, headers=headers, params=params)
response.raise_for_status()

results = response.json()["results"]

created = 0

for item in results:
    row = {
        "set_num": item["set_num"],
        "lego_number": item["set_num"].split("-")[0],
        "name": item["name"],
        "year": item["year"],
        "piece_count": item.get("num_parts"),
        "image_url": item.get("set_img_url"),
        "source": "rebrickable",
    }

    supabase.table("sets").upsert(
        row,
        on_conflict="set_num"
    ).execute()

    created += 1

supabase.table("pipeline_runs").insert({
    "job_name": "sync_rebrickable",
    "status": "success",
    "records_processed": len(results),
    "records_created": created
}).execute()

print(f"Imported {created} Rebrickable sets into BrickTrip.")
