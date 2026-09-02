import os
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

data = {
    "job_name": "github_connection_test",
    "status": "success",
    "records_processed": 0,
    "records_created": 0,
}

response = supabase.table("pipeline_runs").insert(data).execute()

print("BrickTrip connected to Supabase successfully.")
print(response.data)
