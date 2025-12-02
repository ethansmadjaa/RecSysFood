import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_key = os.getenv("VITE_SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY must be set")

supabase = create_client(supabase_url, supabase_key)
