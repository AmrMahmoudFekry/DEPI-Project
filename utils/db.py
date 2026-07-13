import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_supabase_client() -> Client:
    """
    Single shared Supabase connection for the whole app.
    Every module (history_manager, dashboard_stats, app.py) should
    import this instead of creating its own client, so we don't open
    multiple redundant connections and so credentials live in one place.

    Requires SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml
    (or Streamlit Cloud's Settings -> Secrets).
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)
