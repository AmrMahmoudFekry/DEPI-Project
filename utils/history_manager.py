import pandas as pd
from datetime import datetime
import streamlit as st
from supabase import create_client, Client

# تهيئة الاتصال بـ Supabase باستخدام السيكريتس
# استخدمنا @st.cache_resource عشان ما يعملش اتصال جديد مع كل ريفريش للصفحة
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
TABLE_NAME = "prediction_history" 

def save_prediction(data):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }
    
    try:
        supabase.table(TABLE_NAME).insert(row).execute()
    except Exception as e:
        st.error(f"حدث خطأ أثناء حفظ التنبؤ في قاعدة البيانات: {e}")

def load_history():
    try:
        response = supabase.table(TABLE_NAME).select("*").execute()
        
        # لو في بيانات هيرجعها كـ DataFrame، لو مفيش هيرجع DataFrame فاضي
        if response.data:
            return pd.DataFrame(response.data)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل سجل التنبؤات: {e}")
        return pd.DataFrame()
