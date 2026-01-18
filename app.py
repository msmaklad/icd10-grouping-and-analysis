import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time
import os
import re
from docx import Document
from io import BytesIO

# ==========================================
# CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="ICD-10 Hospital Intelligence", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Intelligence: ICD-10 Trend Analyzer")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API KEYS
    api_keys_input = st.text_area("Enter Google Gemini API Keys (One per line)", height=150)
    api_keys = [k.strip() for k in api_keys_input.split('\n') if k.strip()]
    
    st.divider()
    
    # 2. STANDARDIZED DEPARTMENT SELECTION
    st.subheader("📋 Context Metadata")
    
    dept_options = [
        "ICU", 
        "Emergency Room ", 
        "Inpatient Ward", 
        "PICU", 
        "NICU", 
        "CCU", 
    ]
    dept_name = st.selectbox("Department Name", options=dept_options)
    
    # 3. STANDARDIZED DATE SELECTION
    st.write("Data Period:")
    col_m, col_y = st.columns(2)
    
    with col_m:
        sel_month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col_y:
        sel_year = st.selectbox("Year", [str(y) for y in range(2024, 2030)])
        
    data_month = f"{sel_month} {sel_year}"
    
    # 4. HISTORY UPLOAD
    st.divider()
    uploaded_history = st.file_uploader("📂 Upload Previous Master History (Optional)", type=['csv'])

    # 5. QUOTA TRACKER
    st.divider()
    if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0
    quota_placeholder = st.empty()
    quota_placeholder.metric("API Calls This Session", st.session_state['api_calls'])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_json_from_text(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0: return None
        return json.loads(text[start:end])
    except: return None

def get_icd_mapping_optimized(keys_list, unique_diagnoses):
    mapping_dict = {}
    
    # MAXIMIZE BATCH SIZE to save Quota
    # Safe limit ~400 lines per call for Gemini
    batch_size = 400 
    
    current_key_idx = 0
    models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash']
    current_model_idx = 0
    
    # Init first key/model
    genai.configure(api_key=keys_list[current_key_idx])
    model = genai.GenerativeModel(models_to_try[current_model_idx])
    
    # Progress Bar
    total_batches = (len(unique_diagnoses) + batch_size - 1) // batch_size
    pbar = st.progress(0, text="AI Analysis Starting...")
    
    batches = [unique_diagnoses[i:i + batch_size] for i in range(0, len(unique_diagnoses), batch_size)]
    
    for i, batch in enumerate(batches):
        success = False
        # Retry loop (Rotate Keys -> Rotate Models)
        max_attempts = len(keys_list) * 2 
        
        for attempt in range(
