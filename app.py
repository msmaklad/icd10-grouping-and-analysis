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
        "Emergency Room (ER)", 
        "Inpatient Ward", 
        "Outpatient", 
        "PICU", 
        "NICU", 
        "CCU"
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
        
        for attempt in range(max_attempts):
            prompt = f"""
            Act as a Medical Coder. Map these diagnosis strings to ICD-10.
            Return a JSON object: key = original text, value = {{"Level2": "Block Name", "Level3": "Category Name"}}
            NO markdown. JUST JSON.
            List: {json.dumps(batch)}
            """
            try:
                response = model.generate_content(prompt)
                
                # Update Tracker
                st.session_state['api_calls'] += 1
                quota_placeholder.metric("API Calls This Session", st.session_state['api_calls'])
                
                data = extract_json_from_text(response.text)
                if data: 
                    mapping_dict.update(data)
                    success = True
                    break # Done with this batch
                
            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    # 1. Try Next Key
                    if len(keys_list) > 1:
                        current_key_idx = (current_key_idx + 1) % len(keys_list)
                        st.toast(f"Quota hit. Rotating Key...", icon="🔄")
                        genai.configure(api_key=keys_list[current_key_idx])
                        model = genai.GenerativeModel(models_to_try[current_model_idx])
                        time.sleep(1)
                        continue
                    
                    # 2. If Keys exhausted (or only 1 key), Try Next Model
                    if current_model_idx < len(models_to_try) - 1:
                        current_model_idx += 1
                        st.warning(f"Quota hit on 2.5 Flash. Downgrading to {models_to_try[current_model_idx]}...")
                        model = genai.GenerativeModel(models_to_try[current_model_idx])
                        time.sleep(1)
                        continue
                        
                    st.error("❌ CRITICAL: Daily Quota Exceeded on all keys and models.")
                    st.stop()
                else:
                    # Non-quota error? Wait and retry once
                    time.sleep(2)
        
        if success:
            pbar.progress((i + 1) / total_batches, text=f"Processed batch {i+1}/{total_batches}")
            time.sleep(2) # Safety pause
            
    pbar.empty()
    return mapping_dict

def update_master_history(current_stats, dept, month, old_hist=None):
    current_stats['Department'] = dept
    current_stats['Month'] = month
    if old_hist is not None and not old_hist.empty:
        old_hist = old_hist[~((old_hist['Department'] == dept) & (old_hist['Month'] == month))]
        return pd.concat([old_hist, current_stats], ignore_index=True)
    return current_stats

def generate_report(history_df, month, dept, keys):
    genai.configure(api_key=keys[0])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    dept_hist = history_df[history_df['Department'] == dept]
    if dept_hist.empty: return None
    
    prompt = f"""
    Analyze this hospital trend data for Dept: {dept}, Month: {month}.
    CSV Data:
    {dept_hist.to_csv(index=False)}
    
    Write a brief Strategic Report (Executive Summary, Trends, Outliers, Recommendations).
    """
    try:
        response = model.generate_content(prompt)
        doc = Document()
        doc.add_heading(f'Strategic Report: {dept} - {month}', 0)
        doc.add_paragraph(response.text)
        b = BytesIO()
        doc.save(b)
        b.seek(0)
        return b
    except: return None

# ==========================================
# MAIN APP
# ==========================================
st.subheader("1. Upload Data")
uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file and api_keys:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file
