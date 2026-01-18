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
    api_keys_input = st.text_area("Enter Google Gemini API Keys (One per line)", height=150)
    api_keys = [k.strip() for k in api_keys_input.split('\n') if k.strip()]
    
    st.divider()
    dept_name = st.text_input("Department Name", placeholder="e.g., ICU")
    data_month = st.text_input("Data Month/Year", placeholder="e.g., Jan 2026")
    uploaded_history = st.file_uploader("📂 Upload Previous Master History (Optional)", type=['csv'])

    st.divider()
    # QUOTA TRACKER
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
    # Gemini can handle ~2000 tokens easily. 400 lines is safe.
    batch_size = 400 
    
    current_key_idx = 0
    # Fallback logic: Try 2.5 -> then 1.5
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

if uploaded_file and api_keys and dept_name and data_month:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head(3))
        
        st.divider()
        st.subheader("2. Map Columns")
        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        col_id = c1.selectbox("Patient ID", cols, index=0)
        col_diag = c2.selectbox("Diagnosis", cols, index=min(1, len(cols)-1))
        col_rev = c3.selectbox("Revenue", cols, index=min(2, len(cols)-1))
        col_adm = c1.selectbox("Admission Date", cols, index=min(3, len(cols)-1))
        col_disch = c2.selectbox("Discharge Date", cols, index=min(4, len(cols)-1))
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.status("Processing...", expanded=True):
                # Prep
                wdf = df.rename(columns={col_id:'ID', col_diag:'DIAG', col_rev:'REV', col_adm:'ADM', col_disch:'DISCH'})
                wdf['ADM'] = pd.to_datetime(wdf['ADM'], errors='coerce')
                wdf['DISCH'] = pd.to_datetime(wdf['DISCH'], errors='coerce')
                wdf['LOS'] = (wdf['DISCH'] - wdf['ADM']).dt.days.fillna(0).clip(lower=0)
                wdf['REV'] = pd.to_numeric(wdf['REV'], errors='coerce').fillna(0)
                
                # AI Map
                st.write("🧠 Mapping...")
                unique = wdf['DIAG'].unique().tolist()
                mapping = get_icd_mapping_optimized(api_keys, unique)
                
                if not mapping:
                    st.error("AI Mapping Failed.")
                    st.stop()

                wdf['Level3'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level3", "Unmapped"))
                wdf['Level2'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level2", "Unmapped"))
                
                # Stats
                stats = wdf.groupby('Level3').agg({'DIAG':'count', 'LOS':['mean','median'], 'REV':['mean','median']}).reset_index()
                stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
                stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
                
                # History & Report
                st.write("📝 Reporting...")
                old_hist = pd.read_csv(uploaded_history) if uploaded_history else pd.DataFrame()
                full_hist = update_master_history(stats, dept_name, data_month, old_hist)
                report_doc = generate_report(full_hist, data_month, dept_name, api_keys)
                
            st.success("Done!")
            
            # Downloads
            t1, t2 = st.tabs(["📄 Results", "📜 History"])
            with t1:
                if report_doc:
                    st.download_button("📥 Word Report", report_doc, f"Report_{dept_name}.docx")
                st.download_button("📥 Audit Excel", BytesIO(wdf.to_excel(index=False)), f"Audit_{dept_name}.xlsx")
            with t2:
                st.dataframe(full_hist)
                st.download_button("📥 Master History (Save This!)", full_hist.to_csv(index=False).encode(), "master_history.csv")

    except Exception as e:
        st.error(f"Error: {e}")
