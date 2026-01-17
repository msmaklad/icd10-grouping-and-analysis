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
# SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔑 API Keys (Rotation)")
    # FIXED: Removed 'type="password"' because st.text_area does not support it.
    api_keys_input = st.text_area(
        "Enter Google Gemini API Keys (One per line)", 
        height=150,
        help="Paste multiple keys here. The app will rotate through them if one hits the quota limit."
    )
    
    # Process keys into a clean list
    api_keys = [k.strip() for k in api_keys_input.split('\n') if k.strip()]
    
    st.divider()
    st.subheader("📋 Context Metadata")
    dept_name = st.text_input("Department Name", placeholder="e.g., ICU")
    data_month = st.text_input("Data Month/Year", placeholder="e.g., Jan 2026")
    
    st.divider()
    st.info("☁️ **Cloud Mode Tip:** Upload your previous 'Master History' file below to maintain trend continuity.")
    uploaded_history = st.file_uploader("📂 Upload Previous Master History (Optional)", type=['csv'])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_json_from_text(text):
    """Robustly extracts JSON object from mixed text."""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        return None

def get_icd_mapping_with_rotation(keys_list, unique_diagnoses):
    """
    Advanced AI processing with:
    1. Batch Size = 150 (Efficiency)
    2. Auto Key Rotation (Reliability)
    """
    mapping_dict = {}
    
    # --- CONFIGURATION ---
    batch_size = 150  # Process 150 items per call to maximize quota usage
    current_key_idx = 0
    
    # Initialize first key
    genai.configure(api_key=keys_list[current_key_idx])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    total_batches = (len(unique_diagnoses) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="AI is analyzing diagnoses...")
    
    batches = [unique_diagnoses[i:i + batch_size] for i in range(0, len(unique_diagnoses), batch_size)]
    
    for i, batch in enumerate(batches):
        
        # Retry loop for key rotation
        max_retries = len(keys_list) + 1 # Try all keys at least once
        success = False
        
        for attempt in range(max_retries):
            prompt = f"""
            Act as a Medical Coder. Map these diagnosis strings to ICD-10.
            Return a JSON object where keys are the original text, and values contain "Level2" (Block) and "Level3" (Category).
            NO markdown formatting. NO intro text. JUST JSON.
            
            List:
            {json.dumps(batch)}
            """
            
            try:
                response = model.generate_content(prompt)
                batch_data = extract_json_from_text(response.text)
                
                if batch_data:
                    mapping_dict.update(batch_data)
                else:
                    st.warning(f"Batch {i+1} returned invalid JSON. AI Output: {response.text[:50]}...")
                
                success = True
                time.sleep(2) # Short pause for politeness
                break # Break retry loop, move to next batch
                
            except Exception as e:
                error_msg = str(e)
                # Check for Quota Limit (429)
                if "429" in error_msg or "Quota" in error_msg:
                    # ROTATE KEY
                    old_key = current_key_idx
                    current_key_idx = (current_key_idx + 1) % len(keys_list)
                    
                    # Prevent infinite loop if ALL keys are dead
                    if current_key_idx == 0 and attempt > 0:
                        st.error("❌ CRITICAL: All provided API keys have hit their quota limits. Please add more keys or wait.")
                        st.stop()
                        
                    st.toast(f"Quota hit on Key #{old_key+1}. Switching to Key #{current_key_idx+1}...", icon="🔄")
                    
                    # Re-configure with new key
                    genai.configure(api_key=keys_list[current_key_idx])
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    time.sleep(1)
                    continue # Retry the same batch with new key
                else:
                    st.error(f"API Error on batch {i+1}: {e}")
                    break # Fatal error, stop trying this batch

        progress_bar.progress((i + 1) / total_batches, text=f"Mapping batch {i+1}/{total_batches}")
            
    progress_bar.empty()
    return mapping_dict

def update_master_history(current_stats_df, dept, month, existing_history_df=None):
    current_stats_df['Department'] = dept
    current_stats_df['Month'] = month
    
    if existing_history_df is not None and not existing_history_df.empty:
        history_df = existing_history_df[~((existing_history_df['Department'] == dept) & (existing_history_df['Month'] == month))]
        combined_df = pd.concat([history_df, current_stats_df], ignore_index=True)
    else:
        combined_df = current_stats_df
    return combined_df

def generate_strategic_report(history_df, current_month, dept, keys_list):
    """Generates report using a working key."""
    # Just use the first key, if it fails, the user can re-run since mapping is done
    genai.configure(api_key=keys_list[0]) 
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    dept_history = history_df[history_df['Department'] == dept].copy()
    if dept_history.empty:
        return None, "No history found for this department."
        
    csv_data = dept_history.to_csv(index=False)
    
    prompt = f"""
    You are a Strategic Healthcare Executive. 
    Analyze this HISTORICAL dataset for the '{dept}' department.
    Most recent data: {current_month}.
    
    Dataset (CSV):
    {csv_data}
    
    Write a Strategic Insight Report covering:
    1. Executive Summary ({current_month})
    2. Trend Analysis (Compare {current_month} vs previous months)
    3. Operational Deep Dive (Outliers/Skewed LOS)
    4. Recommendations
    """
    try:
        response = model.generate_content(prompt)
        doc = Document()
        doc.add_heading(f'Strategic Trend Report: {dept}', 0)
        doc.add_heading(f'Period: Ending {current_month}', 1)
        doc.add_paragraph(response.text)
        
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer, None
    except Exception as e:
        return None, str(e)

# ==========================================
# MAIN APP LOGIC
# ==========================================

# 1. FILE UPLOAD
st.subheader("1. Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"File loaded! {len(df)} rows found.")
        st.dataframe(df.head(3))
    except Exception as e:
        st.error(f"Error reading file: {e}")

if df is not None:
    st.divider()
    if not api_keys:
        st.warning("👈 Please enter at least one API Key in the sidebar.")
    elif not dept_name or not data_month:
        st.warning("👈 Please enter Department and Month in the sidebar.")
    else:
        # 2. MAPPING INTERFACE
        st.subheader("2. Map Columns")
        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        col_id = c1.selectbox("Patient ID", cols, index=0)
        col_diag = c2.selectbox("Diagnosis", cols, index=min(1, len(cols)-1))
        col_rev = c3.selectbox("Revenue", cols, index=min(2, len(cols)-1))
        col_adm = c1.selectbox("Admission Date", cols, index=min(3, len(cols)-1))
        col_disch = c2.selectbox("Discharge Date", cols, index=min(4, len(cols)-1))

        # 3. RUN BUTTON
        st.divider()
        if st.button("🚀 Run Analysis (Gemini 2.5 Only)", type="primary", use_container_width=True):
            
            with st.status("Processing Data...", expanded=True) as status:
                
                # A. Clean Data
                st.write("🔧 Cleaning data...")
                wdf = df.rename(columns={col_id:'ID', col_diag:'DIAG', col_rev:'REV', col_adm:'ADM', col_disch:'DISCH'})
                wdf['ADM'] = pd.to_datetime(wdf['ADM'], errors='coerce')
                wdf['DISCH'] = pd.to_datetime(wdf['DISCH'], errors='coerce')
                wdf['LOS'] = (wdf['DISCH'] - wdf['ADM']).dt.days.fillna(0).clip(lower=0)
                wdf['REV'] = pd.to_numeric(wdf['REV'], errors='coerce').fillna(0)
                
                # B. AI Mapping (WITH ROTATION)
                st.write(f"🧠 AI Mapping Diagnoses (Using {len(api_keys)} API Keys)...")
                unique = wdf['DIAG'].unique().tolist()
                
                # CALL THE NEW ROTATION FUNCTION
                mapping = get_icd_mapping_with_rotation(api_keys, unique)
                
                if not mapping:
                    st.error("❌ AI Mapping failed completely.")
                    st.stop()
                    
                wdf['Level3'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level3", "Unmapped"))
                wdf['Level2'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level2", "Unmapped"))
                
                # C. Stats
                st.write("📊 Calculating Statistics...")
                stats = wdf.groupby('Level3').agg({'DIAG':'count', 'LOS':['mean','median'], 'REV':['mean','median']}).reset_index()
                stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
                stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
                
                # D. History
                st.write("📜 Updating History...")
                old_history_df = pd.read_csv(uploaded_history) if uploaded_history else pd.DataFrame()
                full_history = update_master_history(stats, dept_name, data_month, old_history_df)
                
                # E. Report
                st.write("📝 Generating Word Report...")
                report_doc, err = generate_strategic_report(full_history, data_month, dept_name, api_keys)
                
                if err:
                    st.error(f"Report Generation Failed: {err}")
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- DOWNLOADS ---
            st.success("✅ Done!")
            
            t1, t2, t3 = st.tabs(["📄 Reports", "📊 Master History", "📈 Stats"])
            
            with t1:
                if report_doc:
                    st.download_button("📥 Download Word Report", data=report_doc, file_name=f"Report_{dept_name}_{data_month}.docx", type="primary")
                else:
                    st.warning("Report could not be generated.")

                audit_buffer = BytesIO()
                wdf.to_excel(audit_buffer, index=False)
                audit_buffer.seek(0)
                st.download_button("📥 Download Audit Excel", data=audit_buffer, file_name=f"Audit_{dept_name}_{data_month}.xlsx")
            
            with t2:
                st.dataframe(full_history)
                hist_csv = full_history.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download MASTER HISTORY (Save this!)", data=hist_csv, file_name="master_diagnosis_history.csv")
                
            with t3:
                st.dataframe(stats)
