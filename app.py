import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time
import os
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
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.divider()
    st.subheader("📋 Context Metadata")
    dept_name = st.text_input("Department Name", placeholder="e.g., ICU")
    data_month = st.text_input("Data Month/Year", placeholder="e.g., Jan 2026")
    
    st.divider()
    st.info("☁️ **Cloud Mode Tip:** Upload your previous 'Master History' file below to maintain trend continuity.")
    
    # HISTORY UPLOADER
    uploaded_history = st.file_uploader("📂 Upload Previous Master History (Optional)", type=['csv'])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_icd_mapping_from_ai(model, unique_diagnoses):
    """Batched AI processing."""
    mapping_dict = {}
    batch_size = 30
    total_batches = (len(unique_diagnoses) + batch_size - 1) // batch_size
    
    progress_text = "AI is mapping diagnoses..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, start_idx in enumerate(range(0, len(unique_diagnoses), batch_size)):
        batch = unique_diagnoses[start_idx : start_idx + batch_size]
        prompt = f"""
        Act as an expert Medical Coder. Map these raw diagnosis descriptions to ICD-10.
        Return ONLY valid JSON. Format: {{"original text": {{"Level2": "Block Name", "Level3": "Category Name"}}}}
        List: {json.dumps(batch)}
        """
        try:
            response = model.generate_content(prompt)
            clean_json = response.text.replace('```json', '').replace('```', '').strip()
            mapping_dict.update(json.loads(clean_json))
            time.sleep(1) 
            my_bar.progress((i + 1) / total_batches, text=f"Mapping batch {i+1}/{total_batches}")
        except Exception as e:
            pass 
            
    my_bar.empty()
    return mapping_dict

def update_master_history(current_stats_df, dept, month, existing_history_df=None):
    """Merges current stats with uploaded history."""
    current_stats_df['Department'] = dept
    current_stats_df['Month'] = month
    
    if existing_history_df is not None and not existing_history_df.empty:
        # Remove old data for this Dept+Month to prevent duplicates
        history_df = existing_history_df[~((existing_history_df['Department'] == dept) & (existing_history_df['Month'] == month))]
        combined_df = pd.concat([history_df, current_stats_df], ignore_index=True)
    else:
        combined_df = current_stats_df

    return combined_df

def generate_strategic_report(history_df, current_month, dept, model):
    """Generates Word Doc."""
    dept_history = history_df[history_df['Department'] == dept].copy()
    csv_data = dept_history.to_csv(index=False)
    
    prompt = f"""
    You are a Strategic Healthcare Executive. 
    Analyze this HISTORICAL dataset for the '{dept}' department.
    Most recent data: {current_month}.
    
    Dataset (CSV):
    {csv_data}
    
    Write a Strategic Insight Report covering:
    1. **Executive Summary ({current_month})**: Key volume/revenue highlights.
    2. **Trend Analysis**: Compare {current_month} vs previous history. Identify RISING volumes.
    3. **Operational Deep Dive**: Analyze 'Skewed' LOS groups (Outliers).
    4. **Recommendations**: Actionable advice.
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
        return buffer
    except Exception as e:
        return None

# ==========================================
# MAIN APP LOGIC
# ==========================================

# 1. DATA UPLOAD
st.subheader("1. Upload Current Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file and api_key and dept_name and data_month:
    
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 2. MAPPING INTERFACE
    st.divider()
    st.subheader("2. Map Columns")
    cols = list(df.columns)
    c1, c2, c3 = st.columns(3)
    col_id = c1.selectbox("Patient ID", cols, index=0)
    col_diag = c2.selectbox("Diagnosis", cols, index=min(1, len(cols)-1))
    col_rev = c3.selectbox("Revenue", cols, index=min(2, len(cols)-1))
    col_adm = c1.selectbox("Admission Date", cols, index=min(3, len(cols)-1))
    col_disch = c2.selectbox("Discharge Date", cols, index=min(4, len(cols)-1))

    # 3. EXECUTION
    st.divider()
    if st.button("🚀 Run Analysis & Generate Reports", type="primary", use_container_width=True):
        
        genai.configure(api_key=api_key)
        
        # --- MODEL SELECTION LOGIC (UPDATED FOR 2.5) ---
        model = None
        # Priority 1: Gemini 2.5 Flash (As requested)
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            model.generate_content("test") # Test connection
            st.toast("Using Model: Gemini 2.5 Flash", icon="🤖")
        except:
            # Priority 2: Gemini 2.0 Flash Exp (Fallback)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                model.generate_content("test")
                st.toast("2.5 not found. Using Model: Gemini 2.0 Flash", icon="⚠️")
            except:
                # Priority 3: Gemini 1.5 Flash (Safe Fallback)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    st.toast("Using Model: Gemini 1.5 Flash", icon="⚠️")
                except:
                    st.error("Could not connect to ANY Gemini model. Please check your API Key.")
                    st.stop()

        with st.status("Processing... Please wait.", expanded=True) as status:
            
            # A. PREP
            st.write("🔧 Cleaning data & Calculating LOS...")
            wdf = df.rename(columns={col_id:'ID', col_diag:'DIAG', col_rev:'REV', col_adm:'ADM', col_disch:'DISCH'})
            wdf['ADM'] = pd.to_datetime(wdf['ADM'], errors='coerce')
            wdf['DISCH'] = pd.to_datetime(wdf['DISCH'], errors='coerce')
            wdf['LOS'] = (wdf['DISCH'] - wdf['ADM']).dt.days.fillna(0).clip(lower=0)
            wdf['REV'] = pd.to_numeric(wdf['REV'], errors='coerce').fillna(0)
            
            # B. AI MAPPING
            st.write("🧠 AI Mapping Diagnoses...")
            unique = wdf['DIAG'].unique().tolist()
            mapping = get_icd_mapping_from_ai(model, unique)
            wdf['Level3'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level3", "Unmapped"))
            wdf['Level2'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level2", "Unmapped"))
            
            # C. STATS
            st.write("📊 Calculating Statistics...")
            stats = wdf.groupby('Level3').agg({'DIAG':'count', 'LOS':['mean','median'], 'REV':['mean','median']}).reset_index()
            stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
            stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
            
            # D. HISTORY MERGE
            st.write("📜 Updating Historical Database...")
            old_history_df = pd.read_csv(uploaded_history) if uploaded_history else pd.DataFrame()
            full_history = update_master_history(stats, dept_name, data_month, old_history_df)
            
            # E. REPORT GENERATION
            st.write("📝 Writing Strategic Report...")
            report_doc = generate_strategic_report(full_history, data_month, dept_name, model)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # 4. RESULTS DISPLAY & DOWNLOAD
        st.divider()
        st.subheader("3. Results & Downloads")
        
        tab1, tab2, tab3 = st.tabs(["📄 Word Report", "📜 Master History", "📊 Current Stats"])
        
        with tab1:
            st.success("Report Generated Successfully.")
            if report_doc:
                st.download_button("📥 Download Strategic Report (.docx)", data=report_doc, file_name=f"Report_{dept_name}_{data_month}.docx", type="primary")
                
        with tab2:
            st.write("This file contains the trends from this month AND previous uploads.")
            st.dataframe(full_history)
            hist_csv = full_history.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Updated Master History (.csv)", data=hist_csv, file_name="master_diagnosis_history.csv", help="Save this! Upload it next time to keep your trends.")
            
        with tab3:
            st.dataframe(stats)
            stats_csv = stats.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Current Stats (.csv)", data=stats_csv, file_name=f"Stats_{dept_name}_{data_month}.csv")

        audit_buffer = BytesIO()
        wdf.to_excel(audit_buffer, index=False)
        audit_buffer.seek(0)
        st.download_button("📥 Download Audit File (Excel)", data=audit_buffer, file_name=f"Audit_{dept_name}_{data_month}.xlsx")

elif not uploaded_file:
    st.info("👈 Waiting for file upload...")
elif not api_key:
    st.warning("👈 Please enter API Key in Sidebar")
