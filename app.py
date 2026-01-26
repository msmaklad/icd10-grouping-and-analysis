import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time
import os
import zipfile
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
    # --- 1. INSTRUCTIONS SECTION ---
    with st.expander("📖 **Step-by-Step Guide**", expanded=True):
        st.markdown("""
        **1. Initialize History:**
           - If you have a history file (e.g., `Jan_2026-ICU-history.csv`), select **"Yes"** and upload it.
           - If this is your first time, select **"No (Start Fresh)"**.
        
        **2. Configure:**
           - Enter your **API Keys**.
           - Select **Department** & **Month**.
        
        **3. Upload Data:**
           - Upload your raw hospital file (CSV/Excel).
           - Map the columns (Diagnosis, Date, etc.) to match your file.
           
        **4. Run:**
           - Click **🚀 Run Analysis**.
           - Wait for AI mapping & report generation.
           
        **5. Download:**
           - Click **📦 DOWNLOAD ALL FILES**.
           - **Crucial:** The ZIP contains your new `(Month)-(Dept)-history.csv`. Save it!
        """)
    
    st.divider()
    st.header("⚙️ Configuration")
    
    # --- 2. API KEYS SECTION ---
    st.subheader("🔑 API Access")
    st.markdown("To ensure unlimited usage, please use your own **Free** Google Gemini Keys.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("🛠️ Get Key", "https://aistudio.google.com/app/apikey")
    with c2:
        st.link_button("🎥 Tutorial", "https://www.youtube.com/results?search_query=how+to+get+google+gemini+api+key")

    api_keys_input = st.text_area("Enter API Keys (One per line)", height=100, placeholder="Paste keys here...")
    api_keys = [k.strip() for k in api_keys_input.split('\n') if k.strip()]
    
    st.divider()
    
    # --- 3. STANDARDIZED SELECTIONS (UPDATED LIST) ---
    dept_options = ["ICU", "CCU", "PICU", "Inpatient Ward", "ER"]
    dept_name = st.selectbox("Department Name", options=dept_options)
    
    col_m, col_y = st.columns(2)
    with col_m: sel_month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col_y: sel_year = st.selectbox("Year", [str(y) for y in range(2024, 2030)])
    data_month = f"{sel_month} {sel_year}"
    
    st.divider()
    if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0
    quota_placeholder = st.empty()
    quota_placeholder.metric("API Calls This Session", st.session_state['api_calls'])
    
    # --- 4. RESET BUTTON ---
    st.divider()
    if st.button("🔄 Reset App State", use_container_width=True):
        st.rerun()

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
    batch_size = 400 
    
    current_key_idx = 0
    # --- UPDATED MODEL LIST ---
    models_to_try = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash']
    current_model_idx = 0
    
    genai.configure(api_key=keys_list[current_key_idx])
    model = genai.GenerativeModel(models_to_try[current_model_idx])
    
    total_batches = (len(unique_diagnoses) + batch_size - 1) // batch_size
    pbar = st.progress(0, text="AI Analysis Starting...")
    
    batches = [unique_diagnoses[i:i + batch_size] for i in range(0, len(unique_diagnoses), batch_size)]
    
    for i, batch in enumerate(batches):
        success = False
        max_attempts = len(keys_list) * 2 
        
        for attempt in range(max_attempts):
            try:
                json_payload = json.dumps(batch, default=str)
                prompt = f"""
                Act as a Medical Coder. Map these diagnosis strings to ICD-10.
                Return a JSON object: key = original text, value = {{"Level2": "Block Name", "Level3": "Category Name"}}
                NO markdown. JUST JSON.
                List: {json_payload}
                """
                response = model.generate_content(prompt)
                
                st.session_state['api_calls'] += 1
                quota_placeholder.metric("API Calls This Session", st.session_state['api_calls'])
                
                data = extract_json_from_text(response.text)
                if data: 
                    mapping_dict.update(data)
                    success = True
                    break 
                
            except Exception as e:
                err = str(e)
                if "429" in err or "Quota" in err:
                    if len(keys_list) > 1:
                        current_key_idx = (current_key_idx + 1) % len(keys_list)
                        st.toast(f"Quota hit. Rotating Key...", icon="🔄")
                        genai.configure(api_key=keys_list[current_key_idx])
                        model = genai.GenerativeModel(models_to_try[current_model_idx])
                        time.sleep(1)
                        continue
                    
                    if current_model_idx < len(models_to_try) - 1:
                        current_model_idx += 1
                        st.warning(f"Quota hit. Downgrading to {models_to_try[current_model_idx]}...")
                        model = genai.GenerativeModel(models_to_try[current_model_idx])
                        time.sleep(1)
                        continue
                    
                    st.error("❌ **QUOTA FAILURE** Daily limit reached. Wait 24h or add new keys.")
                    st.stop()
                else:
                    print(f"Batch Error: {e}") 
                    time.sleep(1)
        
        if success:
            pbar.progress((i + 1) / total_batches, text=f"Processed batch {i+1}/{total_batches}")
            time.sleep(1)
            
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
    
    if dept_hist.empty: return None, "No history found for this department."
    
    prompt = f"""
    Act as a **Dual-Role Executive: Strategic Director & Clinical Governance Lead**.
    Your task is to produce a comprehensive **Integrated Case Mix & Strategic Analysis Report** for the '{dept}' department.
    
    **Context:**
    - Current Period: {month}
    - Data Source: Historical ICD-10 Diagnosis Groups (Level 3)
    
    **Dataset (CSV):**
    {dept_hist.to_csv(index=False)}
    
    **Report Structure:**
    1. **Executive Summary** (Business & Clinical)
    2. **Trend & Case Mix Analysis** (Volume & Severity Shifts)
    3. **Operational Efficiency & Clinical Governance** (Skewed LOS & Risk)
    4. **Revenue Integrity** (Profitability vs Cost)
    5. **Integrated Recommendations** (Pathways, Capacity, Coding)
    """
    
    try:
        response = model.generate_content(prompt)
        doc = Document()
        doc.add_heading(f'Strategic Report: {dept} - {month}', 0)
        doc.add_paragraph(response.text)
        b = BytesIO()
        doc.save(b)
        b.seek(0)
        return b, None 
    except Exception as e: 
        return None, str(e)

# ==========================================
# MAIN APP - WORKFLOW ENFORCEMENT
# ==========================================

if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'zip_buffer' not in st.session_state:
    st.session_state['zip_buffer'] = None
if 'history_filename' not in st.session_state:
    st.session_state['history_filename'] = "master_history.csv"

# STEP 1: MANDATORY HISTORY LOAD
st.subheader("1. Initialize History (Mandatory)")
st.info("To maintain trends, please upload your previous history file (e.g., `Jan_2026-ICU-history.csv`).")

history_mode = st.radio("Do you have a previous history file?", ["Yes, I have it", "No, this is Day 1 (Start Fresh)"], horizontal=True)

history_df = pd.DataFrame() 

if history_mode == "Yes, I have it":
    uploaded_history = st.file_uploader("📂 Upload History CSV", type=['csv'], key="hist_up")
    if uploaded_history:
        try:
            history_df = pd.read_csv(uploaded_history)
            st.success(f"✅ History Loaded! Contains {len(history_df)} records.")
        except:
            st.error("Invalid CSV file.")
    else:
        st.warning("⚠️ You must upload the history file to proceed.")
        st.stop()
else:
    st.success("✅ Starting with a fresh history database.")

st.divider()

# STEP 2: DATA UPLOAD
st.subheader("2. Upload Current Month Data")
uploaded_file = st.file_uploader("Upload Raw Hospital Data (CSV/Excel)", type=['csv', 'xlsx'], key="data_up")

if uploaded_file and api_keys:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head(3))
        
        st.divider()
        st.subheader("3. Map Columns & Run")
        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        col_id = c1.selectbox("Patient ID", cols, index=0)
        col_diag = c2.selectbox("Diagnosis", cols, index=min(1, len(cols)-1))
        col_rev = c3.selectbox("Revenue", cols, index=min(2, len(cols)-1))
        col_adm = c1.selectbox("Admission Date", cols, index=min(3, len(cols)-1))
        col_disch = c2.selectbox("Discharge Date", cols, index=min(4, len(cols)-1))
        
        if st.button("🚀 Run Analysis & Update History", type="primary"):
            with st.status("Processing...", expanded=True):
                # CLEAN
                st.write("🔧 Cleaning data...")
                wdf = df.rename(columns={col_id:'ID', col_diag:'DIAG', col_rev:'REV', col_adm:'ADM', col_disch:'DISCH'})
                wdf = wdf.dropna(subset=['DIAG']) 
                wdf['DIAG'] = wdf['DIAG'].astype(str) 
                wdf = wdf[wdf['DIAG'].str.strip() != ''] 
                wdf['ADM'] = pd.to_datetime(wdf['ADM'], errors='coerce')
                wdf['DISCH'] = pd.to_datetime(wdf['DISCH'], errors='coerce')
                wdf['LOS'] = (wdf['DISCH'] - wdf['ADM']).dt.days.fillna(0).clip(lower=0)
                wdf['REV'] = pd.to_numeric(wdf['REV'], errors='coerce').fillna(0)
                
                # MAP
                st.write(f"🧠 Mapping {len(wdf)} records...")
                unique = wdf['DIAG'].unique().tolist()
                mapping = get_icd_mapping_optimized(api_keys, unique)
                
                if not mapping:
                    st.error("AI Mapping Failed.")
                    st.stop()

                wdf['Level3'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level3", "Unmapped"))
                
                # STATS
                stats = wdf.groupby('Level3').agg({'DIAG':'count', 'LOS':['mean','median'], 'REV':['mean','median']}).reset_index()
                stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
                stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
                
                # MERGE HISTORY
                st.write("📜 Integrating with Master History...")
                full_hist = update_master_history(stats, dept_name, data_month, history_df)
                
                # REPORT
                st.write("📝 Generating Strategic Report...")
                report_doc, report_error = generate_report(full_hist, data_month, dept_name, api_keys)
                
                # GENERATE DYNAMIC FILE NAMES
                safe_dept = dept_name.replace(" ", "_").replace("/", "-")
                safe_month = data_month.replace(" ", "_")
                
                history_filename = f"{safe_month}-{safe_dept}-history.csv"
                report_filename = f"Report_{safe_month}-{safe_dept}.docx"
                audit_filename = f"Audit_{safe_month}-{safe_dept}.xlsx"
                zip_filename = f"Hospital_Analysis_{safe_month}-{safe_dept}.zip"
                
                # ZIP CREATION
                st.write("📦 Packaging Files...")
                audit_buffer = BytesIO()
                wdf.to_excel(audit_buffer, index=False)
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # WRITE HISTORY WITH NEW NAME
                    zf.writestr(history_filename, full_hist.to_csv(index=False))
                    
                    if report_doc:
                        zf.writestr(report_filename, report_doc.getvalue())
                    zf.writestr(audit_filename, audit_buffer.getvalue())
                    
                zip_buffer.seek(0)
                
                st.session_state['zip_buffer'] = zip_buffer
                st.session_state['analysis_done'] = True
                st.session_state['full_hist'] = full_hist
                st.session_state['zip_filename'] = zip_filename
                
            st.success("Analysis Complete!")
            st.rerun()

    except Exception as e:
        st.error(f"Critical App Error: {e}")

# RESULTS DISPLAY
if st.session_state.get('analysis_done') and st.session_state.get('zip_buffer'):
    st.divider()
    st.subheader("✅ Results Ready")
    st.warning("⬇️ **STEP 4: Download Your Files**")
    
    st.download_button(
        label="📦 DOWNLOAD ALL FILES (ZIP)",
        data=st.session_state['zip_buffer'],
        file_name=st.session_state['zip_filename'],
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
    
    st.divider()
    col_reset, _ = st.columns([1,3])
    with col_reset:
        if st.button("🔄 Start New Analysis (Reset Page)", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.divider()
    if 'full_hist' in st.session_state:
        st.write("📊 **History Preview:**")
        st.dataframe(st.session_state['full_hist'].tail(10))
