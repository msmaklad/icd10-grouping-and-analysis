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
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API KEYS
    api_keys_input = st.text_area("Enter Google Gemini API Keys (One per line)", height=150)
    api_keys = [k.strip() for k in api_keys_input.split('\n') if k.strip()]
    
    st.divider()
    
    # 2. STANDARDIZED SELECTIONS
    dept_options = ["ICU", "Emergency Room (ER)", "Inpatient Ward", "CCU", "PICU", "NICU"]
    dept_name = st.selectbox("Department Name", options=dept_options)
    
    col_m, col_y = st.columns(2)
    with col_m: sel_month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    with col_y: sel_year = st.selectbox("Year", [str(y) for y in range(2024, 2030)])
    data_month = f"{sel_month} {sel_year}"
    
    st.divider()
    uploaded_history = st.file_uploader("📂 Upload Previous Master History (Optional)", type=['csv'])
    
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
    batch_size = 400 
    
    current_key_idx = 0
    models_to_try = ['gemini-2.5-flash']
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
                    
                    st.error(f"""
                    ❌ **QUOTA FAILURE** The AI has stopped accepting requests.
                    - **Reason:** Daily limit reached for all keys.
                    - **Limit:** ~20 requests/day (Gemini 2.5 Flash Free Tier).
                    
                    **Action Required:**
                    1. Wait 24 hours.
                    2. Or add a fresh API key in the sidebar.
                    """)
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
    if dept_hist.empty: return None
    
    # --- UPDATED PROMPT: DUAL ROLE ---
    prompt = f"""
    Act as a **Dual-Role Executive: Strategic Director & Clinical Governance Lead**.
    Your task is to produce a comprehensive **Integrated Case Mix & Strategic Analysis Report** for the '{dept}' department.
    
    **Context:**
    - Current Period: {month}
    - Data Source: Historical ICD-10 Diagnosis Groups (Level 3)
    
    **Dataset (CSV):**
    {dept_hist.to_csv(index=False)}
    
    **Report Structure & Requirements:**
    
    1. **Executive Summary (Business & Clinical):**
       - High-level overview of patient volume, revenue performance, and clinical stability for {month}.
       - Highlight the most critical "Takeaway" for hospital leadership.

    2. **Trend & Case Mix Analysis:**
       - **Volume Trends:** Compare {month} vs previous months. Identify which conditions are rising (Historical Trend).
       - **Case Mix Profile:** Identify the "Top 5" diagnosis groups by volume. Are we seeing a shift towards more complex/severe cases?

    3. **Operational Efficiency & Clinical Governance:**
       - **LOS Outlier Analysis:** Analyze rows where 'Skew Status' is 'Skewed' (Mean LOS > Median LOS).
       - **Risk Implication:** Explain that these skewed groups represent potential **Clinical Risks** (complications/safety issues) and **Operational Bottlenecks** (bed blocking).
       - Identify specific conditions causing the biggest efficiency drag.

    4. **Revenue Integrity & Resource Utilization:**
       - Compare "High Revenue" groups against "Long LOS" groups.
       - **Profitability vs. Cost:** Are high-revenue cases being managed efficiently, or is the extended length of stay eroding the margin?

    5. **Integrated Recommendations (Action Plan):**
       - **Strategic:** Advice on capacity planning or resource allocation based on volume trends.
       - **Clinical Governance:** Recommend specific **Clinical Pathways** or protocols for the high-skew/outlier groups to improve safety.
       - **Documentation:** Advice on improving coding specificity if "Unmapped" or vague codes appear.
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
                # 1. CLEAN DATA
                st.write("🔧 Cleaning data...")
                wdf = df.rename(columns={col_id:'ID', col_diag:'DIAG', col_rev:'REV', col_adm:'ADM', col_disch:'DISCH'})
                
                # FORCE STRING & DROP EMPTY
                wdf = wdf.dropna(subset=['DIAG']) 
                wdf['DIAG'] = wdf['DIAG'].astype(str) 
                wdf = wdf[wdf['DIAG'].str.strip() != ''] 
                
                # Date & Num processing
                wdf['ADM'] = pd.to_datetime(wdf['ADM'], errors='coerce')
                wdf['DISCH'] = pd.to_datetime(wdf['DISCH'], errors='coerce')
                wdf['LOS'] = (wdf['DISCH'] - wdf['ADM']).dt.days.fillna(0).clip(lower=0)
                wdf['REV'] = pd.to_numeric(wdf['REV'], errors='coerce').fillna(0)
                
                # 2. AI MAPPING
                st.write(f"🧠 Mapping {len(wdf)} records...")
                unique = wdf['DIAG'].unique().tolist()
                
                if not unique:
                    st.error("No valid diagnoses found.")
                    st.stop()
                    
                mapping = get_icd_mapping_optimized(api_keys, unique)
                
                if not mapping:
                    st.error("AI Mapping Failed completely.")
                    st.stop()

                wdf['Level3'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level3", "Unmapped"))
                wdf['Level2'] = wdf['DIAG'].map(lambda x: mapping.get(x, {}).get("Level2", "Unmapped"))
                
                # 3. STATS
                stats = wdf.groupby('Level3').agg({'DIAG':'count', 'LOS':['mean','median'], 'REV':['mean','median']}).reset_index()
                stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
                stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
                
                # 4. REPORT
                st.write("📝 Reporting...")
                old_hist = pd.read_csv(uploaded_history) if uploaded_history else pd.DataFrame()
                full_hist = update_master_history(stats, dept_name, data_month, old_hist)
                report_doc = generate_report(full_hist, data_month, dept_name, api_keys)
                
            st.success("Done!")
            
            t1, t2 = st.tabs(["📄 Results", "📜 History"])
            with t1:
                if report_doc: st.download_button("📥 Word Report", report_doc, f"Report_{dept_name}_{data_month}.docx")
                
                audit_buffer = BytesIO()
                wdf.to_excel(audit_buffer, index=False)
                audit_buffer.seek(0)
                st.download_button("📥 Audit Excel", data=audit_buffer, file_name=f"Audit_{dept_name}_{data_month}.xlsx")

            with t2:
                st.dataframe(full_hist)
                st.download_button("📥 Master History (Save This!)", full_hist.to_csv(index=False).encode(), "master_history.csv")

    except Exception as e:
        st.error(f"Error: {e}")
