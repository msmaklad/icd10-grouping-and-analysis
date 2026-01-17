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
st.set_page_config(
    page_title="ICD-10 Hospital Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Intelligence: ICD-10 Trend Analyzer")
st.markdown("""
This tool uses **AI (Gemini)** to standardize hospital data into ICD-10 groups and generate strategic insights.
It maintains a historical trend file to track performance over time.
""")

# ==========================================
# SIDEBAR: CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get this from aistudio.google.com")
    
    st.divider()
    
    # Metadata Inputs
    st.subheader("📋 Context Metadata")
    dept_name = st.text_input("Department Name", placeholder="e.g., ICU, ER, Ward A")
    data_month = st.text_input("Data Month/Year", placeholder="e.g., Jan 2026")
    
    st.info("ℹ️ The 'Master History' file is updated automatically to enable trend analysis across months.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_icd_mapping_from_ai(model, unique_diagnoses):
    """Batched AI processing with a visual progress bar."""
    mapping_dict = {}
    # Process in batches to avoid API timeouts
    batch_size = 30 
    total_batches = (len(unique_diagnoses) + batch_size - 1) // batch_size
    
    progress_text = "AI is mapping diagnoses to ICD-10 standards..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, start_idx in enumerate(range(0, len(unique_diagnoses), batch_size)):
        batch = unique_diagnoses[start_idx : start_idx + batch_size]
        
        prompt = f"""
        Act as an expert Medical Coder. Map these raw diagnosis descriptions to ICD-10.
        
        For each description, return a JSON object with:
        - "Level2": The ICD-10 Chapter/Block text (e.g., "Influenza and pneumonia").
        - "Level3": The specific ICD-10 Category text (e.g., "Influenza due to identified virus").
        
        Return ONLY valid JSON. 
        Format: {{"original text": {{"Level2": "Block Name", "Level3": "Category Name"}}}}
        
        List to map:
        {json.dumps(batch)}
        """
        try:
            response = model.generate_content(prompt)
            # Clean response to ensure valid JSON
            clean_json = response.text.replace('```json', '').replace('```', '').strip()
            batch_result = json.loads(clean_json)
            mapping_dict.update(batch_result)
            
            # Update progress
            time.sleep(1) # Rate limit politeness
            my_bar.progress((i + 1) / total_batches, text=f"Mapping batch {i+1} of {total_batches}...")
            
        except Exception as e:
            st.warning(f"Batch {i+1} skipped due to AI error: {e}")
            
    my_bar.empty()
    return mapping_dict

def update_master_history(current_stats_df, dept, month):
    """Updates the permanent history file for trend analysis."""
    history_file = "master_diagnosis_history.csv"
    
    # Tag current data
    current_stats_df['Department'] = dept
    current_stats_df['Month'] = month
    
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        # Prevent duplicates: Remove old data for this specific Dept+Month
        history_df = history_df[~((history_df['Department'] == dept) & (history_df['Month'] == month))]
        combined_df = pd.concat([history_df, current_stats_df], ignore_index=True)
    else:
        combined_df = current_stats_df

    combined_df.to_csv(history_file, index=False)
    return combined_df

def generate_strategic_report(history_df, current_month, dept, model):
    """Generates the Word document report."""
    # Filter for the specific department's history
    dept_history = history_df[history_df['Department'] == dept].copy()
    
    # Convert to CSV string for the AI
    csv_data = dept_history.to_csv(index=False)
    
    prompt = f"""
    You are a Strategic Healthcare Executive. 
    Analyze this HISTORICAL dataset for the '{dept}' department.
    The most recent data is for month: {current_month}.
    
    Dataset (CSV):
    {csv_data}
    
    Write a Strategic Insight Report covering:
    1. **Executive Summary ({current_month})**: Key volume and revenue highlights.
    2. **Trend Analysis (Historical)**: Compare {current_month} vs previous months. Identify RISING volumes or changing efficiency.
    3. **Operational Deep Dive**: Analyze rows where 'Skew Status' is 'Skewed'. Explain that Mean LOS > Median LOS implies outliers.
    4. **Recommendations**: Actionable advice.
    """
    
    try:
        response = model.generate_content(prompt)
        
        # Create Word Doc
        doc = Document()
        doc.add_heading(f'Strategic Trend Report: {dept}', 0)
        doc.add_heading(f'Period: Ending {current_month}', 1)
        doc.add_paragraph(response.text)
        
        # Save to BytesIO for download
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None

# ==========================================
# MAIN APP LOGIC
# ==========================================

# 1. FILE UPLOAD
uploaded_file = st.file_uploader("📂 Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    # Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("### 🔍 Data Preview")
        st.dataframe(df.head(3))
        
        # Check if config is ready
        if not api_key or not dept_name or not data_month:
            st.warning("⚠️ Please fill in the API Key, Department, and Month in the sidebar to proceed.")
            st.stop()
            
        # 2. COLUMN MAPPING
        st.divider()
        st.write("### 🗺️ Column Mapping")
        st.caption("Select the columns from your file that match the required fields.")
        
        cols = list(df.columns)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            c_id = st.selectbox("Patient ID", cols, index=0)
            c_rev = st.selectbox("Revenue / Cost", cols, index=min(2, len(cols)-1))
        with col2:
            c_diag = st.selectbox("Diagnosis (Text)", cols, index=min(1, len(cols)-1))
            c_adm = st.selectbox("Admission Date", cols, index=min(3, len(cols)-1))
        with col3:
            st.write("") # Spacer
            c_disch = st.selectbox("Discharge Date", cols, index=min(4, len(cols)-1))

        # 3. RUN BUTTON
        st.divider()
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            
            # --- INITIALIZE AI ---
            genai.configure(api_key=api_key)
            try:
                # Try newer model first
                model = genai.GenerativeModel('gemini-2.0-flash-exp') 
                # Note: If 2.0 isn't available to your key yet, it catches error below
                model.generate_content("test")
            except:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                except:
                    st.error("Invalid API Key or Model access. Please check your key.")
                    st.stop()

            # --- PROCESS DATA ---
            with st.status("Processing Data...", expanded=True) as status:
                
                # A. Clean & Calc LOS
                st.write("Cleaning data & Calculating Length of Stay (LOS)...")
                working_df = df.rename(columns={
                    c_id: 'PATIENT ID',
                    c_diag: 'DIAGNOSIS',
                    c_rev: 'REVENUE',
                    c_adm: 'ADM_DATE',
                    c_disch: 'DISCH_DATE'
                })
                
                working_df['ADM_DATE'] = pd.to_datetime(working_df['ADM_DATE'], errors='coerce')
                working_df['DISCH_DATE'] = pd.to_datetime(working_df['DISCH_DATE'], errors='coerce')
                
                # Calculate LOS (Days)
                working_df['LOS'] = (working_df['DISCH_DATE'] - working_df['ADM_DATE']).dt.days
                working_df['LOS'] = working_df['LOS'].fillna(0).clip(lower=0) # Handle errors
                working_df['REVENUE'] = pd.to_numeric(working_df['REVENUE'], errors='coerce').fillna(0)
                
                # B. AI Mapping
                st.write("Consulting AI for Medical Coding...")
                unique_diagnoses = working_df['DIAGNOSIS'].unique().tolist()
                mapping_dict = get_icd_mapping_from_ai(model, unique_diagnoses)
                
                # Apply Map
                working_df['Level2'] = working_df['DIAGNOSIS'].map(lambda x: mapping_dict.get(x, {}).get("Level2", "Unmapped"))
                working_df['Level3'] = working_df['DIAGNOSIS'].map(lambda x: mapping_dict.get(x, {}).get("Level3", "Unmapped"))
                
                # C. Statistics
                st.write("Generating Statistics...")
                stats = working_df.groupby('Level3').agg({
                    'DIAGNOSIS': 'count',
                    'LOS': ['mean', 'median'],
                    'REVENUE': ['mean', 'median']
                }).reset_index()
                
                stats.columns = ['Diagnosis Group', 'Cases', 'Mean LOS', 'Median LOS', 'Mean Rev', 'Median Rev']
                
                # Skew Logic
                stats['Skew Status'] = stats.apply(lambda x: 'Skewed' if x['Mean LOS'] > (x['Median LOS']*1.5) else 'Normal', axis=1)
                
                # D. History & Report
                st.write("Updating Historical Trends & Writing Report...")
                full_history = update_master_history(stats, dept_name, data_month)
                report_buffer = generate_strategic_report(full_history, data_month, dept_name, model)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            # --- DOWNLOADS ---
            st.success("Analysis Finished! Download your results below.")
            
            d1, d2, d3 = st.columns(3)
            
            # 1. Audit Excel
            audit_buffer = BytesIO()
            working_df.to_excel(audit_buffer, index=False)
            audit_buffer.seek(0)
            d1.download_button("📥 Audit Data (Excel)", data=audit_buffer, file_name=f"Audit_{dept_name}_{data_month}.xlsx")
            
            # 2. Stats CSV
            stats_csv = stats.to_csv(index=False).encode('utf-8')
            d2.download_button("📥 Statistics (CSV)", data=stats_csv, file_name=f"Stats_{dept_name}_{data_month}.csv")
            
            # 3. Word Report
            if report_buffer:
                d3.download_button("📥 Strategic Report (Docx)", data=report_buffer, file_name=f"Report_{dept_name}_{data_month}.docx")

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

else:
    st.info("👆 Please upload a CSV or Excel file to begin.")