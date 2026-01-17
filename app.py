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
    api_keys_input = st.text_area(
        "Enter Google Gemini API Keys (One per line)", 
        height=150,
        type="password",
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
        combined_df = pd.concat
