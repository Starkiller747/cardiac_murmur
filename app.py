import os
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tf_keras as legacy_keras
import matplotlib.pyplot as plt
import pywt
import cv2
import joblib
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
import tempfile


def create_pdf_report(patient_info, overall_verdict, valve_results):
    pdf = FPDF()
    pdf.add_page()
    
    # --- HEADER ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="AI Cardiac Murmur Diagnostic Report", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(10)
    
    # --- CLINICAL METADATA ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Patient Clinical Data:", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, value in patient_info.items():
        pdf.cell(0, 6, txt=f"{key}: {value}", ln=True)
    pdf.ln(5)
    
    # --- OVERALL VERDICT ---
    pdf.set_font("Arial", 'B', 14)
    if "Refer" in overall_verdict:
        pdf.set_text_color(200, 0, 0) # Red for Alert
    else:
        pdf.set_text_color(0, 128, 0) # Green for Normal
        
    pdf.cell(0, 10, txt=f"OVERALL DIAGNOSIS: {overall_verdict}", ln=True)
    pdf.set_text_color(0, 0, 0) # Reset to Black
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # --- VALVE BREAKDOWN ---
    for result in valve_results:
        # Check if we need a page break so images don't get cut off
        if pdf.get_y() > 200: 
            pdf.add_page()
            
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=f"Auscultation Location: {result['valve']} Valve", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 6, txt=f"AI Confidence Score: {result['prob']*100:.1f}%", ln=True)
        pdf.ln(2)
        
        # Insert the waveform and CWT images
        pdf.image(result['wave_img'], w=170)
        pdf.image(result['cwt_img'], w=170)
        pdf.ln(5)
        
    # --- EXPORT TO BYTES ---
    # We save to a temporary file, read the binary data, and delete it.
    # This prevents byte-encoding errors between FPDF and Streamlit.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
            
    # Cleanup temp images
    for result in valve_results:
        if os.path.exists(result['wave_img']): os.remove(result['wave_img'])
        if os.path.exists(result['cwt_img']): os.remove(result['cwt_img'])
            
    return pdf_bytes


# --- 1. CONFIGURATION & CACHING ---
st.set_page_config(page_title="Cardiac Murmur Detection", layout="wide")

@st.cache_resource
def load_trained_model():
    # Caching prevents Streamlit from reloading the heavy model on every button click
    return legacy_keras.models.load_model(
        'model_data/murmur_fusion_model.keras', 
        compile=False
    )

@st.cache_resource
def load_scaler():
    return joblib.load('model_data/hw_scaler.pkl')


def slice_audio_windows(y, sr=8000, window_sec=3.0, hop_sec=1.5):
    """
    Slices an already-loaded audio array into fixed-length overlapping windows.
    y: The audio time series array.
    sr: The sampling rate of y.
    """
    # Notice we REMOVED the librosa.load() line here!
    
    # Convert seconds to sample indices
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    
    clips = []
    
    # Slide the window across the audio array
    for start in range(0, len(y) - window_len + 1, hop_len):
        end = start + window_len
        clip = y[start:end]
        clips.append(clip)
        
    return clips


def create_cwt_image(clip, wavelet='morl', target_shape=(256, 100)):
    scales = np.geomspace(5, 150, num=128)
    coef, _ = pywt.cwt(clip, scales, wavelet)
    coef = np.abs(coef)
    c_min, c_max = np.min(coef), np.max(coef)
    coef = (coef - c_min) / (c_max - c_min + 1e-8)
    coef = cv2.resize(coef, (target_shape[1], target_shape[0]))
    return coef

model = load_trained_model()
scaler = load_scaler()

# --- 2. USER INTERFACE (SIDEBAR) ---
st.sidebar.header("Patient Clinical Data")

# Collect Clinical Data
age_input = st.sidebar.selectbox(
    "Age Group", 
    ["Neonate", "Infant", "Child", "Adolescent", "Adult"],
    index=2 # Defaults to "Child" since it's the most common in this pediatric dataset
)
sex_input = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_input == "Female" else 0
height = st.sidebar.number_input("Height (cm)", value=100.0)
weight = st.sidebar.number_input("Weight (kg)", value=35.0)
pregnant_input = st.sidebar.checkbox("Pregnant?")
pregnant = 1 if pregnant_input else 0

age_mapping = {'Neonate': 0, 'Infant': 1, 'Child': 2, 'Adolescent': 3, 'Adult': 4}

age_encoded = age_mapping[age_input]

raw_hw_df = pd.DataFrame([[height, weight]], columns=['Height', 'Weight'])
scaled_hw = scaler.transform(raw_hw_df)
height_scaled = scaled_hw[0][0]
weight_scaled = scaled_hw[0][1]


# --- 3. MAIN DASHBOARD ---
st.title("🫀 AI Cardiac Murmur Detection Report")
st.write("Upload the patient's ECG/Auscultation audio files below.")

st.warning("Make sure to fill the data from the sidebar first before uploading the audio files.")

st.info("For the audio files, make sure they have the valve type in the name like test_AV for the Atrioventricular, TV for the Tricuspid Valve, MV for Mitral Valve and PV for Pulmonary Valve.")

uploaded_files = st.file_uploader("Choose .wav files", type=['wav'], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    st.subheader("📋 Comprehensive Diagnostic Report")
    
    patient_max_prob = 0.0
    valve_results_for_pdf = [] # NEW: We will store our charts here!
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        valve_AV, valve_MV, valve_PV, valve_TV = 0, 0, 0, 0
        if "MV" in filename.upper(): valve_MV = 1
        elif "PV" in filename.upper(): valve_PV = 1
        elif "TV" in filename.upper(): valve_TV = 1
        else: valve_AV = 1
        
        valve_name = "Mitral" if valve_MV else "Pulmonary" if valve_PV else "Tricuspid" if valve_TV else "Aortic"
        
        clinical_features = np.array([[age_encoded, sex, height_scaled, weight_scaled, pregnant, valve_AV, valve_MV, valve_PV, valve_TV]], dtype=np.float32)

        uploaded_file.seek(0)
        y, sr = librosa.load(uploaded_file, sr=8000)
        clips = slice_audio_windows(y, sr=sr)
        
        X_audio_windows = []
        for clip in clips:
            spec = create_cwt_image(clip)
            spec = np.log1p(np.abs(spec))
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
            X_audio_windows.append(spec[..., np.newaxis])
            
        X_audio = np.array(X_audio_windows, dtype=np.float32)
        X_clinical = np.repeat(clinical_features, len(X_audio), axis=0)
        
        predictions = model.predict({"audio_input": X_audio, "clinical_input": X_clinical})
        
        OPTIMAL_THRESHOLD = 0.5623
        max_prob_index = np.argmax(predictions)
        max_prob = predictions[max_prob_index][0]
        
        if max_prob > patient_max_prob:
            patient_max_prob = max_prob
            
        is_abnormal = max_prob > OPTIMAL_THRESHOLD
        
        with st.expander(f"🩺 Analysis: {valve_name} Valve ({filename})", expanded=is_abnormal):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if is_abnormal: st.error(f"⚠️ **Alert:** Murmur detected.")
                else: st.success(f"✅ **Clear:** No murmur detected.")
                st.metric(label="Peak Window Confidence", value=f"{max_prob*100:.1f}%")
                
            with col2:
                tab1, tab2 = st.tabs(["Raw Waveform", "AI Spectrogram (CWT)"])
                
                # Plot and SAVE the Waveform
                fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
                ax_wave.plot(y, color='#1f77b4', linewidth=0.5)
                ax_wave.set_axis_off()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wave:
                    fig_wave.savefig(tmp_wave.name, bbox_inches='tight')
                    wave_path = tmp_wave.name
                    
                with tab1: st.pyplot(fig_wave)
                plt.close(fig_wave)
                
                # Plot and SAVE the CWT
                suspicious_cwt = X_audio[max_prob_index][:, :, 0]
                fig_cwt, ax_cwt = plt.subplots(figsize=(6, 2))
                cax = ax_cwt.imshow(suspicious_cwt, aspect='auto', cmap='jet', origin='lower')
                ax_cwt.set_axis_off()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_cwt:
                    fig_cwt.savefig(tmp_cwt.name, bbox_inches='tight')
                    cwt_path = tmp_cwt.name
                    
                with tab2: st.pyplot(fig_cwt)
                plt.close(fig_cwt)
                
        # Append data for this valve to our PDF list
        valve_results_for_pdf.append({
            'valve': valve_name,
            'prob': max_prob,
            'wave_img': wave_path,
            'cwt_img': cwt_path
        })

    # --- FINAL PATIENT VERDICT & PDF GENERATION ---
    st.divider()
    
    if patient_max_prob > OPTIMAL_THRESHOLD:
        verdict_text = "Refer to Cardiologist (Potential Malignant Murmur)"
        st.error(f"### 🚨 OVERALL DIAGNOSIS: {verdict_text}")
    else:
        verdict_text = "Normal (No Evidence of Malignant Murmur)"
        st.success(f"### 🟢 OVERALL DIAGNOSIS: {verdict_text}")
        
    # Package the dictionary of patient data to pass to the PDF
    patient_metadata = {
        "Age Group": age_input,
        "Sex": sex_input,
        "Height": f"{height} cm",
        "Weight": f"{weight} kg",
        "Pregnant": "Yes" if pregnant_input else "No"
    }

    # Generate the PDF bytes in the background
    pdf_bytes = create_pdf_report(patient_metadata, verdict_text, valve_results_for_pdf)
    
    # Render the Download Button
    st.download_button(
        label="📄 Download Formal PDF Report",
        data=pdf_bytes,
        file_name="Cardiac_Murmur_Report.pdf",
        mime="application/pdf"
    )