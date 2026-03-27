import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import pywt
import cv2
import joblib
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION & CACHING ---
st.set_page_config(page_title="Cardiac Murmur Detection", layout="wide")

@st.cache_resource
def load_trained_model():
    # Caching prevents Streamlit from reloading the heavy model on every button click
    return tf.keras.models.load_model('murmur_fusion_model.keras')

@st.cache_resource
def load_scaler():
    return joblib.load('hw_scaler.pkl')


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
height = st.sidebar.number_input("Height (cm)", value=170.0)
weight = st.sidebar.number_input("Weight (kg)", value=70.0)
pregnant_input = st.sidebar.checkbox("Pregnant?")
pregnant = 1 if pregnant_input else 0

age_mapping = {'Neonate': 0, 'Infant': 1, 'Child': 2, 'Adolescent': 3, 'Adult': 4}

age_encoded = age_mapping[age_input]

raw_hw = np.array([[height, weight]])
scaled_hw = scaler.transform(raw_hw)
height_scaled = scaled_hw[0][0]
weight_scaled = scaled_hw[0][1]


# --- 3. MAIN DASHBOARD ---
st.title("🫀 AI Cardiac Murmur Detection Report")
st.write("Upload the patient's ECG/Auscultation audio files below. (Ensure filenames contain the valve, e.g., 'patient_AV.wav')")

# NEW: Allow multiple files to be uploaded!
uploaded_files = st.file_uploader("Choose .wav files", type=['wav'], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    st.subheader("📋 Comprehensive Diagnostic Report")
    
    # We will track the highest probability across ALL uploaded files
    patient_max_prob = 0.0
    
    # Loop through every file the doctor uploaded
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        # 1. Deduce the valve from the filename (Fallback to AV if it can't find one)
        valve_AV, valve_MV, valve_PV, valve_TV = 0, 0, 0, 0
        if "MV" in filename.upper(): valve_MV = 1
        elif "PV" in filename.upper(): valve_PV = 1
        elif "TV" in filename.upper(): valve_TV = 1
        else: valve_AV = 1 # Default
        
        valve_name = "Mitral" if valve_MV else "Pulmonary" if valve_PV else "Tricuspid" if valve_TV else "Aortic"
        
        # 2. Create the Clinical Array strictly for THIS specific file
        clinical_features = np.array([[age_encoded, sex, height_scaled, weight_scaled, pregnant, valve_AV, valve_MV, valve_PV, valve_TV]], dtype=np.float32)

        # 3. Process the Audio
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
        
        # 4. Predict
        predictions = model.predict({"audio_input": X_audio, "clinical_input": X_clinical})
        
        OPTIMAL_THRESHOLD = 0.5623
        max_prob_index = np.argmax(predictions)
        max_prob = predictions[max_prob_index][0]
        
        # Track the highest probability overall for the final patient diagnosis
        if max_prob > patient_max_prob:
            patient_max_prob = max_prob
            
        is_abnormal = max_prob > OPTIMAL_THRESHOLD
        
        # --- UI DISPLAY FOR THIS SPECIFIC FILE ---
        # Use an expander so the UI doesn't get cluttered if they upload 4+ files
        with st.expander(f"🩺 Analysis: {valve_name} Valve ({filename})", expanded=is_abnormal):
            col1, col2 = st.columns([1, 2]) # Make the chart column slightly wider
            
            with col1:
                if is_abnormal:
                    st.error(f"⚠️ **Alert:** Murmur detected at {valve_name} valve.")
                else:
                    st.success(f"✅ **Clear:** No murmur detected here.")
                    
                st.metric(label="Peak Window Confidence", value=f"{max_prob*100:.1f}%")
                
            with col2:
                # Use tabs to cleanly separate the Waveform and the Spectrogram
                tab1, tab2 = st.tabs(["Raw Waveform", "AI Spectrogram (CWT)"])
                
                with tab1:
                    fig_wave, ax_wave = plt.subplots(figsize=(6, 2))
                    ax_wave.plot(y, color='#1f77b4', linewidth=0.5)
                    ax_wave.set_axis_off()
                    st.pyplot(fig_wave)
                    plt.close(fig_wave) # Prevent memory leaks!
                    
                with tab2:
                    suspicious_cwt = X_audio[max_prob_index][:, :, 0]
                    fig_cwt, ax_cwt = plt.subplots(figsize=(6, 2))
                    cax = ax_cwt.imshow(suspicious_cwt, aspect='auto', cmap='jet', origin='lower')
                    ax_cwt.set_axis_off()
                    st.pyplot(fig_cwt)
                    plt.close(fig_cwt)

    # --- FINAL PATIENT VERDICT ---
    st.divider()
    if patient_max_prob > OPTIMAL_THRESHOLD:
        st.error(f"### 🚨 OVERALL DIAGNOSIS: Refer to Cardiologist")
        st.write(f"The model found strong evidence of a malignant murmur across the uploaded recordings. (Peak Confidence: {patient_max_prob*100:.1f}%)")
    else:
        st.success(f"### 🟢 OVERALL DIAGNOSIS: Normal")
        st.write("The model did not find sufficient evidence of a malignant murmur in any of the uploaded recordings.")