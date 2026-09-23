import io
import av
import streamlit as st
import numpy as np 
import pandas as pd
from streamlit_mic_recorder import mic_recorder
from scipy.fft import rfft, rfftfreq
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="Voice Diagnostic Tool", page_icon="🎤")

# 1. Google Sheets Connection Initialize
conn = st.connection("gsheets", type=GSheetsConnection)

# 2. Audio Decoding Function
def decode_audio(audio_bytes):
    container = av.open(io.BytesIO(audio_bytes))
    stream = container.streams.audio[0]
    sr = stream.rate
    frames = [frame.to_ndarray() for frame in container.decode(stream)]
    audio_data = np.concatenate(frames, axis=1)

    if audio_data.shape[0] > 1:
        audio_data = np.mean(audio_data, axis=0) # Fixed Typo
    else:
        audio_data = audio_data[0]

    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    return sr, audio_data

# 3. Speech Diagnostics Extraction
def speech_parameters(audio_data, sr):
    if len(audio_data) == 0:
        return {}

    energy = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2.0

    n = len(audio_data)
    fft_spectrum = np.abs(rfft(audio_data))
    freqs = rfftfreq(n, 1.0 / sr)
    fft_spectrum[0] = 0
    pitch_hz = freqs[np.argmax(fft_spectrum)]

    frame_len = int(sr * 0.02)
    frames = [audio_data[i : i + frame_len] for i in range(0, len(audio_data) - frame_len, frame_len)]

    frame_energies = [np.sqrt(np.mean(f.astype(np.float64)**2)) for f in frames if len(f) > 0]
    shimmer = (np.mean(np.abs(np.diff(frame_energies))) / np.mean(frame_energies)) if len(frame_energies) > 1 and np.mean(frame_energies) > 0 else 0.0

    frame_pitches = []
    for f in frames:
        if len(f) > 0:
            f_spec = np.abs(rfft(f))
            f_freqs = rfftfreq(len(f), 1.0 / sr)
            f_spec[0] = 0
            frame_pitches.append(f_freqs[np.argmax(f_spec)])
            
    jitter = (np.mean(np.abs(np.diff(frame_pitches))) / np.mean(frame_pitches)) if len(frame_pitches) > 1 and np.mean(frame_pitches) > 0 else 0.0

    return {
        "pitch_hz": round(float(pitch_hz), 1),
        "energy": round(float(energy), 1),
        "zcr": round(float(zcr), 4),
        "instability": round(float(jitter), 4),
        "shimmer": round(float(shimmer), 4)
    }

# Streamlit Interface
st.title("🎤 Voice Recorder & Diagnostic Parameters")

# Fetch existing Google Sheet data
try:
    existing_df = conn.read(ttl=0) # ttl=0 ensures fresh fetch
    tc_number = len(existing_df) + 1
except Exception:
    existing_df = pd.DataFrame()
    tc_number = 1

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="recorder"
)

if audio and audio.get('bytes'):
    st.audio(audio['bytes'], format='audio/wav')
    
    sr, audio_data = decode_audio(audio['bytes'])
    params = speech_parameters(audio_data, sr)
    
    if params:
        st.subheader("📊 Speech Diagnostics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pitch", f"{params['pitch_hz']} Hz")
        col2.metric("Energy (Volume)", params['energy'])
        col3.metric("Noise (ZCR)", params['zcr'])

        col4, col5 = st.columns(2)
        col4.metric("Instability (Jitter)", params['instability'])
        col5.metric("Loudness Fluctuation (Shimmer)", params['shimmer'])
        
        st.divider()
        
        current_tc = f"TC_{tc_number:02d}"
        st.info(f"📌 Next Test Case ID: **{current_tc}**")
        
        context = st.text_input("📝 Recording Context / Description", value="Normal Speaking")
        
        if st.button("🚀 Save to Google Sheet"):
            new_row = pd.DataFrame([{
                "Test Case": current_tc,
                "Pitch (Hz)": params['pitch_hz'],
                "Energy (Volume)": params['energy'],
                "Noise (ZCR)": params['zcr'],
                "Instability (Jitter)": params['instability'],
                "Loudness Fluctuation (Shimmer)": params['shimmer'],
                "Recording Context": context
            }])
            
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            conn.update(data=updated_df)
            st.success(f"✅ {current_tc} saved on google spread sheets !")
            st.rerun()

# Sidebar Data Preview
st.sidebar.subheader("📁 Live Google Sheet Data")
if not existing_df.empty:
    st.sidebar.dataframe(existing_df)
else:
    st.sidebar.info("data is loading ...")
