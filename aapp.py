import io
import av
import streamlit as st
import numpy as np 
from streamlit_mic_recorder import mic_recorder
from scipy.fft import rfft, rfftfreq

# 1. Decode Audio Function (Fixed Typo & Buffer Handling)
def decode_audio(audio_bytes):
    container = av.open(io.BytesIO(audio_bytes))
    stream = container.streams.audio[0]
    sr = stream.rate
    
    frames = [frame.to_ndarray() for frame in container.decode(stream)]
    if not frames:
        return sr, np.array([], dtype=np.int16)
        
    audio_data = np.concatenate(frames, axis=1)

    # Stereo ko Mono mein convert karna (Typo fixed here: sudio_data -> audio_data)
    if audio_data.shape[0] > 1:
        audio_data = np.mean(audio_data, axis=0)
    else:
        audio_data = audio_data[0]

    # Scaling float values to Int16
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
    return sr, audio_data


# 2. Speech Parameters Calculation
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
        "Loudness Fluctuation": round(float(shimmer), 4)
    }


# 3. Streamlit Interface
st.title("🎤 Voice Recorder & Diagnostic Parameters")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="recorder"
)

if audio and audio.get('bytes'):
    # Player format fixed to webm (jo recorder send karta hai)
    st.audio(audio['bytes'], format='audio/webm')
    
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
        col5.metric("Loudness Fluctuation (Shimmer)", params['Loudness Fluctuation'])
