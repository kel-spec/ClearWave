import io
import wave
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy import signal

st.set_page_config(page_title="ClearWave", layout="centered")


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo or multi-channel audio to mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def normalize_input(audio: np.ndarray) -> np.ndarray:
    """Convert input audio to float32 safely in range [-1, 1]."""
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def design_fir_filters(fs: int, order: int = 200):
    """
    Create three FIR filters:
    - low band
    - mid band
    - high band
    """
    nyq = fs / 2

    low_cutoff = 300 / nyq
    mid_cutoff = [300 / nyq, 3000 / nyq]
    high_cutoff = 3000 / nyq

    low_filter = signal.firwin(order + 1, low_cutoff, pass_zero="lowpass")
    mid_filter = signal.firwin(order + 1, mid_cutoff, pass_zero="bandpass")
    high_filter = signal.firwin(order + 1, high_cutoff, pass_zero="highpass")

    return low_filter, mid_filter, high_filter


def apply_equalizer(audio: np.ndarray, fs: int,
                    low_gain: float = 0.8,
                    mid_gain: float = 1.5,
                    high_gain: float = 1.1,
                    order: int = 200) -> np.ndarray:
    """
    Apply FIR-based equalizer similar to the Colab setup:
    split into low, mid, high, then recombine using gains.
    """
    low_filt, mid_filt, high_filt = design_fir_filters(fs, order=order)

    low_band = signal.lfilter(low_filt, 1.0, audio)
    mid_band = signal.lfilter(mid_filt, 1.0, audio)
    high_band = signal.lfilter(high_filt, 1.0, audio)

    enhanced = (
        low_gain * low_band +
        mid_gain * mid_band +
        high_gain * high_band
    )

    return enhanced


def safe_output_scaling(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """
    Scale only once at the end.
    Prevent output from becoming too loud.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * target_peak
    return np.clip(audio, -1.0, 1.0)


def process_audio(audio: np.ndarray, fs: int,
                  low_gain: float,
                  mid_gain: float,
                  high_gain: float,
                  order: int = 200) -> np.ndarray:
    """Full ClearWave processing pipeline."""
    mono = convert_to_mono(audio)
    normalized = normalize_input(mono)

    enhanced = apply_equalizer(
        normalized,
        fs,
        low_gain=low_gain,
        mid_gain=mid_gain,
        high_gain=high_gain,
        order=order
    )

    output = safe_output_scaling(enhanced, target_peak=0.9)
    return output.astype(np.float32)


def wav_bytes_from_float(audio: np.ndarray, fs: int) -> bytes:
    """Convert float audio [-1,1] to WAV bytes."""
    int_audio = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(int_audio.tobytes())

    buffer.seek(0)
    return buffer.read()


st.title("ClearWave")
st.write("Upload a `.wav` file and process it using the FIR equalizer pipeline.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

st.subheader("Enhancement Settings")
low_gain = st.slider("Low Frequency Gain", 0.0, 2.0, 0.8, 0.1)
mid_gain = st.slider("Mid Frequency Gain", 0.0, 3.0, 1.5, 0.1)
high_gain = st.slider("High Frequency Gain", 0.0, 2.0, 1.1, 0.1)

if uploaded_file is not None:
    try:
        fs, audio = wavfile.read(uploaded_file)

        st.subheader("Original Audio")
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Processing audio..."):
            processed = process_audio(
                audio,
                fs,
                low_gain=low_gain,
                mid_gain=mid_gain,
                high_gain=high_gain,
                order=200
            )
            processed_wav = wav_bytes_from_float(processed, fs)

        st.subheader("Processed Audio")
        st.audio(processed_wav, format="audio/wav")

        st.download_button(
            label="Download Processed WAV",
            data=processed_wav,
            file_name="clearwave_output.wav",
            mime="audio/wav"
        )

        st.success("Done.")

    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info("Upload a WAV file to begin.")
