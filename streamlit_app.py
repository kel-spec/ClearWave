import io
import wave
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

st.set_page_config(page_title="ClearWave", layout="centered")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert PCM audio to float32 in [-1, 1]."""
    if np.issubdtype(audio.dtype, np.integer):
        max_val = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / max_val
    return audio.astype(np.float32)


def safe_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize only once at the end."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * target_peak
    return audio


def speech_band_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Gentle speech-focused bandpass filter.
    Keeps most voice information without sounding too thin.
    """
    nyquist = sample_rate / 2.0

    lowcut = 80.0
    highcut = min(7000.0, nyquist * 0.95)

    if highcut <= lowcut:
        return audio

    sos = butter(
        N=4,
        Wn=[lowcut / nyquist, highcut / nyquist],
        btype="bandpass",
        output="sos"
    )
    return sosfiltfilt(sos, audio)


def simple_denoise(audio: np.ndarray) -> np.ndarray:
    """
    Very mild denoise.
    Removes DC offset and applies a tiny smoothing effect only.
    """
    audio = audio - np.mean(audio)

    # Tiny smoothing to reduce harshness, without chopping
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    padded = np.pad(audio, (1, 1), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    # Blend original and smoothed to preserve clarity
    return 0.85 * audio + 0.15 * smoothed


def process_audio(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Main enhancement pipeline."""
    audio = to_mono(audio)
    audio = pcm_to_float32(audio)

    # Prevent clipping if source is already hot
    audio = np.clip(audio, -1.0, 1.0)

    # Gentle filtering only
    audio = speech_band_filter(audio, sample_rate)

    # Mild cleanup only
    audio = simple_denoise(audio)

    # Final normalize once
    audio = safe_normalize(audio, target_peak=0.95)

    return audio.astype(np.float32)


def float_to_wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    """Convert float audio back to 16-bit WAV bytes."""
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_buffer:
        wav_buffer.setnchannels(1)
        wav_buffer.setsampwidth(2)
        wav_buffer.setframerate(sample_rate)
        wav_buffer.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


st.title("ClearWave")
st.write("Upload a WAV file and get the processed output immediately.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    try:
        sample_rate, audio_data = wavfile.read(uploaded_file)

        st.subheader("Original Audio")
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Processing audio..."):
            processed_audio = process_audio(sample_rate, audio_data)
            processed_wav = float_to_wav_bytes(sample_rate, processed_audio)

        st.subheader("Processed Audio")
        st.audio(processed_wav, format="audio/wav")

        st.download_button(
            label="Download Processed WAV",
            data=processed_wav,
            file_name="clearwave_output.wav",
            mime="audio/wav",
        )

        st.success("Processing complete.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a .wav file to begin.")
