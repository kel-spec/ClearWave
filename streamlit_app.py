import io
import wave
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="ClearWave", page_icon="🎧", layout="centered")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo/multi-channel audio to mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 range [-1, 1]."""
    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def bandpass_filter(audio: np.ndarray, sample_rate: int,
                    lowcut: float = 300.0, highcut: float = 3400.0,
                    order: int = 4) -> np.ndarray:
    """Keep the main speech frequency band."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.999)

    if low <= 0 or high >= 1 or low >= high:
        return audio

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, audio)


def noise_gate(audio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Reduce very low-level background noise."""
    gated = np.copy(audio)
    gated[np.abs(gated) < threshold] = 0
    return gated


def pre_emphasis(audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Boost higher speech components slightly for clarity."""
    return np.append(audio[0], audio[1:] - alpha * audio[:-1])


def process_audio(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Simple speech enhancement pipeline."""
    audio = to_mono(audio)
    audio = normalize_audio(audio)
    audio = bandpass_filter(audio, sample_rate, lowcut=300, highcut=3400)
    audio = noise_gate(audio, threshold=0.02)
    audio = pre_emphasis(audio, alpha=0.95)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def float_to_wav_bytes(sample_rate: int, audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to WAV bytes."""
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_buffer:
        wav_buffer.setnchannels(1)
        wav_buffer.setsampwidth(2)  # 16-bit audio
        wav_buffer.setframerate(sample_rate)
        wav_buffer.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


st.title("ClearWave")
st.write("Upload a `.wav` file and get the processed output immediately.")

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
    st.info("Upload a `.wav` file to begin.")
