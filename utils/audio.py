from gtts import gTTS
import io
import streamlit as st

@st.cache_data(show_spinner=False)
def generate_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes
