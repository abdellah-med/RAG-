import streamlit as st
from groq_whisper_live import GroqWhisperLiveTranscriber
import threading
import time

# Initialiser le transcripteur
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = GroqWhisperLiveTranscriber()

# État global pour l'enregistrement
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

st.title("🩺 Transcription Médicale en Temps Réel")
st.markdown("**Spécialité : Allergologie Respiratoire**")

# Affichage des boutons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Démarrer l'écoute", disabled=st.session_state.is_recording):
        st.session_state.is_recording = True
        st.session_state.transcriber.is_recording = True
        st.session_state.recording_thread = threading.Thread(
            target=st.session_state.transcriber.start_recording,
            daemon=True
        )
        st.session_state.recording_thread.start()

with col2:
    if st.button("⏹️ Arrêter", disabled=not st.session_state.is_recording):
        st.session_state.transcriber.stop_recording()
        st.session_state.is_recording = False
        # Petite pause pour laisser le temps à l'arrêt de se propager
        time.sleep(0.5)
        st.rerun()

# Afficher la transcription brute en direct
st.subheader("📝 Transcription brute (live)")
if hasattr(st.session_state.transcriber, 'full_conversation'):
    st.text_area("Texte en cours", 
                st.session_state.transcriber.full_conversation, 
                height=300,
                key="live_transcription")

# Générer manuellement un rapport structuré
if st.button("📄 Générer rapport final", 
            disabled=not st.session_state.is_recording and 
                     (not hasattr(st.session_state.transcriber, 'full_conversation') or 
                      not st.session_state.transcriber.full_conversation.strip())):
    with st.spinner("Analyse en cours avec Gemini..."):
        st.session_state.transcriber.generate_final_report()
        st.rerun()