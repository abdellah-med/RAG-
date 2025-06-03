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

# Affichage transcription brute
if hasattr(st.session_state.transcriber, 'full_conversation'):
    st.text_area("Texte en cours", 
                 st.session_state.transcriber.full_conversation, 
                 height=300,
                 key="live_transcription")
    
    # Détecter les questions manquantes en direct (seulement si la conversation a du contenu)
    if st.session_state.transcriber.full_conversation.strip():
        # Pour éviter de faire la détection trop souvent, on peut la faire toutes les x secondes ou à chaque update
        try:
            st.session_state.suggestions_questions = st.session_state.transcriber.detecter_questions_manquantes(
                st.session_state.transcriber.full_conversation
            )
        except Exception as e:
            st.error(f"Erreur détection questions: {e}")

# Afficher les questions manquantes si détectées
if 'suggestions_questions' in st.session_state and st.session_state.suggestions_questions:
    st.subheader("💡 Questions supplémentaires à poser")
    for q in st.session_state.suggestions_questions:
        st.markdown(f"- {q}")
else:
    st.subheader("💡 Questions supplémentaires à poser")
    st.markdown("*Aucune question complémentaire détectée.*")



# Générer manuellement un rapport structuré
if st.button("📄 Générer rapport final", 
            disabled=not st.session_state.is_recording and 
                     (not hasattr(st.session_state.transcriber, 'full_conversation') or 
                      not st.session_state.transcriber.full_conversation.strip())):
    with st.spinner("Analyse en cours avec Gemini..."):
        st.session_state.transcriber.generate_final_report()
        st.rerun()