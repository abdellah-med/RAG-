import streamlit as st
from streamlit import config
import threading
import time
import queue
import os
from datetime import datetime

# Désactiver le file watcher (identique à interface.py)
config.set_option("server.fileWatcherType", "none")

# Imports des modules exactement comme dans interface.py
from indexall_minilm import (
    connect_to_qdrant,
    create_collection,
    index_all_pdfs,
    get_similar_documents
)
from query_gen import generate_query
from agnooo import retrieve_and_ask
from should_ask import evaluer_recommandation

# Imports pour l'audio
import tempfile
import wave
import pyaudio
import numpy as np
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LiveMedicalAssistant:
    def __init__(self):
        # Configuration API
        self.groq_client = Groq()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configuration audio
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5  # 5 secondes par segment
        
        # Variables de contrôle
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        
        # Données partagées
        self.raw_conversation = ""
        self.structured_conversation = ""
        self.latest_suggestion = ""
        self.conversation_lock = threading.Lock()
        
        # Queues pour communication
        self.update_queue = queue.Queue()
        
    def start_recording(self):
        """Démarre l'enregistrement audio"""
        self.is_recording = True
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Thread d'enregistrement
        recording_thread = threading.Thread(target=self._recording_loop)
        recording_thread.daemon = True
        recording_thread.start()
        
    def _recording_loop(self):
        """Boucle d'enregistrement principale"""
        while self.is_recording:
            audio_file = self._record_segment()
            if audio_file:
                self._process_audio(audio_file)
                
    def _record_segment(self):
        """Enregistre un segment de 5 secondes"""
        frames = []
        
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            if not self.is_recording:
                break
                
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"Erreur d'enregistrement: {e}")
                break
                
        if frames:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            return temp_file.name
        return None
        
    def _process_audio(self, audio_file_path):
        """Traite l'audio: transcription + structuration + analyse"""
        try:
            # 1. Transcription avec Groq
            with open(audio_file_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_file_path, file.read()),
                    model="whisper-large-v3-turbo",
                    language="fr"
                )
            
            text = transcription.text.strip()
            
            if text and len(text) > 1:
                with self.conversation_lock:
                    timestamp = time.strftime("%H:%M:%S")
                    self.raw_conversation += f"[{timestamp}] {text}\n"
                
                # 2. Structuration avec Gemini
                self._structure_conversation()
                
                # 3. Analyse automatique - EXACTEMENT comme interface.py
                try:
                    self._analyze_conversation()
                except Exception as e:
                    print(f"Erreur analyse: {e}")
                    self.latest_suggestion = f"Erreur d'analyse: {str(e)}"
                
                # 4. Notifier l'interface
                self.update_queue.put({
                    'type': 'update',
                    'raw': self.raw_conversation,
                    'structured': self.structured_conversation,
                    'suggestion': self.latest_suggestion,
                    'timestamp': timestamp
                })
                
        except Exception as e:
            print(f"Erreur de traitement: {e}")
        finally:
            try:
                os.unlink(audio_file_path)
            except:
                pass
                
    def _structure_conversation(self):
        """Structure la conversation avec Gemini"""
        prompt = f"""Analyse cette transcription d'une consultation entre un médecin allergologue respiratoire et un patient.

Identifie qui parle et structure la conversation. Réponds UNIQUEMENT par la conversation structurée au format suivant :

MÉDECIN: [texte]
PATIENT: [texte]
MÉDECIN: [texte]
PATIENT: [texte]

Transcription brute:
{self.raw_conversation}

IMPORTANT: Ne réponds que par la conversation structurée, rien d'autre."""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=8192,
                )
            )
            self.structured_conversation = response.text.strip() if response.text else ""
        except Exception as e:
            print(f"Erreur Gemini: {e}")
            
    def _analyze_conversation(self):
        """Analyse la conversation - EXACTEMENT comme interface.py"""
        try:
            # Vérifier que les objets Qdrant sont disponibles
            if 'qdrant_client' not in st.session_state or 'collection_name' not in st.session_state:
                self.latest_suggestion = "⏳ Initialisation du système en cours..."
                return
            
            # 1. Générer la query
            query = generate_query(self.structured_conversation)
            
            # 2. Logigramme médical (EXACTEMENT comme interface.py)
            logigramme = """
            **Logigramme pour le diagnostic des allergies respiratoires**
            1. **L'interrogatoire**
               1.1. **Chronologie des symptômes** :
               - Question 1 : Date de début des symptômes ?
               - Question 2 : Périodes de répit depuis le début ?
               - Question 3 : Présence d'une saisonnalité ?
               
               1.2. **Nature des symptômes** :
               ✓ **Nez** (rechercher) :
               - Obstruction nasale
               - Écoulement
               - Prurit/Éternuements
               - Respiration buccale
               - Renflements
               ✓ **Œil** (rechercher) :
               - Rougeur/Larmoiement
               - Prurit/Sensation de brûlure
               ✓ **Larynx** :
               - Prurit laryngé/Raclement de gorge ?
               ✓ **Poumons** :
               - Gêne respiratoire (repos/effort/fou rire) ?
               - Respiration sifflante ?
               - Toux (diurne/nocturne) ?
               ✓ **Autres** :
               - Reflux gastro-œsophagien ?
               - Antécédents de traitement :
                 * Antihistaminiques (efficacité ?)
                 * Ventoline (efficacité ?)
               1.3. **Environnement** :
               ✓ **Logement** :
               - Humidité/Type de sol
               - Animaux domestiques
               - Exposition tabagique
               - État de la literie
               ✓ **Profession** :
               - Exposition professionnelle (ex : boulanger, coiffeur...)
            """
            
            # 3. Évaluer la qualité
            evaluation_result = evaluer_recommandation(
                self.structured_conversation, 
                f"Logigramme: {logigramme}\nRequête générée: {query}"
            )
            
            if evaluation_result == "oui":
                # 4. Recherche documents
                client = st.session_state.qdrant_client
                collection_name = st.session_state.collection_name
                num_results = 5
                threshold = 0.70
                
                top_docs = get_similar_documents(client, collection_name, query, num_results)
                filtered_docs = [doc for doc in top_docs if doc['score'] > threshold]
                
                # 5. Générer suggestion
                question = (
                    "Propose une seule question pertinente à poser selon les informations de la discussion, "
                    "comme si elle était posée par le médecin. Explique brièvement pourquoi cette question est importante."
                )
                
                if filtered_docs:
                    response = retrieve_and_ask(filtered_docs, question, self.structured_conversation)
                else:
                    response = retrieve_and_ask(top_docs[:2], question, self.structured_conversation)
                
                self.latest_suggestion = response
                
            else:
                self.latest_suggestion = """⚠️ Qualité de discussion insuffisante

La discussion ne contient pas suffisamment d'informations précises sur les symptômes respiratoires. 
Nous vous recommandons de recueillir plus d'informations sur :

• La nature exacte des symptômes respiratoires (toux, essoufflement, sifflements, etc.)
• La contextualisation temporelle des crises/symptômes  
• Des données quantitatives sur la fréquence, durée et intensité des symptômes
• Des réponses plus spécifiques aux questions posées

Veuillez enrichir la discussion et réessayer."""
                
        except Exception as e:
            self.latest_suggestion = f"❌ Erreur d'analyse: {str(e)}"
    
    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
    def get_updates(self):
        """Récupère les mises à jour depuis la queue"""
        updates = []
        try:
            while True:
                update = self.update_queue.get_nowait()
                updates.append(update)
        except queue.Empty:
            pass
        return updates

# Configuration de la page
st.set_page_config(
    page_title="🎤 Assistant Médical Live IA",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Ultra-moderne (inchangé car il est bien fait)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #e8f4ff;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .control-panel {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .recording-status {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .recording-active {
        background: linear-gradient(135deg, #ff6b6b, #ff8e53);
        color: white;
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.5);
    }
    
    .recording-inactive {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .conversation-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        height: 400px;
        overflow-y: auto;
    }
    
    .conversation-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .speaker-medecin {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .speaker-patient {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
    }
    
    .suggestion-card {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #4ecdc4;
    }
    
    .suggestion-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .suggestion-content {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #4a5568;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 0.5rem;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .action-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .stop-button {
        background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    }
    
    .stop-button:hover {
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #a0aec0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# CORRECTION: Initialisation des composants EXACTEMENT comme dans interface.py
@st.cache_resource
def initialize_qdrant():
    # Initialisation de la connexion Qdrant
    collection_name = "corpus_a"
    client = connect_to_qdrant()
    should_index = create_collection(client, collection_name, 384)  # Taille des embeddings MiniLM-L6
    if should_index:
        with st.spinner("Indexation des documents PDF en cours..."):
            index_all_pdfs(client, collection_name, "ALLERG_IA")
    return client, collection_name

@st.cache_resource
def initialize_assistant():
    return LiveMedicalAssistant()

# Vérification des clés API
if not os.getenv("GROQ_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    st.error("⚠️ Veuillez configurer vos clés API GROQ_API_KEY et GEMINI_API_KEY")
    st.stop()

# CORRECTION: Initialisation dans le bon ordre
client, collection_name = initialize_qdrant()

# STOCKER IMMÉDIATEMENT dans session_state
st.session_state.qdrant_client = client
st.session_state.collection_name = collection_name

# Puis initialiser l'assistant APRÈS
if 'assistant' not in st.session_state:
    st.session_state.assistant = initialize_assistant()

if 'recording' not in st.session_state:
    st.session_state.recording = False

if 'conversation_data' not in st.session_state:
    st.session_state.conversation_data = {
        'structured': '',
        'suggestion': '',
        'word_count': 0,
        'last_update': ''
    }

# Interface principale (inchangée)
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# En-tête héroïque
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🎤 Assistant Médical Live IA</h1>
    <p class="hero-subtitle">Analyse en temps réel des consultations d'allergologie respiratoire</p>
</div>
""", unsafe_allow_html=True)

# Panneau de contrôle
st.markdown('<div class="control-panel">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    if not st.session_state.recording:
        if st.button("🎤 DÉMARRER L'ÉCOUTE", key="start_btn"):
            st.session_state.recording = True
            st.session_state.assistant.start_recording()
            st.rerun()
    else:
        if st.button("⏹️ ARRÊTER L'ÉCOUTE", key="stop_btn"):
            st.session_state.recording = False
            st.session_state.assistant.stop_recording()
            st.rerun()

with col2:
    # Statistiques en temps réel
    word_count = len(st.session_state.conversation_data['structured'].split())
    st.metric("Mots capturés", word_count)

st.markdown('</div>', unsafe_allow_html=True)

# Status de l'enregistrement
if st.session_state.recording:
    st.markdown("""
    <div class="recording-status recording-active">
        🔴 ENREGISTREMENT EN COURS - L'IA écoute et analyse en temps réel
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="recording-status recording-inactive">
        ⚫ PRÊT À DÉMARRER - Cliquez sur "Démarrer l'écoute" pour commencer
    </div>
    """, unsafe_allow_html=True)

# Colonnes principales pour conversation et suggestion
col1, col2 = st.columns([1, 1])

with col1:
    # Conversation en temps réel
    st.markdown("""
    <div class="conversation-card">
        <div class="conversation-title">💬 Conversation Live</div>
        <div id="conversation-content">
    """, unsafe_allow_html=True)
    
    conversation_placeholder = st.empty()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

with col2:
    # Suggestion IA
    st.markdown("""
    <div class="suggestion-card">
        <div class="suggestion-title">🤖 Suggestion IA</div>
        <div class="suggestion-content" id="suggestion-content">
    """, unsafe_allow_html=True)
    
    suggestion_placeholder = st.empty()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# Boucle de mise à jour en temps réel
if st.session_state.recording:
    # Récupérer les mises à jour
    updates = st.session_state.assistant.get_updates()
    
    for update in updates:
        if update['type'] == 'update':
            st.session_state.conversation_data = {
                'structured': update['structured'],
                'suggestion': update['suggestion'],
                'word_count': len(update['structured'].split()),
                'last_update': update['timestamp']
            }
    
    # Afficher la conversation structurée
    if st.session_state.conversation_data['structured']:
        conversation_html = ""
        lines = st.session_state.conversation_data['structured'].split('\n')
        
        for line in lines:
            if line.strip():
                if line.startswith('MÉDECIN:'):
                    conversation_html += f'<div class="speaker-medecin">👨‍⚕️ {line}</div>'
                elif line.startswith('PATIENT:'):
                    conversation_html += f'<div class="speaker-patient">🧑‍🤝‍🧑 {line}</div>'
                else:
                    conversation_html += f'<div style="margin: 0.5rem 0; color: #718096;">{line}</div>'
        
        if st.session_state.conversation_data['last_update']:
            conversation_html += f'<div class="timestamp">Dernière mise à jour: {st.session_state.conversation_data["last_update"]}</div>'
        
        conversation_placeholder.markdown(conversation_html, unsafe_allow_html=True)
    else:
        conversation_placeholder.markdown("🎤 En attente de la première transcription...", unsafe_allow_html=True)
    
    # Afficher la suggestion
    if st.session_state.conversation_data['suggestion']:
        suggestion_placeholder.markdown(st.session_state.conversation_data['suggestion'], unsafe_allow_html=True)
    else:
        suggestion_placeholder.markdown("🤔 L'IA analyse la conversation pour générer une suggestion...", unsafe_allow_html=True)
    
    # Auto-refresh toutes les 2 secondes
    time.sleep(2)
    st.rerun()

else:
    # Mode arrêté - afficher les données finales s'il y en a
    if st.session_state.conversation_data['structured']:
        # Afficher la conversation finale
        conversation_html = ""
        lines = st.session_state.conversation_data['structured'].split('\n')
        
        for line in lines:
            if line.strip():
                if line.startswith('MÉDECIN:'):
                    conversation_html += f'<div class="speaker-medecin">👨‍⚕️ {line}</div>'
                elif line.startswith('PATIENT:'):
                    conversation_html += f'<div class="speaker-patient">🧑‍🤝‍🧑 {line}</div>'
                else:
                    conversation_html += f'<div style="margin: 0.5rem 0; color: #718096;">{line}</div>'
        
        conversation_html += '<div style="text-align: center; margin-top: 2rem; color: #667eea; font-weight: 600;">📋 Consultation terminée</div>'
        conversation_placeholder.markdown(conversation_html, unsafe_allow_html=True)
        
        # Afficher la suggestion finale
        if st.session_state.conversation_data['suggestion']:
            suggestion_placeholder.markdown(st.session_state.conversation_data['suggestion'], unsafe_allow_html=True)
    else:
        conversation_placeholder.markdown("💬 Aucune conversation enregistrée. Démarrez l'écoute pour commencer.", unsafe_allow_html=True)
        suggestion_placeholder.markdown("🤖 Les suggestions apparaîtront après le début de l'enregistrement.", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Statistiques finales
if not st.session_state.recording and st.session_state.conversation_data['structured']:
    st.markdown("---")
    st.subheader("📊 Statistiques de la consultation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de mots", st.session_state.conversation_data['word_count'])
    
    with col2:
        medecin_lines = len([line for line in st.session_state.conversation_data['structured'].split('\n') if line.startswith('MÉDECIN:')])
        st.metric("Interventions médecin", medecin_lines)
    
    with col3:
        patient_lines = len([line for line in st.session_state.conversation_data['structured'].split('\n') if line.startswith('PATIENT:')])
        st.metric("Interventions patient", patient_lines)
    
    with col4:
        if st.session_state.conversation_data['last_update']:
            st.metric("Dernière activité", st.session_state.conversation_data['last_update'])
    
    # Option de sauvegarde
    if st.button("💾 Sauvegarder la consultation", use_container_width=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consultation_live_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("CONSULTATION MÉDICALE LIVE\n")
            f.write("="*50 + "\n\n")
            f.write("Conversation structurée:\n")
            f.write(st.session_state.conversation_data['structured'])
            f.write("\n\n" + "="*50 + "\n")
            f.write("Suggestion IA finale:\n")
            f.write(st.session_state.conversation_data['suggestion'])
        
        st.success(f"✅ Consultation sauvegardée dans: {filename}")