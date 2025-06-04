import streamlit as st
from groq_whisper_live import GroqWhisperLiveTranscriber
import threading
import time

from indexall_minilm import (
    connect_to_qdrant,
    create_collection,
    index_all_pdfs,
    get_similar_documents
)

# Titre de l'application
st.set_page_config(page_title="Assistant Médical", layout="centered")

# Fonction pour afficher un spinner pendant le chargement
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

# Sidebar pour la configuration et les informations
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Paramètres de recherche
    st.subheader("Paramètres")
    threshold = st.slider(
        "Seuil de pertinence (score minimum)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Les documents avec un score inférieur à ce seuil seront ignorés"
    )
    
    num_results = st.slider(
        "Nombre de résultats à afficher",
        min_value=1,
        max_value=10,
        value=5,
        help="Nombre maximum de documents similaires à rechercher"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.9rem">
    <b>À propos</b><br>
    Cette application utilise l'IA pour analyser des discussions médicales et suggérer des questions pertinentes basées sur une base documentaire.
    </div>
    """, unsafe_allow_html=True)


# Initialisation de Qdrant (avec cache pour éviter la réindexation)
client, collection_name = initialize_qdrant()

# En-tête principal
st.markdown('<h1 class="main-header">🩺 Assistant IA pour discussions Médicales</h1>', unsafe_allow_html=True)
    
# Initialiser le transcripteur une seule fois
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = GroqWhisperLiveTranscriber()

# État global de l'enregistrement
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Affichage des boutons Démarrer/Arrêter
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
        time.sleep(0.5)
        st.rerun()

# Affichage de la transcription en temps réel
# st.markdown("### 📝 Transcription en direct :")
if st.session_state.is_recording:
    st.info("🎤 Enregistrement en cours... Parlez maintenant.")
else:
    st.warning("⏹️ L'écoute est arrêtée.")

# Afficher les lignes transcrites (si disponibles)
# if hasattr(st.session_state.transcriber, "transcript"):
#     for line in st.session_state.transcriber.transcript[-10:]:  # Les 10 dernières lignes
#         st.markdown(f"- {line}")
# ... (après avoir obtenu la discussion, par exemple après arrêt de l'enregistrement) ...

if not st.session_state.is_recording and st.session_state.transcriber.full_conversation.strip():
    discussion_text = st.session_state.transcriber.full_conversation.strip()
    timings = {
        "start_time": time.time(),
        "query_generation_time": 0,
        "evaluation_time": 0,
        "document_retrieval_time": 0,
        "response_generation_time": 0,
        "total_time": 0
    }

    # Génération de la query
    query_start = time.time()
    with st.spinner("Génération de la requête..."):
        from query_gen import generate_query
        query = generate_query(discussion_text)
    timings["query_generation_time"] = time.time() - query_start

    # Évaluation de la discussion
    eval_start = time.time()
    from should_ask import evaluer_recommandation
    evaluation_result = evaluer_recommandation(discussion_text, f"Requête générée: {query}")
    timings["evaluation_time"] = time.time() - eval_start

    # Recherche de documents similaires
    docs_start = time.time()
    from indexall_minilm import get_similar_documents
    top_docs = get_similar_documents(client, collection_name, query, num_results)
    timings["document_retrieval_time"] = time.time() - docs_start

    # Filtrer les documents avec un score > seuil
    filtered_docs = [doc for doc in top_docs if doc['score'] > threshold]

    # Génération de la suggestion (retrieve_and_ask)
    response_start = time.time()
    from agnooo import retrieve_and_ask
    question = (
        "Propose une seule question pertinente à poser selon les informations de la discussion, "
        "comme si elle était posée par le médecin."
    )
    if filtered_docs:
        response = retrieve_and_ask(filtered_docs, question, discussion_text)
    else:
        response = retrieve_and_ask(top_docs[:2], question, discussion_text)
    timings["response_generation_time"] = time.time() - response_start

    timings["total_time"] = time.time() - timings["start_time"]

    # Affichage des résultats dans l'ordre souhaité

    # 1. Suggestion IA d'abord
    st.markdown("### 💡 Suggestion de l'IA :")
    st.markdown(response)

    # 2. Discussion structurée ensuite (extrait structuré si dispo)
    st.markdown("### 🤖 Discussion structurée (Gemini) :")
    import re
    if hasattr(st.session_state.transcriber, "structured_conversation") and st.session_state.transcriber.structured_conversation:
        text = st.session_state.transcriber.structured_conversation
        match = re.search(r"(=== DIALOGUE STRUCTURÉ ===.*?)(?:\n===|\Z)", text, re.DOTALL)
        if match:
            dialogue_struct = match.group(1).replace("\n", "  \n")
            st.markdown(dialogue_struct)
        else:
            st.markdown("Aucun dialogue structuré trouvé.")
    else:
        st.markdown(discussion_text.replace("\n", "  \n"))

    # (Optionnel) Afficher les timings ou autres infos

# ...le reste de ton code pour l'enregistrement en temps réel reste inchangé...