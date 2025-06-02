import streamlit as st
from streamlit import config
import sounddevice as sd
import wave
import os
import threading
import time
import numpy as np
from datetime import datetime

# Désactiver le file watcher pour éviter les conflits avec PyTorch
config.set_option("server.fileWatcherType", "none")

# Imports des modules personnalisés
try:
    from transcription import transcrire_audio
    from indexall_minilm import (
        connect_to_qdrant,
        create_collection,
        index_all_pdfs,
        get_similar_documents
    )
    from query_gen import generate_query
    from agnooo import retrieve_and_ask
    from should_ask import evaluer_recommandation
except ImportError as e:
    st.error(f"Erreur d'import: {e}")

import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA - discussions Médicales",
    page_icon="🩺",
    layout="wide"
)

# Classe pour l'enregistrement audio intégrée
class StreamlitAudioRecorder:
    def __init__(self, duree_tranche=5, frequence=44100, dossier_sortie="enregistrements"):
        self.duree_tranche = duree_tranche
        self.frequence = frequence
        self.dossier_sortie = dossier_sortie
        self.compteur = 1
        
        # Créer le dossier de sortie
        os.makedirs(self.dossier_sortie, exist_ok=True)
    
    def enregistrer_audio_continu(self):
        """Fonction d'enregistrement audio qui vérifie l'état Streamlit"""
        print(f"🎙️ Démarrage de l'enregistrement par tranches de {self.duree_tranche} secondes...")
        
        try:
            while st.session_state.get("recording", False):
                # Générer un nom de fichier avec timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nom_fichier = os.path.join(
                    self.dossier_sortie, 
                    f"enregistrement_{self.compteur:03d}_{timestamp}.wav"
                )
                
                print(f"🎧 Enregistrement du fichier : {nom_fichier}")
                
                try:
                    # Enregistrer par petits segments pour pouvoir vérifier l'état
                    segments_par_tranche = 10  # 10 segments de 0.5 sec pour une tranche de 5 sec
                    segment_duration = self.duree_tranche / segments_par_tranche
                    samples_per_segment = int(segment_duration * self.frequence)
                    
                    all_audio_data = []
                    
                    for segment in range(segments_par_tranche):
                        # Vérifier si on doit arrêter l'enregistrement
                        if not st.session_state.get("recording", False):
                            break
                        
                        # Enregistrer un petit segment
                        audio_segment = sd.rec(
                            samples_per_segment,
                            samplerate=self.frequence,
                            channels=1,
                            dtype='int16'
                        )
                        sd.wait()
                        all_audio_data.append(audio_segment)
                    
                    # Si on a des données audio, les sauvegarder
                    if all_audio_data and st.session_state.get("recording", False):
                        # Concaténer tous les segments
                        audio_data = np.concatenate(all_audio_data, axis=0)
                        self._sauvegarder_wav(nom_fichier, audio_data)
                        print(f"✅ Fichier sauvegardé : {nom_fichier}")
                        self.compteur += 1
                        
                        # Mettre à jour le statut dans Streamlit
                        st.session_state.last_recorded_file = nom_fichier
                        st.session_state.total_recorded_files = self.compteur - 1
                    
                except Exception as e:
                    print(f"Erreur lors de l'enregistrement du segment {self.compteur}: {e}")
                    continue
                
        except Exception as e:
            print(f"Erreur générale durant l'enregistrement : {e}")
        finally:
            print("🛑 Enregistrement arrêté proprement.")
            st.session_state.recording = False
    
    def _sauvegarder_wav(self, nom_fichier, audio_data):
        """Sauvegarder les données audio au format WAV"""
        with wave.open(nom_fichier, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(self.frequence)
            wf.writeframes(audio_data.tobytes())
    
    def get_all_recordings(self):
        """Retourner la liste de tous les enregistrements"""
        files = []
        if os.path.exists(self.dossier_sortie):
            for filename in os.listdir(self.dossier_sortie):
                if filename.endswith('.wav'):
                    filepath = os.path.join(self.dossier_sortie, filename)
                    files.append({
                        'name': filename,
                        'path': filepath,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                    })
        return sorted(files, key=lambda x: x['modified'], reverse=True)

# Fonctions pour contrôler l'enregistrement
def demarrer_enregistrement_streamlit():
    """Démarre l'enregistrement dans un thread compatible avec Streamlit"""
    if 'audio_recorder' not in st.session_state:
        duree = st.session_state.get('duree_tranche', 5)
        freq = st.session_state.get('frequence', 44100)
        st.session_state.audio_recorder = StreamlitAudioRecorder(duree_tranche=duree, frequence=freq)
    
    if not st.session_state.get("recording", False):
        st.session_state.recording = True
        st.session_state.total_recorded_files = 0
        
        # Démarrer l'enregistrement dans un thread
        thread = threading.Thread(target=st.session_state.audio_recorder.enregistrer_audio_continu)
        thread.daemon = True
        thread.start()
        return True
    return False

def arreter_enregistrement_streamlit():
    """Arrête l'enregistrement"""
    if st.session_state.get("recording", False):
        st.session_state.recording = False
        return True
    return False

# Application des styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0047ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3366ff;
    }
    .doc-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3366ff;
    }
    .recommendation-box {
        background-color: #eef5ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #3366ff;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #6c757d;
    }
    .stButton>button {
        background-color: #3366ff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0047ab;
    }
    .timing-box {
        background-color: #f0f8ff;
        border: 1px solid #3366ff;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .timing-header {
        font-weight: bold;
        color: #0047ab;
        margin-bottom: 0.5rem;
    }
    .timing-detail {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px dotted #dee2e6;
    }
    .timing-label {
        color: #495057;
    }
    .timing-value {
        font-family: monospace;
        color: #0047ab;
    }
    .timing-total {
        font-weight: bold;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 2px solid #dee2e6;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #856404;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #155724;
    }
    .recording-active {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recording-stopped {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour afficher un spinner pendant le chargement
@st.cache_resource
def initialize_qdrant():
    # Initialisation de la connexion Qdrant
    collection_name = "corpus_a"
    try:
        client = connect_to_qdrant()
        should_index = create_collection(client, collection_name, 384)  # Taille des embeddings MiniLM-L6
        if should_index:
            with st.spinner("Indexation des documents PDF en cours..."):
                index_all_pdfs(client, collection_name, "ALLERG_IA")
        return client, collection_name
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de Qdrant: {e}")
        return None, None

# Sidebar pour la configuration et les informations
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Paramètres de recherche
    st.subheader("Paramètres de recherche")
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
    
    # Paramètres d'enregistrement
    st.markdown("### 🎤 Paramètres d'enregistrement")
    
    duree_tranche = st.slider(
        "Durée par tranche (secondes)",
        min_value=1,
        max_value=30,
        value=5,
        help="Durée de chaque segment d'enregistrement"
    )
    st.session_state.duree_tranche = duree_tranche
    
    frequence = st.selectbox(
        "Fréquence d'échantillonnage",
        options=[16000, 22050, 44100, 48000],
        index=2,
        help="Qualité de l'enregistrement (44100 Hz recommandé)"
    )
    st.session_state.frequence = frequence
    
    # Bouton pour nettoyer tous les enregistrements
    if st.button("🧹 Nettoyer tous les enregistrements", type="secondary"):
        try:
            import shutil
            if os.path.exists("enregistrements"):
                shutil.rmtree("enregistrements")
                os.makedirs("enregistrements", exist_ok=True)
            st.session_state.total_recorded_files = 0
            st.session_state.last_recorded_file = None
            st.success("Tous les enregistrements ont été supprimés!")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors du nettoyage: {e}")
    
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

# Tabs pour organiser l'interface
tab1, tab2, tab3 = st.tabs(["Analyse de discussion", "Enregistrement Audio", "Documentation"])

with tab1:
    # Initialiser les variables de session pour le texte de discussion
    if "discussion_text" not in st.session_state:
        st.session_state.discussion_text = ""
    
    # Champ de texte pour la discussion
    discussion_text = st.text_area(
        "Entrez la discussion :", 
        value=st.session_state.discussion_text,
        height=250,
        key="discussion_input"
    )
    
    # Mettre à jour la session state
    st.session_state.discussion_text = discussion_text
    
    # Boutons d'action
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyser la discussion", use_container_width=True)
    with col2:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.discussion_text = ""
            st.rerun()
    
    # Analyse de la discussion
    if analyze_button and client and collection_name:
        if discussion_text.strip():
            # Initialiser les temps pour chaque étape
            timings = {
                "start_time": time.time(),
                "query_generation_time": 0,
                "evaluation_time": 0,
                "document_retrieval_time": 0,
                "response_generation_time": 0,
                "total_time": 0
            }
            
            # Créer un conteneur pour afficher les résultats
            results_container = st.container()
            
            with results_container:
                st.markdown('<div class="subheader">📊 Résultats de l\'analyse</div>', unsafe_allow_html=True)
                
                # Générer la query et mesurer le temps
                query_start = time.time()
                with st.spinner("Génération de la requête..."):
                    query = generate_query(discussion_text)
                timings["query_generation_time"] = time.time() - query_start
                
                # Toujours afficher la query générée
                with st.expander("🔎 Requête générée"):
                    st.info(query)
                
                # Définir le logigramme médical pour l'allergologie respiratoire
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
                
                # Évaluer la qualité de la discussion avec la query générée et le logigramme
                eval_start = time.time()
                with st.spinner("Évaluation de la qualité de la discussion..."):
                    # Passage de la query générée et du logigramme comme contexte
                    evaluation_result = evaluer_recommandation(discussion_text, f"Logigramme: {logigramme}\nRequête générée: {query}")
                timings["evaluation_time"] = time.time() - eval_start
                
                # Afficher le résultat de l'évaluation
                if evaluation_result == "oui":
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Discussion de qualité</h3>
                        <p>La discussion contient des informations pertinentes et détaillées sur les symptômes respiratoires. 
                        Nous allons procéder à l'analyse approfondie.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Récupérer les documents similaires et mesurer le temps
                    docs_start = time.time()
                    with st.spinner("Recherche de documents pertinents..."):
                        top_docs = get_similar_documents(client, collection_name, query, num_results)
                    timings["document_retrieval_time"] = time.time() - docs_start
                    
                    # Filtrer les documents avec un score > seuil
                    filtered_docs = [doc for doc in top_docs if doc['score'] > threshold]
                    
                    # Affichage des documents récupérés dans un tableau
                    if top_docs:
                        st.markdown('<div class="subheader">📚 Documents pertinents trouvés</div>', unsafe_allow_html=True)
                        
                        # Créer un DataFrame pour afficher les résultats de manière plus organisée
                        docs_df = pd.DataFrame([
                            {
                                "Fichier": doc['file_name'],
                                "Score": f"{doc['score']:.2f}",
                                "Pertinent": "✅" if doc['score'] > threshold else "❌"
                            } for doc in top_docs
                        ])
                        
                        st.dataframe(docs_df, use_container_width=True)
                        
                        # Afficher le contenu des documents dans des expanders
                        for i, doc in enumerate(top_docs):
                            with st.expander(f"Document {i+1}: {doc['file_name']} (Score: {doc['score']:.2f})"):
                                st.markdown(f"""
                                <div class="doc-card">
                                    <p><strong>Fichier:</strong> {doc['file_name']}</p>
                                    <p><strong>Chunk:</strong> {doc['chunk_number']}</p>
                                    <p><strong>Score:</strong> {doc['score']:.4f}</p>
                                    <hr>
                                    <p><strong>Contenu:</strong></p>
                                    <pre style="white-space: pre-wrap;">{doc['chunk_text']}</pre>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Aucun document similaire n'a été trouvé.")
                    
                    # Générer la réponse et mesurer le temps
                    response_start = time.time()
                    with st.spinner("Génération de la suggestion..."):
                        question = (
                            "Propose une seule question pertinente à poser selon les informations de la discussion, "
                            "comme si elle était posée par le médecin. Explique brièvement pourquoi cette question est importante."
                        )
                        
                        # S'assurer que filtered_docs contient des documents pertinents
                        if filtered_docs:
                            response = retrieve_and_ask(filtered_docs, question, discussion_text)
                        else:
                            # Si aucun document ne dépasse le seuil, utiliser quand même les meilleurs documents disponibles
                            st.warning("⚠️ Aucun document ne dépasse le seuil de pertinence. Utilisation des meilleurs documents disponibles.")
                            # Utiliser les 2 meilleurs documents même s'ils sont sous le seuil
                            response = retrieve_and_ask(top_docs[:2], question, discussion_text)
                    
                    timings["response_generation_time"] = time.time() - response_start
                    
                    # Afficher la réponse générée
                    st.markdown('<div class="subheader">💡 Suggestion de l\'IA</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="recommendation-box">{response}</div>', unsafe_allow_html=True)
                    
                else:  # Si evaluation_result == "non"
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ Qualité de discussion insuffisante</h3>
                        <p>La discussion ne contient pas suffisamment d'informations précises sur les symptômes respiratoires. 
                        Nous vous recommandons de recueillir plus d'informations sur :</p>
                        <ul>
                            <li>La nature exacte des symptômes respiratoires (toux, essoufflement, sifflements, etc.)</li>
                            <li>La contextualisation temporelle des crises/symptômes</li>
                            <li>Des données quantitatives sur la fréquence, durée et intensité des symptômes</li>
                            <li>Des réponses plus spécifiques aux questions posées</li>
                        </ul>
                        <p>Veuillez enrichir la discussion et réessayer.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Calculer le temps total
                timings["total_time"] = time.time() - timings["start_time"]
                
                # Afficher les informations de temps d'exécution
                st.markdown('<div class="subheader">⏱️ Performance du système</div>', unsafe_allow_html=True)

                # Formater les temps pour l'affichage
                formatted_timings = {
                    "Génération de la requête": f"{timings['query_generation_time']:.2f} sec",
                    "Évaluation de la discussion": f"{timings['evaluation_time']:.2f} sec",
                }
                
                # Ajouter les autres timings seulement si l'évaluation est positive
                if evaluation_result == "oui":
                    formatted_timings.update({
                        "Recherche de documents": f"{timings['document_retrieval_time']:.2f} sec",
                        "Génération de la suggestion": f"{timings['response_generation_time']:.2f} sec",
                    })
                
                formatted_timings["Temps total"] = f"{timings['total_time']:.2f} sec"

                # Heure de début et de fin
                start_time_str = datetime.fromtimestamp(timings["start_time"]).strftime("%H:%M:%S")
                end_time_str = datetime.fromtimestamp(timings["start_time"] + timings["total_time"]).strftime("%H:%M:%S")

                # Afficher la boîte de timing
                st.markdown(f"""
                <div class="timing-box">
                    <div class="timing-header">Analyse effectuée de {start_time_str} à {end_time_str}</div>
                </div>
                """, unsafe_allow_html=True)

                # Utiliser des colonnes pour afficher les timings
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Étape**", unsafe_allow_html=True)
                    for label in list(formatted_timings.keys())[:-1]:  # Exclure le temps total
                        st.markdown(f"{label}", unsafe_allow_html=True)
                    st.markdown(f"**Temps total d'analyse:**", unsafe_allow_html=True)

                with col2:
                    st.markdown("**Durée**", unsafe_allow_html=True)
                    for label in list(formatted_timings.keys())[:-1]:  # Exclure le temps total
                        st.markdown(f"{formatted_timings[label]}", unsafe_allow_html=True)
                    st.markdown(f"**{formatted_timings['Temps total']}**", unsafe_allow_html=True)

                # Graphique de répartition du temps
                st.markdown('<div class="subheader">📊 Répartition du temps d\'exécution</div>', unsafe_allow_html=True)
                
                # Préparer les données pour le graphique
                time_data = {
                    "Étape": ["Génération de la requête", "Évaluation de la discussion"],
                    "Temps (sec)": [
                        timings["query_generation_time"],
                        timings["evaluation_time"]
                    ]
                }
                
                # Ajouter les autres étapes si l'évaluation est positive
                if evaluation_result == "oui":
                    time_data["Étape"].extend(["Recherche de documents", "Génération de la suggestion"])
                    time_data["Temps (sec)"].extend([
                        timings["document_retrieval_time"],
                        timings["response_generation_time"]
                    ])
                
                time_df = pd.DataFrame(time_data)
                st.bar_chart(time_df.set_index("Étape"))
                
        else:
            st.error("⚠️ Veuillez entrer une discussion avant d'analyser.")
    elif analyze_button and not client:
        st.error("⚠️ Erreur de connexion à la base de données. Veuillez réessayer.")

with tab2:
    st.markdown('<div class="subheader">🎤 Enregistrement vocal par tranches</div>', unsafe_allow_html=True)

    # Initialiser les variables de session si nécessaire
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "total_recorded_files" not in st.session_state:
        st.session_state.total_recorded_files = 0
    if "last_recorded_file" not in st.session_state:
        st.session_state.last_recorded_file = None

    # Interface de contrôle d'enregistrement
    col_rec1, col_rec2, col_rec3 = st.columns([1, 1, 1])

    with col_rec1:
        if st.button("🎙️ Démarrer l'enregistrement", disabled=st.session_state.recording):
            if demarrer_enregistrement_streamlit():
                st.success("🔴 Enregistrement démarré!")
                st.rerun()

    with col_rec2:
        if st.button("⏹️ Arrêter l'enregistrement", disabled=not st.session_state.recording):
            if arreter_enregistrement_streamlit():
                st.success("⏹️ Enregistrement arrêté!")
                st.rerun()

    with col_rec3:
        if st.button("🔄 Actualiser le statut"):
            st.rerun()

    # Affichage du statut d'enregistrement
    if st.session_state.recording:
        st.markdown(f"""
        <div class="recording-active">
            <h4 style="color: #d32f2f; margin: 0;">🔴 Enregistrement en cours...</h4>
            <p style="margin: 0.5rem 0 0 0;">
                L'enregistrement se fait par tranches de {st.session_state.get('duree_tranche', 'valeur par défaut')} minutes.
            </p>
        </div>
        """, unsafe_allow_html=True)
