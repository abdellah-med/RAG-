import streamlit as st
from indexall_minilm import (
    connect_to_qdrant,
    create_collection,
    index_all_pdfs,
    get_similar_documents
)
from query_gen import generate_query
from agnooo import retrieve_and_ask
import pandas as pd
import time
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA - discussions Médicales",
    page_icon="🩺",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

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

# Tabs pour organiser l'interface
tab1, tab2 = st.tabs(["Analyse de discussion", "Documentation"])

with tab1:
    # Champ de texte pour la discussion
    discussion_text = st.text_area(
        "Entrez la discussion :",
        height=250
    )
    
    # Boutons d'action
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyser la discussion", use_container_width=True)
    with col2:
        if st.button("🗑️ Effacer", use_container_width=True):
            discussion_text = ""
            st.experimental_rerun()
    
    # Analyse de la discussion
    if analyze_button:
        if discussion_text.strip():
            # Initialiser les temps pour chaque étape
            timings = {
                "start_time": time.time(),
                "query_generation_time": 0,
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
                
                # Afficher la requête générée dans un expander
                with st.expander("🔎 Requête générée"):
                    st.info(query)
                
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
                        "Propose une seule question pertinente à poser selon les informations et la discussion, "
                        "comme si elle était posée par le médecin. Explique également quelles ressources (Documentation, Logigramme, etc.) "
                        "tu as utilisées pour choisir cette question."
                    )
                    
                    if not filtered_docs:
                        st.warning("❌ Aucun document ne dépasse le seuil de pertinence minimum.")
                        response = retrieve_and_ask([{"chunk_text": "aucun document"}], question, discussion_text)
                    else:
                        response = retrieve_and_ask(filtered_docs, question, discussion_text)
                timings["response_generation_time"] = time.time() - response_start
                
                # Calculer le temps total
                timings["total_time"] = time.time() - timings["start_time"]
                
                # Afficher la réponse générée
                st.markdown('<div class="subheader">💡 Suggestion de l\'IA</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="recommendation-box">{response}</div>', unsafe_allow_html=True)
                
                # Afficher les informations de temps d'exécution
                st.markdown('<div class="subheader">⏱️ Performance du système</div>', unsafe_allow_html=True)

                # Formater les temps pour l'affichage
                formatted_timings = {
                    "Génération de la requête": f"{timings['query_generation_time']:.2f} sec",
                    "Recherche de documents": f"{timings['document_retrieval_time']:.2f} sec",
                    "Génération de la suggestion": f"{timings['response_generation_time']:.2f} sec",
                    "Temps total": f"{timings['total_time']:.2f} sec"
                }

                # Heure de début et de fin
                start_time_str = datetime.fromtimestamp(timings["start_time"]).strftime("%H:%M:%S")
                end_time_str = datetime.fromtimestamp(timings["start_time"] + timings["total_time"]).strftime("%H:%M:%S")

                # Afficher la boîte de timing en utilisant des composants Streamlit natifs
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

                # Ajouter un graphique pour visualiser la répartition du temps
                st.markdown('<div class="subheader">📊 Répartition du temps d\'exécution</div>', unsafe_allow_html=True)
                time_df = pd.DataFrame({
                    "Étape": ["Génération de la requête", "Recherche de documents", "Génération de la suggestion"],
                    "Temps (sec)": [
                        timings["query_generation_time"],
                        timings["document_retrieval_time"],
                        timings["response_generation_time"]
                    ]
                })
                st.bar_chart(time_df.set_index("Étape"))
                
        else:
            st.error("⚠️ Veuillez entrer une discussion avant d'analyser.")

with tab2:
    st.markdown('<div class="subheader">📖 Documentation du système</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Comment utiliser cette application

    1. **Entrez la transcription** de la discussion médecin-patient dans le champ de texte
    2. **Cliquez sur 'Analyser'** pour lancer le traitement
    3. **Consultez les résultats** :
       - La requête générée par l'IA
       - Les documents pertinents trouvés dans la base documentaire
       - La question recommandée par l'IA
       - Les métriques de performance du système

    ### Fonctionnalités principales

    - **Analyse de discussion** : extraction des informations clés
    - **Recherche sémantique** : identification des documents pertinents
    - **IA générative** : suggestion de questions adaptées au contexte
    - **Analyse de performance** : mesure des temps d'exécution de chaque étape
    
    ### Base documentaire
    
    L'application utilise une base de connaissances spécialisée en allergologie stockée dans la collection "corpus_a".
    """)

# Pied de page
st.markdown("""
<div class="footer">
    © 2025 - Assistant IA pour discussions Médicales - Développé avec Streamlit
</div>
""", unsafe_allow_html=True)