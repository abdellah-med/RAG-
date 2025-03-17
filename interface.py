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

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA - Conversations Médicales",
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
    Cette application utilise l'IA pour analyser des conversations médicales et suggérer des questions pertinentes basées sur une base documentaire.
    </div>
    """, unsafe_allow_html=True)

# Initialisation de Qdrant (avec cache pour éviter la réindexation)
client, collection_name = initialize_qdrant()

# En-tête principal
st.markdown('<h1 class="main-header">🩺 Assistant IA pour Conversations Médicales</h1>', unsafe_allow_html=True)

# Tabs pour organiser l'interface
tab1, tab2 = st.tabs(["Analyse de Conversation", "Documentation"])

with tab1:
    # Champ de texte pour la conversation
    conversation_text = st.text_area(
        "Entrez la transcription de la conversation médicale :",
        height=250,
        placeholder="Exemple: Le patient se plaint de démangeaisons cutanées et de difficultés respiratoires après avoir consommé des fruits à coque..."
    )
    
    # Boutons d'action
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyser la conversation", use_container_width=True)
    with col2:
        if st.button("🗑️ Effacer", use_container_width=True):
            conversation_text = ""
            st.experimental_rerun()
    
    # Analyse de la conversation
    if analyze_button:
        if conversation_text.strip():
            # Afficher un spinner pendant le traitement
            with st.spinner("Analyse en cours..."):
                # Générer la query
                query = generate_query(conversation_text)
                
                # Créer un conteneur pour afficher les résultats
                results_container = st.container()
                
                with results_container:
                    st.markdown('<div class="subheader">📊 Résultats de l\'analyse</div>', unsafe_allow_html=True)
                    
                    # Afficher la requête générée dans un expander
                    with st.expander("🔎 Requête générée"):
                        st.info(query)
                    
                    # Récupérer les documents similaires
                    top_docs = get_similar_documents(client, collection_name, query, num_results)
                    
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
                    
                    # Générer la réponse
                    question = (
                        "Propose une seule question pertinente à poser selon les informations et la discussion, "
                        "comme si elle était posée par le médecin. Explique également quelles ressources (Documentation, Logigramme, etc.) "
                        "tu as utilisées pour choisir cette question."
                    )
                    
                    if not filtered_docs:
                        st.warning("❌ Aucun document ne dépasse le seuil de pertinence minimum.")
                        response = retrieve_and_ask([{"chunk_text": "aucun document"}], question, conversation_text)
                    else:
                        response = retrieve_and_ask(filtered_docs, question, conversation_text)
                    
                    # Afficher la réponse générée
                    st.markdown('<div class="subheader">💡 Suggestion de l\'IA</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="recommendation-box">{response}</div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Veuillez entrer une conversation avant d'analyser.")

with tab2:
    st.markdown('<div class="subheader">📖 Documentation du système</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Comment utiliser cette application

    1. **Entrez la transcription** de la conversation médecin-patient dans le champ de texte
    2. **Cliquez sur 'Analyser'** pour lancer le traitement
    3. **Consultez les résultats** :
       - La requête générée par l'IA
       - Les documents pertinents trouvés dans la base documentaire
       - La question recommandée par l'IA

    ### Fonctionnalités principales

    - **Analyse de conversation** : extraction des informations clés
    - **Recherche sémantique** : identification des documents pertinents
    - **IA générative** : suggestion de questions adaptées au contexte
    
    ### Base documentaire
    
    L'application utilise une base de connaissances spécialisée en allergologie stockée dans la collection "corpus_a".
    """)

# Pied de page
st.markdown("""
<div class="footer">
    © 2025 - Assistant IA pour Conversations Médicales - Développé avec Streamlit
</div>
""", unsafe_allow_html=True)