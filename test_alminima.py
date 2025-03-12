import os
import uuid
import pymupdf  # PyMuPDF
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from agnooo import retrieve_and_ask
from query_gen import generate_query

# Chargement de MiniLM-L6
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embedding(text):
    """Convertit un texte en embedding avec MiniLM-L6."""
    return model.encode(text, convert_to_numpy=False).tolist()

def generer_chunks_paragraphes(paragraphes, taille_chunk, chevauchement):
    """Génère des chunks en fonction du nombre de mots avec chevauchement."""
    chunks = []
    mots = " ".join(paragraphes).split()
    n = len(mots)

    if chevauchement >= taille_chunk:
        raise ValueError("Le chevauchement doit être inférieur à la taille du chunk.")

    debut = 0
    while debut < n:
        fin = min(debut + taille_chunk, n)  
        chunk = " ".join(mots[debut:fin])  
        chunks.append(chunk)
        
        if fin == n:
            break

        debut += (taille_chunk - chevauchement)

    return chunks

def extract_text_from_pdf(pdf_path):
    """Extrait et nettoie le texte d'un PDF en paragraphes."""
    try:
        doc = pymupdf.open(pdf_path)
        contenu_complet = []
        
        for num_page in range(len(doc)):
            page = doc[num_page]
            texte = page.get_text("text").strip()
            
            if texte:
                contenu_complet.append(texte)
        
        doc.close()
        
        if not contenu_complet:
            print(f"Le document {pdf_path} est vide ou illisible.")
            return []
        
        texte_propre = re.sub(r"\n{2,}", "\n", "\n".join(contenu_complet))
        texte_propre = re.sub(r"\s+", " ", texte_propre).strip()
        
        return [p.strip() for p in texte_propre.split("\n") if p.strip()]
    
    except Exception as e:
        print(f"Erreur de lecture du PDF {pdf_path} : {e}")
        return []

def connect_to_qdrant():
    """Connexion au serveur Qdrant."""
    return QdrantClient(url="http://localhost:6333")

def create_collection(client, collection_name, vector_size=384):
    """Crée une collection dans Qdrant si elle n'existe pas."""
    collections = client.get_collections()
    
    if any(col.name == collection_name for col in collections.collections):
        print(f"✅ Collection '{collection_name}' existe déjà. Skipping creation.")
        return False
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' créée.")
    return True

def index_all_pdfs(client, collection_name, folder_path, taille_chunk=128, chevauchement=50, batch_size=50):
    """Indexe tous les PDFs d'un dossier dans Qdrant en découpant le texte en chunks."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ Aucun fichier PDF trouvé dans '{folder_path}' !")
        return
    
    points = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        paragraphes = extract_text_from_pdf(pdf_path)

        if not paragraphes:
            continue
        
        chunks = generer_chunks_paragraphes(paragraphes, taille_chunk, chevauchement)
        
        for j, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "file_name": pdf_file,
                    "chunk_number": j + 1,
                    "chunk_text": chunk
                }
            )
            points.append(point)
            print(f"Chunk {j+1} de '{pdf_file}' indexé avec ID {point.id}.")
            
            if len(points) >= batch_size:
                client.upsert(collection_name=collection_name, wait=True, points=points)
                print(f" {len(points)} chunks envoyés à Qdrant.")
                points = []
    
    if points:
        client.upsert(collection_name=collection_name, wait=True, points=points)
        print(f"✅ Derniers {len(points)} chunks envoyés à Qdrant.")

def get_similar_documents(client, collection_name, query_text, top_k):
    """Recherche les documents similaires dans Qdrant et retourne les résultats."""
    query_embedding = get_embedding(query_text)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    if not results:
        print(" Aucun document similaire trouvé.")
        return []
    
    return [
        {
            "id": res.id,
            "score": res.score,
            "file_name": res.payload.get("file_name"),
            "chunk_number": res.payload.get("chunk_number"),
            "chunk_text": res.payload.get("chunk_text")
        }
        for res in results
    ]

if __name__ == "__main__":
    collection_name = "corpus_a"
    client = connect_to_qdrant()
    
    should_index = create_collection(client, collection_name, 384)  # Taille des embeddings MiniLM-L6
    
    if should_index:
        index_all_pdfs(client, collection_name, "ALLERG_IA")


    conversation_text = """
    "Allergologue : Bonjour, Monsieur. Je suis le Dr. Morel, allergologue. Que puis-je faire pour vous aujourd'hui ?\n"
    "Patient : Bonjour, docteur. Depuis quelque temps, j’ai souvent une sensation de fatigue et des maux de tête. J’ai aussi le nez qui coule sans raison apparente.\n\n"
    "Allergologue : Avez-vous remarqué si ces symptômes s’accentuent à certains moments de la journée ?\n"
    "Patient : Oui, surtout en fin d’après-midi et parfois la nuit. Par contre, le matin, ça semble aller un peu mieux.\n\n"
    "Allergologue : D’accord. Est-ce que vous ressentez une gêne respiratoire ou des douleurs au niveau de la poitrine ?\n"
    "Patient : Pas de douleur, mais parfois j’ai l’impression de devoir respirer plus profondément, comme si l’air était plus lourd.\n\n"
    "Allergologue : Avez-vous des antécédents d’allergies ou de problèmes respiratoires ?\n"
    "Patient : Non, jamais eu d’allergies connues, mais mon frère a eu des crises d’asthme dans son enfance.\n\n"
    """

    # 🔹 Générer la query
    query = generate_query(conversation_text)

    # 🔹 Afficher le résultat
    print("🔎 Query générée :", query)    
    

    top_docs = get_similar_documents(client, collection_name, query, 5)
    
    for doc in top_docs:
        print(f"**Fichier** : {doc['file_name']}")
        print(f"**Chunk** : {doc['chunk_number']}")
        print(f"**Score** : {doc['score']:.4f}")
        print(f"**Contenu du chunk** :\n{doc['chunk_text']}\n")
        print("-" * 80)


     # Filter documents with score > 0.75
    filtered_docs = [doc for doc in top_docs if doc['score'] > 0.70]

    
        

       
    if not filtered_docs:
        print("\n❌ Aucun document avec un score > 0.75")
        response = "aucun document"
            

        # Only ask question if we have valid documents
    question = "Propose une seule question pertinente à poser ? et dis-moi quelle ressources (Documentation , Logigramme , ...  ) tu as utilisés pour la choisir. "
        
        # Pass "aucun document" to retrieve_and_ask when no valid documents found
    if not filtered_docs:
        response = retrieve_and_ask([{"chunk_text": "aucun document"}], question)
    else:
            response = retrieve_and_ask(filtered_docs, question)

    print(response)

    

