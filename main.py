import os
import uuid
import pymupdf  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch

def generer_chunks(text, taille_chunk, chevauchement):
    """Découpe le texte en morceaux de taille donnée avec un chevauchement."""
    chunks = []
    debut = 0
    while debut < len(text):
        fin = debut + taille_chunk
        chunks.append(text[debut:fin])
        debut = fin - chevauchement  # Déplacer avec chevauchement
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extrait le texte page par page d'un PDF."""
    doc = pymupdf.open(pdf_path)
    return [page.get_text("text") for page in doc]  # Liste de texte par page

def connect_to_qdrant():
    """Connexion au serveur Qdrant."""
    return QdrantClient(url="http://localhost:6333")

# Charger SapBERT
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def get_embedding(text):
    """Convertit un texte en embedding avec SapBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliser le [CLS] token comme embedding de la phrase
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding

def create_collection(client, collection_name, vector_size=768):
    """Crée une collection dans Qdrant si elle n'existe pas."""
    collections = client.get_collections()
    
    if any(col.name == collection_name for col in collections.collections):
        print(f"Collection '{collection_name}' existe déjà. Skipping creation.")
        return False
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' créée.")
    return True

def index_all_pdfs(client, collection_name, folder_path, taille_chunk=2000, chevauchement=500, batch_size=50):
    """Indexe tous les PDFs d'un dossier dans Qdrant en découpant le texte en chunks et en envoyant par batch."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️ Aucun fichier PDF trouvé dans '{folder_path}' !")
        return

    points = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pages = extract_text_from_pdf(pdf_path)
        
        for i, page_text in enumerate(pages):
            if page_text.strip():  # Ignore les pages vides
                chunks = generer_chunks(page_text, taille_chunk, chevauchement)
                for j, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "file_name": pdf_file,
                            "page_number": i + 1,
                            "chunk_number": j + 1,
                            "chunk_text": chunk  # On stocke le texte du chunk
                        }
                    )
                    points.append(point)
                    print(f" Chunk {j+1} de la page {i+1} de '{pdf_file}' indexé avec ID {point.id}.")

                    
                    # Envoyer un batch lorsque la taille atteint `batch_size`
                    if len(points) >= batch_size:
                        client.upsert(collection_name=collection_name, wait=True, points=points)
                        print(f"✅ {len(points)} chunks envoyés à Qdrant.")
                        points = []  # Réinitialiser la liste
    
    # Envoyer les derniers points restants
    if points:
        client.upsert(collection_name=collection_name, wait=True, points=points)
        print(f"✅ Derniers {len(points)} chunks envoyés à Qdrant.")

def get_similar_documents(client, collection_name, query_text, top_k):
    """Recherche les documents similaires dans Qdrant et retourne les résultats avec leur contenu."""
    query_embedding = get_embedding(query_text)
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    if not results:
        print("Aucun document similaire trouvé.")
        return []
    
    return [
        {
            "id": res.id,
            "score": res.score,
            "page_number": res.payload.get("page_number"),
            "chunk_number": res.payload.get("chunk_number"),
            "file_name": res.payload.get("file_name"),
            "chunk_text": res.payload.get("chunk_text")  # Récupération du texte du chunk
        }
        for res in results
    ]

if __name__ == "__main__":
    collection_name = "corpus"
    client = connect_to_qdrant()
    
    # Vérifier si la collection existe déjà
    should_index = create_collection(client, collection_name, 768)  # Taille des embeddings SapBERT
    
    # Indexer uniquement si la collection n'existe pas
    if should_index:
        index_all_pdfs(client, collection_name, "ALLERG_IA")
    
    query = "quelle sont les symptomes de l'asthme ?"
    top_docs = get_similar_documents(client, collection_name, query, 5)
    
    # Affichage des résultats
    print("\n🔍 Résultats de la recherche :\n")
    for doc in top_docs:
        print(f"Fichier : {doc['file_name']}")
        print(f"Page : {doc['page_number']}, Chunk : {doc['chunk_number']}")
        print(f"Score : {doc['score']:.4f}")
        print(f"Contenu du chunk :\n{doc['chunk_text']}\n")
        print("-" * 80)
