from qdrant_client import QdrantClient

def delete_all_collections():
    """Connecte à Qdrant et supprime toutes les collections."""
    client = QdrantClient(url="http://localhost:6333")  # Connexion à Qdrant
    collections = client.get_collections().collections  # Récupérer toutes les collections
    
    for col in collections:
        client.delete_collection(col.name)  # Supprimer chaque collection
        print(f"Collection '{col.name}' supprimée.")

    print("Toutes les collections ont été supprimées.")

if __name__ == "__main__":
    delete_all_collections()
