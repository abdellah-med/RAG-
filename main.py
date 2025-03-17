from indexall_minilm import (
    connect_to_qdrant,
    create_collection,
    index_all_pdfs,
    get_similar_documents
)
from query_gen import generate_query
from agnooo import retrieve_and_ask


if __name__ == "__main__":
    collection_name = "corpus_a"
    client = connect_to_qdrant()

    should_index = create_collection(client, collection_name, 384)  # Taille des embeddings MiniLM-L6

    if should_index:
        index_all_pdfs(client, collection_name, "ALLERG_IA")

    conversation_text = """ 

    Allergologue : Bonjour, qu’est-ce qui vous amène aujourd’hui ?
    Patiente : Bonjour, docteur. Je viens vous voir parce que depuis quelque temps, j’ai des difficultés à respirer. Par moments, j’ai une sensation d’oppression dans la poitrine, et ça arrive surtout le matin… Je ne sais pas trop pourquoi.
    Allergologue : ourriez-vous me décrire plus précisément la nature de vos difficultés respiratoires matinales ? S'agit-il d'une sensation d'oppression, d'essoufflement, de sifflements, de toux ?  Ressentez-vous également des symptômes au niveau du nez, des yeux ou de la gorge, comme des démangeaisons, des écoulements ou des éternuements ? 
    Patiente : Au niveau du nez et des yeux… hmm… maintenant que vous le dites, il m’arrive d’avoir le nez un peu bouché le matin, mais ce n’est pas toujours le cas. Par contre, pas vraiment d’éternuements ni de démangeaisons particulières.
    allergologue : Pourriez-vous me décrire plus précisément la nature de vos difficultés respiratoires matinales ? S'agit-il d'une sensation d'oppression, d'essoufflement, de sifflements, de toux ?  Ressentez-vous également des symptômes au niveau du nez, des yeux ou de la gorge, comme des démangeaisons, des écoulements ou des éternuements ? 
    patiente : Mon matelas n’est pas très vieux, je l’ai changé il y a environ deux ans. J’ai un oreiller en mousse, et je change mes draps toutes les deux semaines environ.
               Non, je n’ai pas d’animaux à la maison, je n’ai jamais eu de problème avec ça. Mais… enfin… je ne pense pas que ça vienne de ma literie, parce que mes symptômes ne sont pas présents quand je vais me coucher, c’est plutôt après le réveil, quand je commence ma journée.
    allergologue: Votre chambre est-elle exposée à l'humidité ou à des moisissures visibles, par exemple sur les murs ou autour des fenêtres ?  
    Partinente : Non, je ne pense pas. Ma chambre est plutôt bien ventilée, et je n’ai jamais remarqué de traces de moisissures sur les murs ou autour des fenêtres.
                 D’ailleurs, je n’ai pas vraiment ces problèmes de respiration quand je passe du temps dans ma chambre… C’est surtout le matin, après ma douche, que je ressens cette gêne.           
    
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

    # Filtrer les documents avec un score > 0.70
    filtered_docs = [doc for doc in top_docs if doc['score'] > 0.70]



    question = "Propose une seule question pertinente à poser selon les informations et la discussion, comme si elle était posée par le médecin. Explique également quelles ressources (Documentation, Logigramme, etc.) tu as utilisées pour choisir cette question. "
        
        # Pass "aucun document" to retrieve_and_ask when no valid documents found
    if not filtered_docs:
        response = retrieve_and_ask([{"chunk_text": "aucun document"}], question, conversation_text )
    else:
        response = retrieve_and_ask(filtered_docs, question,conversation_text)
     


    print(response)
    

