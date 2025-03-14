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
    Allergologue : Bonjour Mme Dubois, je suis le Dr Martin. Comment allez-vous aujourd'hui ?
    Patient : Bonjour Docteur… Franchement, pas terrible. Ces éternuements et cette toux me rendent folle.
    Allergologue : Pourriez-vous me préciser si ces symptômes surviennent plutôt à une période spécifique de l'année, ou s'ils sont présents toute l'année ? 
    Patient : Hmm… je dirais que c’est pire au printemps, mais pas seulement.
    Allergologue : Pourriez-vous décrire plus précisément la nature de votre toux (sèche, grasse,  nocturne, diurne) et si elle est accompagnée de sifflements ou de gêne respiratoire,  notamment lors d'efforts ou de fous rires ?  
    Patient : Euh… Je sais pas trop… Ça vient et ça repart
    Allergologue : Ressentez-vous des démangeaisons ou des picotements au niveau du nez, des yeux ou de la gorge, en plus de vos éternuements et de votre toux ?
    Patient : Parfois, un peu au niveau des yeux…
    allergologue : Avez-vous des animaux domestiques à la maison, ou êtes-vous régulièrement en contact avec des animaux ?
    Patient : Non, pas vraiment…
    allergologue : Votre logement présente-t-il des signes d'humidité, et quel est le type de sol (moquette, carrelage, parquet…) ? 
                   Pouvez-vous également me décrire l'état de votre literie (âge du matelas, type d'oreiller…) ?"   
    Patient : Je dirais qu’il y a un peu d’humidité dans la salle de bain, mais sinon, je vois rien de spécial. Mon matelas, ça fait des années que je l'ai, et j'ai des oreillers en plumes, je crois.        
    allergologue : Avez-vous remarqué si vos symptômes s'aggravent dans certaines pièces de votre logement, comme la salle de bain par exemple, ou lorsque vous manipulez votre literie ?
    patient : Hmm… Peut-être un peu plus dans la chambre, surtout le matin.   
    allergologue :   Avez-vous déjà envisagé ou testé des housses anti-acariens pour votre matelas et vos oreillers ?" 
    patient: Non, jamais essayé… Ça pourrait aider ?  
    allergologue : En plus du printemps, vos symptômes s'aggravent-ils après avoir aéré votre literie ou après une longue période sans aérer votre chambre ?
    patient :  Maintenant que vous le dites… J’ai l’impression que c’est pire quand je reste longtemps sans aérer.   ,
    allegologue : Êtes-vous exposée à des moisissures dans votre environnement, que ce soit à votre domicile (autres pièces que la salle de bain) ou sur votre lieu de travail ?
    patient : Pas que je sache… Peut-être un peu dans un coin du salon, mais rien de flagrant.
    allergologue : Pourriez-vous me décrire la nature de l'écoulement nasal (clair, épais, couleur) et sa fréquence, notamment au réveil et  au printemps ?
    patient : C'est plutôt clair, et ça coule surtout le matin… Au printemps, c’est presque tous les jours.  
    allergologue : Ressentez-vous une amélioration de vos symptômes lorsque vous êtes en vacances, notamment en altitude ou dans un  environnement sec et aéré ?  
    Patient : Oui, en général ça va mieux quand je suis à la montagne ou au bord de la mer.
    allergologue : Avez-vous déjà réalisé des tests allergologiques, notamment pour les acariens, les moisissures et les pollens de graminées ?"  
    patient : Non, jamais… Vous pensez que ce serait utile ?
    allergologue : avez-vous remarqué si vos symptômes s'améliorent lorsque vous êtes loin de chez vous pendant une période prolongée, par  exemple plus d'une semaine  ? 
    Patient : Oui, je crois… Quand je pars plusieurs jours, surtout ailleurs qu’en ville, ça va un peu mieux.
    allergologue : Ressentez-vous une différence dans l'intensité de vos symptômes entre le début et la fin de la semaine, notamment si vous travaillez à l'extérieur de votre domicile ?
    Patient : Maintenant que j’y pense… Oui, c’est souvent pire en début de semaine, après le week-end à la maison.
    allergologue : Votre literie est composée de quel type de matériaux (matelas, sommier, oreillers, couvertures) ?.
    patient : Mon matelas est en mousse, le sommier est à lattes. J’ai des oreillers en plumes et une couette synthétique.
    allergologue : Vos symptômes s'améliorent-ils lorsque vous lavez vos draps et vos taies d'oreiller à haute température (60°C) ?"
    Patient : Je n’ai jamais vraiment fait attention… Peut-être un peu, mais pas de façon flagrante.
    allergologue : Avez-vous déjà utilisé un déshumidificateur d'air, que ce soit dans votre chambre, votre salle de bain ou votre salon, et si oui, avez-vous constaté une différence au niveau de vos symptômes ?"  
    Patient : Non, jamais essayé… Vous pensez que ça pourrait aider ?
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

    if not filtered_docs:
        print("\n❌ Aucun document avec un score > 0.70")
        response = "aucun document"


    question = "Propose une seule question pertinente à poser selon les informations et la discussion ? et dis-moi quelle ressources (Documentation , Logigramme , ...  ) tu as utilisés pour la choisir. "
        
        # Pass "aucun document" to retrieve_and_ask when no valid documents found
    if not filtered_docs:
        response = retrieve_and_ask([{"chunk_text": "aucun document"}], question, conversation_text )
    else:
            response = retrieve_and_ask(filtered_docs, question,conversation_text)

    

