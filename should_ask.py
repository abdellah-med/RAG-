import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluer_recommandation(discussion: str, contexte: str) -> str:
    """
    Évalue si une discussion médicale contient suffisamment d'informations pertinentes
    sur les symptômes respiratoires pour poursuivre l'analyse, en tenant également 
    compte de la requête générée et du logigramme médical.
    
    Args:
        discussion: Le texte de la discussion médecin-patient
        contexte: Contient le logigramme et la requête générée
        
    Returns:
        'oui' si la discussion est de qualité suffisante (score > 0.5), 'non' sinon
    """

    system_prompt = f"""
    [Rôle]
    Vous êtes un expert en analyse de conversations médicales d'allergologie respiratoire.

    [Contexte]
    Vous analysez une discussion médicale ainsi que la requête générée à partir de celle-ci.
    Le logigramme fourni dans le contexte sert de référence pour évaluer si la discussion contient 
    des éléments pertinents pour un diagnostic d'allergologie respiratoire.

    [Consignes]
    1. Analysez minutieusement la conversation entre médecin allergologue et patient
    2. Examinez la requête générée et le logigramme fournis dans le contexte
    3. Évaluez la qualité et pertinence de la discussion sur une échelle de 0 à 1
    4. Répondez UNIQUEMENT avec un nombre décimal entre 0 et 1 
       - 1.0 signifie que la discussion est parfaitement détaillée et pertinente
       - 0.0 signifie que la discussion est totalement inadéquate
    
    Un score élevé est attribué si:
    - Symptômes respiratoires clairement décrits
    - Contextualisation temporelle précise des symptômes
    - La requête générée cible correctement des éléments du logigramme
    
    Un score faible est attribué si:
    - La conversation manque de précision sur les symptômes
    - La requête générée est trop générique
    - La discussion ne permet pas d'avancer dans le diagnostic
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation à analyser:\n{discussion}\n\nContexte (logigramme et requête générée):\n{contexte}"}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,  # Température plus faible pour obtenir un nombre précis
            max_tokens=3,  # Très peu de tokens pour forcer une réponse courte
        )

        reponse = response.choices[0].message.content.strip()
        print(f"Score de confiance: {reponse}")
        
        try:
            # Convertir la réponse en nombre flottant
            indice = float(reponse)
            
            # S'assurer que l'indice est entre 0 et 1
            indice = max(0.0, min(1.0, indice))
            
            print(f"Score normalisé: {indice}")
            
            # Retourner 'oui' si l'indice est supérieur à 0.5, 'non' sinon
            return "oui" if indice > 0.1 else "non"
            
        except ValueError:
            print(f"Erreur: impossible de convertir '{reponse}' en nombre")
            # Fallback sur la méthode par mot-clé
            return "oui" if "oui" in reponse.lower() else "non"

    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        return "non"  # En cas d'erreur, on préfère ne pas continuer le traitement
