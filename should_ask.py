import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def evaluer_recommandation(discussion: str, contexte: str) -> str:

    system_prompt = f"""
    [Rôle]
    Vous êtes un expert en analyse de conversations médicales d'allergologie respiratoire.

    [Consignes]
    1. Analysez cette conversation entre médecin allergologue et patient et les symptômes respiratoires mentionnés
    2. Répondez UNIQUEMENT par 'oui' ou 'non' sans ponctuation
    3. 'oui' si :
    - Symptômes respiratoires clairement décrits (toux, essoufflement, sifflements, etc.) OU
    - Contextualisation temporelle précise des crises/symptômes OU
    - Informations quantitatives présentes (fréquence, durée, intensité) OU
    - Questions du médecin pertinentes et réponses spécifiques du patient OU
    - Vous identifiez des questions importantes que le médecin aurait dû poser mais n'a pas posées
    4. 'non' si :
    - La conversation manque de précision sur les symptômes OU
    - Le langage utilisé est trop technique ou difficile à comprendre OU
    - Le ton de la conversation est inapproprié ou trop alarmiste OU
    - Dans tous les autres cas
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation à analyser:\n{discussion}"}
            ],
            model="qwen-2.5-32b",
            temperature=0.7,
            max_tokens=3,
            stop=["\n"]
        )

        reponse = response.choices[0].message.content.lower().strip()
        return "oui" if reponse == "oui" else "non"  # Validation exacte

    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        return "non"
