import os
from dotenv import load_dotenv
import groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration du client Groq
client = groq.Client(api_key=GROQ_API_KEY)

def generate_query(conversation_text):
    prompt = """
    Tu es un assistant médical expert en analyse de conversations cliniques. Reformule l'échange entre l'allergologue et le patient en un texte optimisé pour la recherche RAG, extrayant uniquement les informations techniques essentielles à la décision médicale.

    Génère UNIQUEMENT une reformulation structurée contenant:
    1. Terminologie médicale précise (pathologies, médicaments, symptômes)
    2. Mots-clés contextuels (antécédents, déclencheurs, chronologie)
    3. Points d'ambiguïté clinique
    4. Hypothèses diagnostiques implicites

    Format: Phrases concises, termes techniques séparés par des virgules, sans questions. Priorise les éléments actionnables pour l'exploration médicale.
        """
    
    response = client.chat.completions.create(
        model="qwen-2.5-32b",  # Vous pouvez choisir un autre modèle Groq comme "mixtral-8x7b-32768"
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation_text}
        ],
        temperature=0.3,
        max_tokens=50
    )
    
    return response.choices[0].message.content.strip()