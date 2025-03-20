import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration de l'API Gemini
configure(api_key=GEMINI_API_KEY)

model = GenerativeModel("gemini-1.5-flash-latest")

def generate_query(conversation_text):
    prompt = f"""
    Tu es un assistant médical expert en analyse de conversations cliniques. Reformule l'échange entre l'allergologue et le patient en un texte optimisé pour la recherche RAG, extrayant uniquement les informations techniques essentielles à la décision médicale.

--- Conversation ---
{conversation_text}
--- Fin de la conversation ---

Génère UNIQUEMENT une reformulation structurée contenant:
1. Terminologie médicale précise (pathologies, médicaments, symptômes)
2. Mots-clés contextuels (antécédents, déclencheurs, chronologie)
3. Points d'ambiguïté clinique
4. Hypothèses diagnostiques implicites

Format: Phrases concises, termes techniques séparés par des virgules, sans questions. Priorise les éléments actionnables pour l'exploration médicale.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()
