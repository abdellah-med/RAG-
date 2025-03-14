import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration de l'API Gemini
configure(api_key=GEMINI_API_KEY)

model = GenerativeModel("gemini-1.5-pro-latest")

def generate_query(conversation_text):
    prompt = f"""
    Tu es un assistant médical expert en analyse de conversations cliniques. 
    Ton rôle est de reformuler les échanges entre l'allergologue et le patient 
    en un texte optimisé pour la recherche RAG, afin d'extraire des informations techniques 
    nécessaires à la prise de décision médicale.

    --- Conversation ---
    {conversation_text}
    --- Fin de la conversation ---

    Génère UNIQUEMENT une reformulation structurée contenant :
    1. Terminologie médicale précise (pathologies, médicaments, symptômes)
    2. Mots-clés contextuels (antécédents, déclencheurs, chronologie)
    3. Points d'ambiguïté à éclaircir
    4. Hypothèses diagnostiques implicites

    Format souhaité : Phrases concises et termes techniques séparés par des virgules, 
    sans formulation de question explicite. Priorisez les éléments actionnables pour guider 
    l'exploration médicale.

    Exemple de sortie : "rhinite allergique saisonnière, antécédent d'asthme infantile, 
    réaction au pollen de bouleau, corticoïdes nasaux inefficaces, suspicion d'allergie croisée"
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()
