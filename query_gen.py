import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔹 Initialiser le client Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_query(conversation_text):
    prompt = f"""
    Tu es un assistant spécialisé en extraction d'informations médicales.
    À partir de cette conversation entre un allergologue et un patient, génère une query concise et pertinente
    pour interroger un système RAG afin d'obtenir des informations utiles.

    --- Conversation ---
    {conversation_text}
    --- Fin de la conversation ---

    Génère uniquement une query sous forme de question médicale claire et précise.
    """

    response = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()
