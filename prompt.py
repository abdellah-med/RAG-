import json
import re
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("La clé API GEMINI_API_KEY n'est pas définie dans les variables d'environnement.")

agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0),
    description="Tu es un modèle capable d'identifier si un texte est prononcé par un allergologue ou un patient, en utilisant des balises.",
    instructions=[
        "Réponds uniquement par la balise correcte : <Allergologue> ou <Patient>.",
        "Ne donne aucune autre explication."
    ],
    markdown=False
)

def split_into_sentences(text):
    # Sépare le texte en phrases sur . ! ? suivi d'espace ou fin de texte
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def classify_speaker(text: str) -> str:
    prompt = (
        "Tu es un modèle qui identifie si une phrase est dite par un allergologue ou un patient.\n"
        "Tu dois répondre uniquement par une balise exacte : <Allergologue> ou <Patient>, sans autre texte.\n"
        "Exemples :\n"
        "<Allergologue>Depuis combien de temps avez-vous ces symptômes ?</Allergologue>\n"
        "<Patient>Ça a commencé il y a deux semaines, surtout la nuit.</Patient>\n"
        "<Allergologue>Est-ce que vous avez des antécédents d’allergies ?</Allergologue>\n"
        "<Patient>Non, aucun antécédent connu.</Patient>\n\n"
        f"Phrase à classifier : \"{text}\"\n"
        "Balise correcte :"
    )
    response = agent.run(prompt)
    tag = response.content.strip()
    if "<Allergologue>" in tag:
        return "Allergologue"
    else:
        return "Patient"
    # En cas d'erreur ou d'ambiguïté, retourner Patient par défaut
    # else:
    #     return "Patient"

def process_transcription_file(filename: str):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    
    sentences = split_into_sentences(text)
    results = []
    for sentence in sentences:
        speaker = classify_speaker(sentence)
        results.append({"speaker": speaker, "text": sentence})
    return results

def save_results_as_json(results, output_filename="transcription_classified.json"):
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    filename = "transcription.txt"
    results = process_transcription_file(filename)
    save_results_as_json(results)
    print(f"Fichier classifié sauvegardé dans transcription_classified.json")
