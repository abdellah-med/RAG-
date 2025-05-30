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
    description="Tu es un modèle capable d'identifier qui parle dans une transcription (allergologue ou patient).",
    instructions=[
        "Tague chaque réplique du texte avec <Allergologue> ou <Patient> autour de chaque phrase.",
        "Si plusieurs phrases sont prononcées par la même personne à la suite, regroupe-les dans la même balise.",
        "Ne modifie pas le contenu.",
        "Ne donne aucune explication ni commentaire."
    ],
    markdown=False
)

def classify_whole_transcript(text: str) -> list:
    prompt = (
        "Tu es un modèle qui identifie qui parle dans un dialogue entre un allergologue et un patient.\n"
        "Tu dois encadrer chaque réplique avec une balise : <Allergologue> ... </Allergologue> ou <Patient> ... </Patient>.\n"
        "Ne modifie pas le texte. Ne donne aucune explication.\n"
        "Voici la transcription :\n"
        f"{text}\n"
        "\nRéponse attendue :"
    )

    response = agent.run(prompt)
    tagged_text = response.content.strip()

    # Extraction des blocs tagués
    pattern = r"<(Allergologue|Patient)>(.*?)</\1>"
    matches = re.findall(pattern, tagged_text, re.DOTALL)

    results = []
    for speaker, speech in matches:
        speech = speech.strip()
        if speech:
            results.append({
                "speaker": speaker,
                "text": speech
            })
    return results

def process_transcription_file(filename: str):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return classify_whole_transcript(text)

def save_results_as_json(results, output_filename="transcription_classified.json"):
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    filename = "transcription.txt"
    results = process_transcription_file(filename)
    save_results_as_json(results)
    print(f"Fichier classifié sauvegardé dans transcription_classified.json")
