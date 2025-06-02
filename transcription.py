# transcription.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def transcrire_audio(filename):
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,  # PAS besoin de tuple (filename, data)
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )

    texte = transcription["text"]  # Accès correct au texte selon le format verbose_json
    print("📄 Transcription :", texte)

    # On enregistre dans un fichier transcription_<nom>.txt
    base = os.path.splitext(os.path.basename(filename))[0]
    output_file = f"transcription_{base}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(texte)

    return texte

if __name__ == "__main__":
    # Exemple : transcrire tous les fichiers dans "enregistrements"
    dossier = "enregistrements"
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".wav")]

    for fichier in fichiers:
        chemin_fichier = os.path.join(dossier, fichier)
        print(f"\n🔍 Transcription du fichier : {chemin_fichier}")
        transcrire_audio(chemin_fichier)
