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
            file=(filename, file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    texte = transcription.text
    print("Transcription :", transcription.text)

    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(texte)

    return transcription.text
