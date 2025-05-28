import sounddevice as sd
from scipy.io.wavfile import write

def enregistrer_audio(fichier_sortie="enregistrement.wav", duree=60, frequence=44100):
    print(f"🎙️ Enregistrement en cours pendant {duree} secondes...")
    audio = sd.rec(int(duree * frequence), samplerate=frequence, channels=1, dtype='int16')
    sd.wait()
    write(fichier_sortie, frequence, audio)
    print(f"✅ Enregistrement terminé. Fichier sauvegardé sous : {fichier_sortie}")


if __name__ == "__main__":
    enregistrer_audio()