import os
import time
import tempfile
from groq import Groq
import sounddevice as sd
import soundfile as sf
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Charger le modèle d'embedding vocal
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

def record_and_transcribe(duration=5, sample_rate=16000, similarity_threshold=0.7):
    """
    Enregistre et transcrit l'audio avec diarisation pour exactement deux locuteurs
    """
    client = Groq()
    
    # Stockage des embeddings de référence pour les deux locuteurs
    speaker_embeddings = {
        1: None,  # Premier locuteur
        2: None   # Deuxième locuteur
    }
    current_speaker_id = 1  # On commence avec le premier locuteur
    
    print("\nDIARISATION POUR DEUX LOCUTEURS")
    print("===============================")
    print("Le système alternera entre Speaker 1 et Speaker 2")
    print("Appuyez sur Ctrl+C pour arrêter l'enregistrement")
    
    try:
        while True:
            # Enregistrement audio
            print(f"\nEnregistrement de {duration} secondes...")
            audio_data = sd.rec(int(duration * sample_rate), 
                              samplerate=sample_rate, 
                              channels=1)
            sd.wait()
            
            # Sauvegarde dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)
                temp_file.flush()

                # Extraction de l'embedding vocal
                try:
                    signal = classifier.load_audio(temp_file.name)
                    embedding = classifier.encode_batch(signal)
                    embedding = embedding.squeeze().cpu().numpy()
                except Exception as e:
                    print(f"Erreur d'extraction d'embedding: {e}")
                    embedding = None
                
                # Détermination du locuteur actuel
                if embedding is not None:
                    # Initialisation des embeddings si nécessaire
                    if speaker_embeddings[1] is None:
                        # Premier segment audio, on l'attribue au premier locuteur
                        speaker_embeddings[1] = embedding
                        current_speaker_id = 1
                    elif speaker_embeddings[2] is None:
                        # Si le second embedding est assez différent du premier, on l'attribue au second locuteur
                        similarity = cosine_similarity([embedding], [speaker_embeddings[1]])[0][0]
                        if similarity < similarity_threshold:
                            speaker_embeddings[2] = embedding
                            current_speaker_id = 2
                        else:
                            # Sinon, on met à jour l'embedding du premier locuteur
                            speaker_embeddings[1] = 0.8 * speaker_embeddings[1] + 0.2 * embedding
                            current_speaker_id = 1
                    else:
                        # Les deux locuteurs sont connus, on compare avec les deux
                        similarity_1 = cosine_similarity([embedding], [speaker_embeddings[1]])[0][0]
                        similarity_2 = cosine_similarity([embedding], [speaker_embeddings[2]])[0][0]
                        
                        # On attribue au locuteur le plus similaire
                        if similarity_1 > similarity_2:
                            current_speaker_id = 1
                            # Mise à jour de l'embedding avec une moyenne pondérée
                            speaker_embeddings[1] = 0.8 * speaker_embeddings[1] + 0.2 * embedding
                        else:
                            current_speaker_id = 2
                            # Mise à jour de l'embedding avec une moyenne pondérée
                            speaker_embeddings[2] = 0.8 * speaker_embeddings[2] + 0.2 * embedding
                
                # Transcription
                with open(temp_file.name, "rb") as file:
                    try:
                        transcription = client.audio.transcriptions.create(
                            file=(temp_file.name, file.read()),
                            model="whisper-large-v3-turbo",
                            response_format="text",
                        )

                        # Affichage du résultat
                        print(f"\nSpeaker {current_speaker_id}:")
                        print("-" * 50)
                        print(getattr(transcription, 'text', transcription))
                        print("-" * 50)

                    except Exception as e:
                        print(f"Erreur de transcription: {e}")

    except KeyboardInterrupt:
        print("\nArrêt de l'enregistrement.")

if __name__ == "__main__":
    print("Outil d'enregistrement et transcription avec diarisation")
    input("Appuyez sur Entrée pour commencer...")
    record_and_transcribe()