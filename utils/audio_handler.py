# audio_handler.py
import pyaudio
import numpy as np
import asyncio
import base64
from .config import Config

import webrtcvad

class AudioHandler:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.audio_queue = asyncio.Queue()
        
        # Paramètres de détection du silence
        self.SILENCE_THRESHOLD = 100  # Ajusté pour être plus sensible
        self.SILENCE_DURATION = 15    # Durée du silence en secondes
        self.silence_frames = 0
        self.frames_per_second = Config.SAMPLE_RATE / Config.CHUNK_SIZE
        self.silence_limit = int(self.SILENCE_DURATION * self.frames_per_second)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

    def create_input_stream(self):
        return self.audio.open(
            format=Config.AUDIO_FORMAT,
            channels=Config.CHANNELS,
            rate=Config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=Config.CHUNK_SIZE
        )

    def create_output_stream(self):
        return self.audio.open(
            format=Config.AUDIO_FORMAT,
            channels=Config.CHANNELS,
            rate=Config.OUTPUT_RATE,
            output=True
        )

    def encode_audio_data(self, audio_data):
        return base64.b64encode(audio_data).decode()

    def decode_audio_data(self, audio_data):
        return base64.b64decode(audio_data)

    def get_audio_level(self, audio_data):
        """
        Calcule le niveau audio d'un chunk de manière plus robuste
        """
        # Convertir les données audio en tableau numpy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Utiliser la valeur absolue moyenne comme mesure du niveau audio
        return np.mean(np.abs(audio_array))

    def is_silent(self, audio_data):
        """
        Détecte si un segment audio est silencieux en utilisant la valeur absolue moyenne
        """
        level = self.get_audio_level(audio_data)
        is_silent = False
        is_silent = (int(level) < int(self.SILENCE_THRESHOLD))
        
        # print(f"Silent frame detected ?. Level: {level}, Threshold: {self.SILENCE_THRESHOLD} is_silent: {is_silent} ")
        return is_silent, level


    async def process_audio_chunk(self, chunk):
        """
        Traite un chunk audio et met à jour le compteur de silence
        Retourne True si le silence est détecté pendant assez longtemps
        """
        isSilent, level = self.is_silent(chunk)
        if isSilent :
            self.silence_frames += 1
            # if self.silence_frames >= self.silence_limit:
            #     print(f"Silence detected for {self.SILENCE_DURATION} seconds, self.silence_frames : {self.silence_frames}")

            return True, level
        else:
            self.silence_frames = 0
        return False, level

    def reset_silence_detection(self):
        """
        Réinitialise le compteur de silence
        """
        self.silence_frames = 0

    def cleanup(self):
        """
        Nettoie les ressources audio
        """
        self.audio.terminate()

    def adjust_silence_threshold(self, audio_data, factor=1.2):
        """
        Ajuste dynamiquement le seuil de silence basé sur le niveau audio ambiant
        """
        current_level = self.get_audio_level(audio_data)
        self.SILENCE_THRESHOLD = current_level * factor
        return self.SILENCE_THRESHOLD