# config.py
from dotenv import load_dotenv
import os
import pyaudio


load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL = "gemini-2.0-flash-exp"
    URI = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
    
    # Audio settings
    AUDIO_FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_SIZE = 256
    SAMPLE_RATE = 16000
    OUTPUT_RATE = 24000