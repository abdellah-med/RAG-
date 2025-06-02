import os
import time
import threading
import tempfile
import wave
import pyaudio
import numpy as np
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class GroqWhisperLiveTranscriber:
    def __init__(self, api_key=None, gemini_api_key=None):
        # Initialiser le client Groq
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        self.client = Groq()
        
        # Configuration Gemini 2.0 Flash
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configuration audio
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3  # Segments de 3 secondes
        
        # Variables de contrôle
        self.is_recording = False
        self.full_conversation = ""  # Accumulation de toute la discussion
        self.conversation_lock = threading.Lock()
        
        # Initialiser PyAudio
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        """Démarre l'enregistrement et transcription en temps réel"""
        self.is_recording = True
        
        # Configuration du stream audio
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("🎤 Écoute démarrée - Diarisation Médecin/Patient avec Gemini 2.0 Flash...")
        print("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while self.is_recording:
                # Enregistrer un segment audio
                audio_file_path = self.record_segment()
                
                # Transcrire le segment avec Groq
                if audio_file_path:
                    threading.Thread(
                        target=self.transcribe_and_process, 
                        args=(audio_file_path,)
                    ).start()
                    
        except KeyboardInterrupt:
            print("\n🛑 Arrêt de l'écoute...")
            self.stop_recording()
            
    def record_segment(self):
        """Enregistre un segment audio de 3 secondes"""
        frames = []
        
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            if not self.is_recording:
                break
                
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print(f"Erreur d'enregistrement: {e}")
                break
                
        if frames:
            # Créer un fichier temporaire
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            
            # Sauvegarder l'audio en WAV
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            return temp_file.name
        
        return None
        
    def transcribe_and_process(self, audio_file_path):
        """Transcrit avec Groq puis accumule et traite avec Gemini"""
        try:
            # Transcription avec Groq
            with open(audio_file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_file_path, file.read()),
                    model="whisper-large-v3-turbo",
                    language="fr"
                )
            
            text = transcription.text.strip()
            
            if text and len(text) > 1:
                # Ajouter à la conversation complète
                with self.conversation_lock:
                    timestamp = time.strftime("%H:%M:%S")
                    self.full_conversation += f"[{timestamp}] {text}\n"
                
                # Envoyer à Gemini pour diarisation
                self.process_with_gemini()
                
        except Exception as e:
            print(f"❌ Erreur de transcription: {e}")
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(audio_file_path)
            except:
                pass
    
    def process_with_gemini(self):
        """Envoie la conversation complète à Gemini 2.0 Flash pour diarisation"""
        try:
            with self.conversation_lock:
                conversation_copy = self.full_conversation
            
            if not conversation_copy.strip():
                return
                
            # Prompt pour la diarisation médecin/patient
            prompt = f"""Tu es un expert en diarisation de conversations médicales. 
Analyse cette transcription d'une consultation entre un médecin allergologue respiratoire et un patient.

Identifie qui parle (MÉDECIN ou PATIENT) et structure la conversation de manière claire.

Transcription brute:
{conversation_copy}

Retourne la conversation structurée au format:
MÉDECIN: [texte]
PATIENT: [texte]
MÉDECIN: [texte]
etc.

Sois précis dans l'identification des interlocuteurs basé sur le contexte médical (questions vs réponses, vocabulaire médical, etc.)."""

            # Appel à Gemini 2.0 Flash
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=8192,
                )
            )
            
            if response.text:
                # Effacer l'écran et afficher la conversation structurée
                os.system('clear' if os.name == 'posix' else 'cls')
                print("\n" + "="*60)
                print("🤖 CONSULTATION STRUCTURÉE (Gemini 2.0 Flash):")
                print("="*60)
                print(response.text)
                print("="*60 + "\n")
            else:
                print("❌ Réponse Gemini vide")
                
        except Exception as e:
            print(f"❌ Erreur traitement Gemini: {e}")
            
    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio.terminate()
        
        # Traitement final
        print("\n📄 Génération du rapport final...")
        self.generate_final_report()
        print("✅ Écoute arrêtée")
    
    def generate_final_report(self):
        """Génère un rapport final structuré"""
        if self.full_conversation.strip():
            print("\n" + "="*70)
            print("📊 RAPPORT FINAL DE CONSULTATION")
            print("="*70)
            self.process_with_gemini()  # Traitement final complet
            
            # Sauvegarder dans un fichier
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"consultation_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("CONSULTATION MÉDICALE\n")
                f.write("="*50 + "\n\n")
                f.write("Transcription brute:\n")
                f.write(self.full_conversation)
                f.write("\n\nStructuration réalisée par Gemini 2.0 Flash")
            
            print(f"💾 Consultation sauvegardée dans: {filename}")

def main():
    # Vérifier les clés API
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Veuillez définir votre clé API Groq:")
        print("export GROQ_API_KEY='votre_clé_ici'")
        return
        
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Veuillez définir votre clé API Gemini:")
        print("export GEMINI_API_KEY='votre_clé_ici'")
        print("Obtenez votre clé sur: https://aistudio.google.com/app/apikey")
        return
    
    # Créer et démarrer le transcripteur
    transcriber = GroqWhisperLiveTranscriber()
    transcriber.start_recording()

if __name__ == "__main__":
    main()