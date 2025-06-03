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
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        
        # Contexte médical spécialisé
        self.medical_context = """Contexte spécialisé:
- Spécialité: Allergologie respiratoire
- Vocabulaire médical typique: 
  * Symptômes: rhinorrhée, prurit, dyspnée, wheezing
  * Allergènes: acariens, pollens, phanères animaux
  * Traitements: antihistaminiques, corticostéroïdes
- Structure typique de consultation:
  1. Anamnèse (questions du médecin)
  2. Description des symptômes (patient)
  3. Examen clinique
  4. Recommandations
"""
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
                time.sleep(3)
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
    
    def detecter_questions_manquantes(self, texte_conversation: str) -> list:
        """
        Analyse automatique du dialogue structuré pour suggérer des questions médicales
        non abordées dans la consultation.
        """
        try:
            time.sleep(6)

            prompt = f"""
    Tu es un assistant médical expert en consultations d’allergologie.

    Tu as reçu le compte-rendu structuré suivant entre un médecin et un patient :

    {texte_conversation}

    Analyse cette discussion et identifie les **thèmes médicaux importants NON abordés** parmi :
    - Symptômes précis
    - Durée des symptômes
    - Environnement allergène (animaux, acariens, pollen…)
    - Traitements déjà essayés
    - Antécédents médicaux ou familiaux
    - Facteurs aggravants (saison, activité, lieu...)

    **Objectif :** proposer des questions que le MÉDECIN aurait pu poser mais n’a pas posées.

    **Format de réponse attendu :**
    === QUESTIONS COMPLÉMENTAIRES SUGGÉRÉES ===
    - [Question 1]
    - [Question 2]
    ...
    """

            response = self.gemini_model.generate_content(prompt)

            if response.text and "QUESTIONS COMPLÉMENTAIRES" in response.text:
                lines = response.text.strip().splitlines()
                suggestions = [
                    line.strip("- ").strip()
                    for line in lines
                    if line.startswith("- ")
                ]
                return suggestions
            else:
                return []

        except Exception as e:
            print(f"❌ Erreur lors de la détection de questions manquantes: {e}")
            return []

    def process_with_gemini(self):
        """Envoie la conversation complète à Gemini 2.0 Flash pour diarisation améliorée"""
        try:
            with self.conversation_lock:
                conversation_copy = self.full_conversation
            
            if not conversation_copy.strip():
                return
            
            time.sleep(6)
            # Nouveau prompt amélioré avec contexte médical
            prompt = f"""{self.medical_context}

Tu es un assistant médical expert en transcription et analyse de consultations d'allergologie. 

**Tâches :**
1. Identifier clairement les rôles (MÉDECIN vs PATIENT) selon le contenu : questions techniques pour le médecin, réponses descriptives pour le patient.
2. Corriger les erreurs de transcription, en particulier les confusions courantes liées à l'oral médical (ex : "toux" compris comme "tout", "rhume" mal orthographié, etc.).
3. Reconstituer les phrases interrompues ou incomplètes.
4. Structurer le dialogue de manière lisible, ligne par ligne, comme un échange :  
   - MÉDECIN : Bonjour, que puis-je faire pour vous aujourd’hui ?  
   - PATIENT : J’ai une toux sèche depuis trois jours...
   
**Transcription brute:**
{conversation_copy}

**Format de sortie:**
=== DIALOGUE STRUCTURÉ ===
[MÉDECIN/PATIENT clairement identifiés avec texte corrigé]

=== INDICES CLINIQUES ===
[Points clés à retenir]

**Exemple:**
MÉDECIN: Depuis quand présentez-vous ces symptômes respiratoires ?
PATIENT: Depuis environ 3 semaines, surtout le matin.
...
=== INDICES CLINIQUES ===
- Symptômes: majoration matinale
- Durée: 3 semaines
"""

            # Configuration améliorée
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,  # Un peu plus flexible
                    top_k=40,
                    top_p=0.95,
                    max_output_tokens=8192,
                )
            )
            
            if response.text:
                self.display_structured_conversation(response.text)

                # ➕ Suggestions intelligentes de questions manquantes
                suggestions = self.detecter_questions_manquantes(response.text)
                if suggestions:
                    print("\n💡 QUESTIONS SUPPLÉMENTAIRES À POSER :")
                    print("="*40)
                    for q in suggestions:
                        print(f"- {q}")
                else:
                    print("\n✅ Aucune question importante ne semble avoir été oubliée.")
            else:
                print("❌ Réponse Gemini vide")
                
        except Exception as e:
            print(f"❌ Erreur traitement Gemini: {e}")
    
    def display_structured_conversation(self, text):
        """Affiche la conversation structurée avec mise en forme"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n" + "="*60)
        print("🤖 CONSULTATION STRUCTURÉE (Gemini 2.0 Flash):")
        print("="*60)
        
        # Détection des sections
        if "===" in text:
            print(text)
        elif "MÉDECIN:" in text and "PATIENT:" in text:
            print(text)
        else:
            print("Dialogue non structuré détecté, formatage basique:")
            print("-"*40)
            print(text)
            print("-"*40)
        
        print("="*60 + "\n")
            
    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.is_recording = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()

        # Sauvegarder la transcription brute dans un fichier texte
        if self.full_conversation.strip():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            txt_filename = f"transcription_brute_{timestamp}.txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(self.full_conversation)
            print(f"📝 Transcription brute sauvegardée dans : {txt_filename}")

        # Traitement final amélioré
        # print("\n📄 Génération du rapport final...")
        self.generate_final_report()
        print("✅ Écoute arrêtée")

    
    def generate_final_report(self):
        """Génère un rapport final complet avec analyse"""
        if not self.full_conversation.strip():
            return
        
        if hasattr(self, 'structured_conversation') and self.structured_conversation:
            self.suggestions_questions = self.detecter_questions_manquantes(self.structured_conversation)
        else:
            self.suggestions_questions = []

        # print("\n" + "="*70)
        # print("📊 RAPPORT FINAL DE CONSULTATION (ANALYSE COMPLÈTE)")
        # print("="*70)


        time.sleep(6)
        
        # Prompt pour le rapport complet
        final_prompt = f"""{self.medical_context}

**Génère un rapport médical structuré contenant:**

1. Dialogue complet corrigé (format MÉDECIN/PATIENT)
2. Analyse des symptômes et signes cliniques
3. Hypothèses diagnostiques plausibles
4. Recommandations potentielles

**Transcription originale:**
{self.full_conversation}

**Format exigé:**
=== SYNTHÈSE CLINIQUE ===
[Date et heure] Consultation d'allergologie respiratoire

=== HISTORIQUE ===
- Motif consultation:
- Antécédents pertinents: 
- Traitements en cours:

=== DIALOGUE CORRIGÉ ===
[Structure claire MÉDECIN/PATIENT]

=== ANALYSE ===
- Symptômes clés: 
- Facteurs déclenchants: 
- Hypothèses diagnostiques: 
- Examens complémentaires suggérés: 

=== RECOMMANDATIONS ===
1. Conduite à tenir
2. Traitements proposés
3. Suivi recommandé
"""

        try:
            response = self.gemini_model.generate_content(final_prompt)
            
            # if response.text:
            #     # Sauvegarde améliorée
            #     timestamp = time.strftime("%Y%m%d_%H%M%S")
            #     filename = f"consultation_allergo_{timestamp}.txt"
                
            #     with open(filename, 'w', encoding='utf-8') as f:
            #         f.write("=== RAPPORT MÉDICAL COMPLET ===\n")
            #         f.write(f"Date: {timestamp}\n")
            #         f.write("Spécialité: Allergologie respiratoire\n")
            #         f.write("="*50 + "\n\n")
            #         f.write(response.text)
            #         f.write("\n\n=== TRANSCRIPTION BRUTE ===\n")
            #         f.write(self.full_conversation)
                
            #     print(response.text)
            #     print(f"\n💾 Fichier sauvegardé: {filename}")
            # else:
            #     print("❌ Échec de génération du rapport")
                
        except Exception as e:
            print(f"❌ Erreur lors de la génération du rapport: {e}")

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