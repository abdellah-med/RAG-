# Modifications à apporter dans votre fichier record_audio.py

import sounddevice as sd
import wave
import os
import threading
import time
import numpy as np
from datetime import datetime
import streamlit as st

class StreamlitAudioRecorder:
    def __init__(self, duree_tranche=5, frequence=44100, dossier_sortie="enregistrements"):
        self.duree_tranche = duree_tranche
        self.frequence = frequence
        self.dossier_sortie = dossier_sortie
        self.compteur = 1
        
        # Créer le dossier de sortie
        os.makedirs(self.dossier_sortie, exist_ok=True)
    
    def enregistrer_audio_continu(self):
        """Fonction d'enregistrement audio qui vérifie l'état Streamlit"""
        print(f"🎙️ Démarrage de l'enregistrement par tranches de {self.duree_tranche} secondes...")
        
        try:
            while st.session_state.get("recording", False):
                # Générer un nom de fichier avec timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nom_fichier = os.path.join(
                    self.dossier_sortie, 
                    f"enregistrement_{self.compteur:03d}_{timestamp}.wav"
                )
                
                print(f"🎧 Enregistrement du fichier : {nom_fichier}")
                
                try:
                    # Enregistrer par petits segments pour pouvoir vérifier l'état
                    segments_par_tranche = 10  # 10 segments de 0.5 sec pour une tranche de 5 sec
                    segment_duration = self.duree_tranche / segments_par_tranche
                    samples_per_segment = int(segment_duration * self.frequence)
                    
                    all_audio_data = []
                    
                    for segment in range(segments_par_tranche):
                        # Vérifier si on doit arrêter l'enregistrement
                        if not st.session_state.get("recording", False):
                            break
                        
                        # Enregistrer un petit segment
                        audio_segment = sd.rec(
                            samples_per_segment,
                            samplerate=self.frequence,
                            channels=1,
                            dtype='int16'
                        )
                        sd.wait()
                        all_audio_data.append(audio_segment)
                    
                    # Si on a des données audio, les sauvegarder
                    if all_audio_data and st.session_state.get("recording", False):
                        # Concaténer tous les segments
                        audio_data = np.concatenate(all_audio_data, axis=0)
                        self._sauvegarder_wav(nom_fichier, audio_data)
                        print(f"✅ Fichier sauvegardé : {nom_fichier}")
                        self.compteur += 1
                        
                        # Mettre à jour le statut dans Streamlit
                        st.session_state.last_recorded_file = nom_fichier
                        st.session_state.total_recorded_files = self.compteur - 1
                    
                except Exception as e:
                    print(f"Erreur lors de l'enregistrement du segment {self.compteur}: {e}")
                    continue
                
        except Exception as e:
            print(f"Erreur générale durant l'enregistrement : {e}")
        finally:
            print("🛑 Enregistrement arrêté proprement.")
            st.session_state.recording = False
    
    def _sauvegarder_wav(self, nom_fichier, audio_data):
        """Sauvegarder les données audio au format WAV"""
        with wave.open(nom_fichier, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(self.frequence)
            wf.writeframes(audio_data.tobytes())
    
    def get_all_recordings(self):
        """Retourner la liste de tous les enregistrements"""
        files = []
        for filename in os.listdir(self.dossier_sortie):
            if filename.endswith('.wav'):
                filepath = os.path.join(self.dossier_sortie, filename)
                files.append({
                    'name': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                })
        return sorted(files, key=lambda x: x['modified'], reverse=True)

# Fonction à utiliser dans votre interface Streamlit
def demarrer_enregistrement_streamlit():
    """Démarre l'enregistrement dans un thread compatible avec Streamlit"""
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = StreamlitAudioRecorder()
    
    if not st.session_state.get("recording", False):
        st.session_state.recording = True
        st.session_state.total_recorded_files = 0
        
        # Démarrer l'enregistrement dans un thread
        thread = threading.Thread(target=st.session_state.audio_recorder.enregistrer_audio_continu)
        thread.daemon = True
        thread.start()
        return True
    return False

def arreter_enregistrement_streamlit():
    """Arrête l'enregistrement"""
    if st.session_state.get("recording", False):
        st.session_state.recording = False
        return True
    return False