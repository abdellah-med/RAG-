import asyncio
import websockets
import json
import streamlit as st
import os
import signal
import threading
import time
from utils.audio_handler import AudioHandler

# Supprimer les messages d'erreur ALSA
os.environ['ALSA_CARD'] = 'Generic'

class TranscriptionClientStreamlit:
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.ws = None
        self.is_recording = False
        self.server_processing = False
        self.local_buffer = []
        self.disconnect_ack_received = False
        self.transcript_history = []
        self.status_messages = []
        
    async def connect(self, server_url='ws://192.168.1.103:8765'):
        """Connect to WebSocket server"""
        try:
            self.ws = await websockets.connect(server_url, ping_interval=20)
            self.add_status_message("Connecté au serveur")
            return True
        except Exception as e:
            self.add_status_message(f"Erreur de connexion: {e}")
            return False
            
    async def start_recording(self):
        if not self.ws:
            raise RuntimeError("Non connecté au serveur")
            
        stream = self.audio_handler.create_input_stream()
        self.is_recording = True
        self.add_status_message("Enregistrement démarré")
        
        try:
            send_task = asyncio.create_task(self._send_audio_loop(stream))
            receive_task = asyncio.create_task(self._receive_server_messages())
            
            # Attendre que les tâches se terminent ou soient annulées
            await asyncio.gather(send_task, receive_task, return_exceptions=True)
                
        finally:
            stream.stop_stream()
            stream.close()
            self.add_status_message("Enregistrement arrêté")

    async def _send_audio_loop(self, stream):
        """Handle audio sending with buffering"""
        while self.is_recording:
            data = await asyncio.to_thread(stream.read, 256)
            
            if self.server_processing:
                # Buffer during processing
                self.local_buffer.append(data)
            else:
                # Real-time mode
                if self.local_buffer:
                    # First send buffered data
                    self.add_status_message(f"Envoi de {len(self.local_buffer)} chunks en mémoire tampon...")
                    for buffered_data in self.local_buffer:
                        await self.send_audio_chunk(buffered_data)
                    self.local_buffer = []
                
                # Send current chunk
                await self.send_audio_chunk(data)

    async def _receive_server_messages(self):
        """Handle server messages and state"""
        while self.is_recording:
            try:
                response = await self.ws.recv()
                response_data = json.loads(response)
                
                # Update processing state
                if response_data.get("status") == "processing":
                    self.server_processing = True
                elif response_data.get("status") == "success":
                    self.server_processing = False
                    new_messages = response_data.get("new_messages", [])
                    if new_messages:
                        for msg in new_messages:
                            speaker = msg.get("speaker_role", "Unknown")
                            message_text = msg.get("message", "")
                            self.add_transcript(f"[{speaker}]: {message_text}")
                    else:
                        self.add_status_message("Pas de transcription disponible pour le moment")
                elif response_data.get("type") == "disconnect_ack":
                    self.add_status_message("Le serveur a confirmé la déconnexion")
                    if "final_response" in response_data:
                        self.add_transcript(f"Analyse finale: {response_data.get('final_response')}")
                    self.disconnect_ack_received = True
                    break
                
            except websockets.exceptions.ConnectionClosed:
                self.add_status_message("Connexion fermée")
                break
            except Exception as e:
                self.add_status_message(f"Erreur lors de la réception du message: {e}")
                break

    async def send_audio_chunk(self, data):
        """Send a single audio chunk to server"""
        encoded_data = self.audio_handler.encode_audio_data(data)
        await self.ws.send(json.dumps({
            "realtime_input": {
                "media_chunks": [{
                    "data": encoded_data,
                    "mime_type": "audio/pcm"
                }]
            }
        }))

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.ws:
            try:
                # Send disconnect signal
                disconnect_signal = {
                    "type": "disconnect",
                    "status": "client_closing"
                }
                self.add_status_message("Envoi du signal de déconnexion...")
                await self.ws.send(json.dumps(disconnect_signal))

                # Wait for the disconnect acknowledgment
                try:
                    timeout = 60
                    start_time = time.time()
                    while not self.disconnect_ack_received and time.time() - start_time < timeout:
                        await asyncio.sleep(0.1)
                    
                    if not self.disconnect_ack_received:
                        self.add_status_message("Timeout en attendant la confirmation de déconnexion")
                
                finally:
                    # Close connection
                    await self.ws.close()
                    self.ws = None
                    self.add_status_message("Connexion fermée")
                
            except websockets.exceptions.ConnectionClosed:
                self.add_status_message("Connexion déjà fermée")
                self.ws = None
            except Exception as e:
                self.add_status_message(f"Erreur lors de la déconnexion: {e}")
                self.ws = None
            
    async def stop_recording(self):
        """Stop recording"""
        if self.is_recording:
            self.is_recording = False
            self.add_status_message("Arrêt de l'enregistrement...")
            
            if self.ws:
                await self.disconnect()
            
            self.audio_handler.cleanup()
            self.add_status_message("Nettoyage audio terminé")

    # Méthodes pour l'interface Streamlit
    def add_transcript(self, message):
        """Ajouter un message de transcription à l'historique"""
        self.transcript_history.append(message)
        
    def add_status_message(self, message):
        """Ajouter un message de statut"""
        self.status_messages.append(message)
        
    def get_transcript_history(self):
        """Récupérer l'historique des transcriptions"""
        return self.transcript_history
        
    def get_status_messages(self):
        """Récupérer les messages de statut"""
        return self.status_messages

# Function to run async tasks from Streamlit
def run_async(coro):
    return asyncio.run(coro)

# Streamlit App
def main():
    st.title("Transcription Audio en Temps Réel")
    
    # Initialize session state
    if 'client' not in st.session_state:
        st.session_state.client = TranscriptionClientStreamlit()
        st.session_state.is_connected = False
        st.session_state.is_recording = False
    
    client = st.session_state.client
    
    # Connection section
    st.subheader("Connexion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        server_url = st.text_input("URL du serveur", value="ws://192.168.1.103:8765")
    
    with col2:
        if not st.session_state.is_connected:
            if st.button("Connecter"):
                if run_async(client.connect(server_url)):
                    st.session_state.is_connected = True
                    st.experimental_rerun()
        else:
            if st.button("Déconnecter"):
                run_async(client.disconnect())
                st.session_state.is_connected = False
                st.session_state.is_recording = False
                st.experimental_rerun()
    
    # Recording section (enabled only if connected)
    st.subheader("Enregistrement")
    
    if st.session_state.is_connected:
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.is_recording:
                if st.button("Démarrer l'enregistrement"):
                    st.session_state.is_recording = True
                    # Run in a separate thread to avoid blocking the UI
                    threading.Thread(target=lambda: run_async(client.start_recording())).start()
        
        with col2:
            if st.session_state.is_recording:
                if st.button("Arrêter l'enregistrement"):
                    run_async(client.stop_recording())
                    st.session_state.is_recording = False
    else:
        st.info("Connectez-vous d'abord au serveur pour démarrer l'enregistrement")
    
    # Transcription display
    st.subheader("Transcription en temps réel")
    
    # Create a placeholder for dynamic updates
    transcript_placeholder = st.empty()
    
    # Display transcripts
    transcript_text = "\n\n".join(client.get_transcript_history())
    if transcript_text:
        transcript_placeholder.markdown(f"```\n{transcript_text}\n```")
    else:
        transcript_placeholder.info("En attente de transcription...")
    
    # Status messages
    st.subheader("Messages de statut")
    status_placeholder = st.empty()
    
    # Display status messages
    status_text = "\n".join(client.get_status_messages())
    if status_text:
        status_placeholder.code(status_text)
    
    # Auto-refresh to update transcripts (every 1 second)
    st.empty()
    time.sleep(1)
    st.experimental_rerun()

if __name__ == "__main__":
    main()