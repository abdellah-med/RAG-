import asyncio
import websockets
import json
from utils.audio_handler import AudioHandler
import signal
import os

# Supprimer les messages d'erreur ALSA
os.environ['ALSA_CARD'] = 'Generic'

class TranscriptionClient:
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.ws = None
        self.is_recording = False
        self.loop = asyncio.get_event_loop()
        self.server_processing = False
        self.local_buffer = []
        self.buffer_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.disconnect_ack_received = asyncio.Event()
        
        # Remplacer keyboard par signal.SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Gestionnaire pour Ctrl+C"""
        print("\nCtrl+C détecté. Arrêt de l'enregistrement...")
        self.shutdown_event.set()  # Déclencher l'événement de shutdown
        self.loop.create_task(self.stop_recording())

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            # Ajouter ping_interval pour maintenir la connexion active
            self.ws = await websockets.connect('ws://192.168.1.103:8765', ping_interval=20)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
            
    async def start_recording(self):
        if not self.ws:
            raise RuntimeError("Not connected to server")
            
        stream = self.audio_handler.create_input_stream()
        self.is_recording = True
        
        try:
            # Create separate tasks
            send_task = asyncio.create_task(self._send_audio_loop(stream))
            receive_task = asyncio.create_task(self._receive_server_messages())
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())
            
            # Utiliser wait avec l'événement de shutdown
            done, pending = await asyncio.wait(
                [send_task, receive_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Annuler les tâches restantes
            for task in pending:
                task.cancel()
                
        finally:
            stream.stop_stream()
            stream.close()

    async def _send_audio_loop(self, stream):
        """Handle audio sending with buffering"""
        while self.is_recording:
            data = await asyncio.to_thread(stream.read, 256)
            
            if self.server_processing:
                # Buffer during processing
                # print("Buffering locally...")
                self.local_buffer.append(data)
            else:
                # Real-time mode
                if self.local_buffer:
                    # First send buffered data
                    print(f"Sending {len(self.local_buffer)} buffered chunks...")
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
                    print("Live Transcription:", response_data.get("new_messages", "No available transcription yet"))
                elif response_data.get("type") == "disconnect_ack":
                    print("Server acknowledged disconnect")
                    print("Final Analysis:", response_data.get("final_response", "No final analysis"))
                    self.disconnect_ack_received.set()
                    break
                
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
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
                await self.ws.send(json.dumps(disconnect_signal))

                # Wait for the disconnect acknowledgment using the event
                try:
                    await asyncio.wait_for(self.disconnect_ack_received.wait(), timeout=60.0)
                except asyncio.TimeoutError:
                    print("Timeout waiting for disconnect acknowledgment")
                
                finally:
                    # Close connection
                    await self.ws.close()
                    await self.ws.wait_closed()
                    self.ws = None
                    self.shutdown_event.set()
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection already closed")
                self.ws = None
            except Exception as e:
                print(f"Error during disconnect: {e}")
                self.ws = None
            
    async def stop_recording(self):
        """Stop recording"""
        # self.is_recording = False
        if self.ws:
            await self.disconnect()
            self.ws = None

        self.is_recording = False
        self.audio_handler.cleanup()

async def main():
    client = TranscriptionClient()
    
    if await client.connect():
        print("Connected to server. Starting recording...")
        print("\nPress Ctrl+C to stop recording...")
        try:
            await client.start_recording()
        except KeyboardInterrupt:
            print("\nStopping recording...")
        finally:
            await client.stop_recording()
    else:
        print("Failed to connect to server")

if __name__ == "__main__":
    asyncio.run(main())

