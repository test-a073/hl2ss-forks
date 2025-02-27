### 002_main_pipeline.py

# This script implements a HoloLens-based audio recording system 
# that captures audio data, saves recordings, transcribes them 
# using Google Speech Recognition, and appends the transcription 
# to a file. 

import logging
import os
from datetime import datetime
from contextlib import contextmanager
import io
import time
from typing import Optional

import hl2ss
import hl2ss_lnm
import hl2ss_utilities
from pydub import AudioSegment
from pynput import keyboard
import speech_recognition as sr

class AudioRecordingError(Exception):
    """Custom exception for audio recording errors"""
    pass

class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass

class HoloLensAudioRecorder:
    def __init__(self, host: str, output_folder: str, transcription_file: str):
        self.host = host
        self.output_folder = output_folder
        self.transcription_file = transcription_file
        self.enable = True
        self.is_recording = False
        self.combined_audio = AudioSegment.empty()
        self.client = None
        self.current_audio_file = None
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Validate transcription file path
        transcription_dir = os.path.dirname(transcription_file)
        if transcription_dir:
            os.makedirs(transcription_dir, exist_ok=True)
            
        # Test if transcription file is writable
        try:
            with open(transcription_file, 'a') as f:
                pass
        except IOError as e:
            raise AudioRecordingError(f"Cannot write to transcription file: {e}")

    @contextmanager
    def connect_client(self):
        """Context manager for HoloLens client connection"""
        try:
            self.client = hl2ss_lnm.rx_microphone(
                self.host, 
                hl2ss.StreamPort.MICROPHONE, 
                profile=hl2ss.AudioProfile.RAW
            )
            self.client.open()
            logging.info(f"Connected to HoloLens at {self.host}")
            yield self.client
        except Exception as e:
            raise AudioRecordingError(f"Failed to connect to HoloLens: {e}")
        finally:
            if self.client:
                self.client.close()
                logging.info("HoloLens microphone client closed")

    def on_press(self, key) -> bool:
        try:
            if key.char == 'w':  # Start recording
                if not self.is_recording:
                    logging.info("Recording started...")
                    self.is_recording = True
                    self.combined_audio = AudioSegment.empty()
            
            elif key.char == 'q':  # Stop recording
                if self.is_recording:
                    self._save_recording()
                    self.enable = False
                    return False  # Stop listener
                    
        except AttributeError:
            pass
        return True

    def _save_recording(self):
        """Save the current recording"""
        if len(self.combined_audio) < 100:  # Minimum length check (100ms)
            logging.warning("Recording too short, discarding")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recording_{timestamp}.wav"
        self.current_audio_file = os.path.join(self.output_folder, filename)
        
        try:
            self.combined_audio.export(self.current_audio_file, format="wav")
            logging.info(f"Audio saved as {self.current_audio_file}")
        except Exception as e:
            raise AudioRecordingError(f"Failed to save audio: {e}")
        finally:
            self.is_recording = False

    def transcribe_audio(self, audio_file: str, timeout: int = 30) -> Optional[str]:
        """Transcribe audio file with timeout"""
        start_time = time.time()
        
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                
            text = recognizer.recognize_google(audio_data)
            
            if time.time() - start_time > timeout:
                raise TranscriptionError("Transcription timeout")
                
            return text
            
        except sr.UnknownValueError:
            logging.error("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logging.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None

    def append_transcription(self, text: str):
        """Append transcription to file with error handling"""
        if not text:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.transcription_file, "a") as f:
                f.write(f"\n[{timestamp}] {text}")
            logging.info("Transcription appended successfully")
        except IOError as e:
            logging.error(f"Failed to write transcription: {e}")

    def run(self):
        """Main execution method"""
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        logging.info("Press 'W' to start recording, 'Q' to stop")

        with self.connect_client() as client:
            while self.enable:
                try:
                    data = client.get_next_packet()
                    audio = data.payload
                    
                    if self.is_recording:
                        segment = AudioSegment.from_file(
                            io.BytesIO(audio.tobytes()),
                            format="raw",
                            frame_rate=48000,
                            channels=2,
                            sample_width=2
                        )
                        self.combined_audio += segment
                except Exception as e:
                    logging.error(f"Error during recording: {e}")
                    self.enable = False

        listener.join()
        
        if self.current_audio_file:
            text = self.transcribe_audio(self.current_audio_file)
            if text:
                self.append_transcription(text)
                return text

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load configuration
    try:
        import config
        HOLOLENS_IP = config.HOLOLENS_IP
    except ImportError:
        logging.error("Could not load config.py")
        HOLOLENS_IP = input("Please enter HoloLens IP address: ")
    
    # Initialize and run recorder
    try:
        recorder = HoloLensAudioRecorder(
            host=HOLOLENS_IP,
            output_folder="sasika_stream",
            transcription_file="sasika_stream/transcription.txt"
        )
        transcribed_text = recorder.run()

        import json

        # Define the dictionary
        queries = {
            "text": transcribed_text,  # Replace with the actual transcribed text
            "image": "sasika_stream/sample.jpg"
        }

        # Specify the output file path
        output_file = "sasika_stream/queries.json"

        try:
            # Write the dictionary to the JSON file
            with open(output_file, "w") as json_file:
                json.dump(queries, json_file, indent=4)  # Use indent for readability
            print(f"Data successfully written to {output_file}")
        except IOError as e:
            print(f"Error writing to file: {e}")

        print(queries)

    except Exception as e:
        logging.error(f"Fatal error: {e}")