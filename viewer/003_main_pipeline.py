### 003_main_pipeline.py

# This Python script defines a HoloLensRecorder class,
#  which integrates audio and video recording from a
#  HoloLens device. It enables live video streaming, 
#  periodic image frame saving, and audio recording 
#  with transcription capabilities using Google Speech 
#  Recognition. The script supports keyboard controls 
#  for recording ('W' to start and 'Q' to stop) and 
#  organizes audio transcriptions and associated image
#  frames into structured queries saved in a JSON file. 


import logging
import os
from datetime import datetime
from contextlib import contextmanager
import io
import time
from typing import Optional
import json
import cv2
import multiprocessing as mp

import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import hl2ss_mp
from pydub import AudioSegment
from pynput import keyboard
import speech_recognition as sr



class RecordingError(Exception):
    """Custom exception for recording errors"""
    pass

class HoloLensRecorder:
    def __init__(self, host: str, output_folder: str, transcription_file: str):
        self.host = host
        self.output_folder = output_folder
        self.transcription_file = transcription_file
        self.enable = True
        self.is_recording = False
        self.combined_audio = AudioSegment.empty()
        self.audio_client = None
        self.current_audio_file = None
        
        # Query tracking
        self.queries = []
        self.current_image_path = None
        self.current_text = None
        
        # PV (Personal Video) settings
        self.pv_width = 760
        self.pv_height = 428
        self.pv_framerate = 30
        self.buffer_elements = 150
        self.image_save_interval = 20  # Save frame every 20 seconds
        self.last_image_save_time = 0
        
        # Initialize multiprocessing components
        self.producer = None
        self.consumer = None
        self.sinks = {}
        
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
            raise RecordingError(f"Cannot write to transcription file: {e}")

    @contextmanager
    def initialize_streams(self):
        """Initialize both audio and video streams"""
        try:
            # Start PV Subsystem
            hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
            
            # Initialize producer and consumer
            self.producer = hl2ss_mp.producer()
            self.producer.configure(
                hl2ss.StreamPort.PERSONAL_VIDEO,
                hl2ss_lnm.rx_pv(
                    self.host,
                    hl2ss.StreamPort.PERSONAL_VIDEO,
                    width=self.pv_width,
                    height=self.pv_height,
                    framerate=self.pv_framerate
                )
            )
            self.producer.configure(
                hl2ss.StreamPort.MICROPHONE,
                hl2ss_lnm.rx_microphone(self.host, hl2ss.StreamPort.MICROPHONE)
            )

            self.consumer = hl2ss_mp.consumer()
            manager = mp.Manager()
            
            # Initialize both streams
            for port in [hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.StreamPort.MICROPHONE]:
                self.producer.initialize(port, self.buffer_elements)
                self.producer.start(port)
                self.sinks[port] = self.consumer.create_sink(self.producer, port, manager, None)
                self.sinks[port].get_attach_response()
                while (self.sinks[port].get_buffered_frame(0)[0] != 0):
                    pass
                logging.info(f'Started stream on port {port}')
                
        except Exception as e:
            raise RecordingError(f"Failed to initialize streams: {e}")

    def save_image(self, image):
        """Save the current frame if interval has elapsed"""
        current_time = time.time()
        if current_time - self.last_image_save_time >= self.image_save_interval:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = os.path.join(self.output_folder, f"frame_{timestamp}.jpg")
            cv2.imwrite(image_path, image)
            self.last_image_save_time = current_time
            self.current_image_path = image_path
            logging.info(f"Saved image: {image_path}")

    def process_video_frame(self, payload, show_image=False):
        """Process and save video frames"""
        if payload.image is not None and payload.image.size > 0:
            self.save_image(payload.image)
            if show_image == True:
                cv2.imshow('Personal Video', payload.image)

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
        """Save the current audio recording"""
        # if len(self.combined_audio) < 100:  # Minimum length check (100ms)
        #     logging.warning("Recording too short, discarding")
        #     return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recording_{timestamp}.wav"
        self.current_audio_file = os.path.join(self.output_folder, filename)
        
        try:
            self.combined_audio.export(self.current_audio_file, format="wav")
            logging.info(f"Audio saved as {self.current_audio_file}")
        except Exception as e:
            raise RecordingError(f"Failed to save audio: {e}")
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
                raise RecordingError("Transcription timeout")
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

    def save_queries(self):
        """Save queries to a JSON file"""
        queries_file = os.path.join(self.output_folder, "queries.json")
        try:
            with open(queries_file, 'w') as f:
                json.dump(self.queries, f, indent=2)
            logging.info(f"Saved queries to {queries_file}")
        except IOError as e:
            logging.error(f"Failed to save queries: {e}")

    def load_queries(self):
        """Load existing queries from JSON file"""
        queries_file = os.path.join(self.output_folder, "queries.json")
        try:
            if os.path.exists(queries_file):
                with open(queries_file, 'r') as f:
                    self.queries = json.load(f)
                logging.info(f"Loaded {len(self.queries)} existing queries")
        except IOError as e:
            logging.error(f"Failed to load queries: {e}")

    def append_transcription(self, text: str):
        """Append transcription to file with error handling and update queries"""
        if not text:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Write to transcription file
            with open(self.transcription_file, "a") as f:
                f.write(f"\n[{timestamp}] {text}")
            logging.info("Transcription appended successfully")
            
            self.current_text = text
            # Add to queries list
            if self.current_image_path:
                query = {
                    'image_path': self.current_image_path,
                    'transcription': text,
                    'timestamp': timestamp
                }
                self.queries.append(query)
                logging.info(f"Added query: Image: {self.current_image_path}, Text: {text}")
                
                # Save queries to JSON
                self.save_queries()
                
        except IOError as e:
            logging.error(f"Failed to write transcription: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop streams
            for port in self.sinks:
                self.sinks[port].detach()
                self.producer.stop(port)
                logging.info(f'Stopped stream on port {port}')

            # Stop PV Subsystem
            hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
            
            # Destroy OpenCV windows
            cv2.destroyAllWindows()
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def get_queries(self):
        """Return the list of queries"""
        return self.queries

    def run(self):
        """Main execution method"""
        try:
            # Load existing queries
            # self.load_queries()
            
            # Initialize streams
            self.initialize_streams()
            
            # Start keyboard listener
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()
            logging.info("Press 'W' to start recording, 'Q' to stop")

            # Main loop
            while self.enable:
                # Process video frames
                _, video_data = self.sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame()
                if video_data is not None:
                    self.process_video_frame(video_data.payload, show_image=True)

                # Process audio if recording
                if self.is_recording:
                    _, audio_data = self.sinks[hl2ss.StreamPort.MICROPHONE].get_most_recent_frame()
                    if audio_data is not None:
                        audio = audio_data.payload
                        segment = AudioSegment.from_file(
                            io.BytesIO(audio.tobytes()),
                            format="raw",
                            frame_rate=48000,
                            channels=2,
                            sample_width=2
                        )
                        self.combined_audio += segment

                cv2.waitKey(1)

            listener.join()
            
            # Process final recording if exists
            #------------------------------------------------------------------
            # TODO: Change this (The audio input is noisy when both audio and video inputs are in)
            self.current_audio_file = "sasika_stream/recording_2025-01-07_13-24-55.wav"
            
            if self.current_audio_file:
                text = self.transcribe_audio(self.current_audio_file)
                if text:
                    self.current_text = text
                    self.append_transcription(text)

        except Exception as e:
            logging.error(f"Error during recording: {e}")
        finally:
            self.cleanup()
            # Save final state of queries
            self.save_queries()

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
        recorder = HoloLensRecorder(
            host=HOLOLENS_IP,
            output_folder="sasika_stream",
            transcription_file="sasika_stream/transcription.txt"
        )
        recorder.run()

        queries = {
            "text": recorder.current_text,
            "image" : recorder.current_image_path
        }

        print(queries)

        print()

        import random
        import time

        def policy_function() -> int:
            """Returns the device for inference.
            0 : Workstation
            1 : Jetson Nano
            """
            return random.randint(0, 1)

        # Determine the inference device
        inference_device = policy_function()
        inference_device = 0

        if inference_device == 0:
            import requests

            server_url = "http://10.4.16.46:5000/query"
            image_path = queries["image"]
            query_text = queries["text"]

            try:
                start_time = time.time()

                # Prepare the request payload
                with open(image_path, 'rb') as img_file:
                    files = {'image': img_file}
                    data = {'query': query_text}

                    # Send the POST request
                    response = requests.post(server_url, files=files, data=data)

                # Measure processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Handle the response
                if response.status_code == 200:
                    print("Response:", response.json().get('response', "No response field"))
                else:
                    print("Error:", response.json())

                print(f"Processing Time: {processing_time:.2f} seconds")
            except Exception as e:
                print(f"An error occurred: {e}")

        elif inference_device == 1:
            import ollama

            try:
                start_time = time.time()

                # Perform inference using Jetson Nano with the Moondream model
                jetson_ollama_response = ollama.chat(
                    model="moondream",
                    messages=[{
                        "role": "user",
                        "content": queries["text"],
                        "images": [queries["image"]]
                    }]
                )

                # Measure processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Extract and structure the response
                jetson_ollama_response_dict = dict(jetson_ollama_response)
                jetson_ollama_message_content = jetson_ollama_response_dict.get('message', {}).get('content', "No content")

                response = {
                    "model": jetson_ollama_response_dict.get('model', "Unknown model"),
                    "message": jetson_ollama_message_content
                }

                print("Response:", response)
                print(f"Processing Time: {processing_time:.2f} seconds")

            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Invalid inference device.")


    except Exception as e:
        logging.error(f"Fatal error: {e}")