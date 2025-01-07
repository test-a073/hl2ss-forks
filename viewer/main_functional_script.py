import logging
import os
from datetime import datetime
from contextlib import contextmanager
import io
import time
import json
import cv2
import multiprocessing as mp
import sys

import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import hl2ss_mp
from pydub import AudioSegment
from pynput import keyboard
import speech_recognition as sr

# Global state variables
enable = True
is_recording = False
combined_audio = AudioSegment.empty()
current_audio_file = None
queries = []
current_image_path = None
current_text = None
last_image_save_time = 0

# Constants
PV_WIDTH = 760
PV_HEIGHT = 428
PV_FRAMERATE = 30
BUFFER_ELEMENTS = 150
IMAGE_SAVE_INTERVAL = 20

def setup_logging(output_folder):
    """Set up logging configuration with both file and console handlers"""
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, f"recorder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")

def validate_directories(output_folder, transcription_file):
    """Validate and create necessary directories"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        transcription_dir = os.path.dirname(transcription_file)
        if transcription_dir:
            os.makedirs(transcription_dir, exist_ok=True)
        
        # Test if transcription file is writable
        with open(transcription_file, 'a') as f:
            pass
        
        logging.info(f"Directories validated. Output folder: {output_folder}")
        logging.debug(f"Transcription file location: {transcription_file}")
    except Exception as e:
        logging.error(f"Directory validation failed: {e}")
        raise

def save_image(output_folder, image):
    """Save the current frame if interval has elapsed"""
    global last_image_save_time, current_image_path
    
    current_time = time.time()
    if current_time - last_image_save_time >= IMAGE_SAVE_INTERVAL:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        
        try:
            cv2.imwrite(image_path, image)
            last_image_save_time = current_time
            current_image_path = image_path
            logging.info(f"Frame saved: {image_path}")
            logging.debug(f"Frame dimensions: {image.shape}")
        except Exception as e:
            logging.error(f"Failed to save frame: {e}")

def process_video_frame(output_folder, payload, show_image=False):
    """Process and save video frames"""
    if payload.image is not None and payload.image.size > 0:
        logging.debug(f"Processing video frame. Size: {payload.image.size}")
        save_image(output_folder, payload.image)
        if show_image:
            try:
                cv2.imshow('Personal Video', payload.image)
            except Exception as e:
                logging.error(f"Failed to display frame: {e}")

def save_recording(output_folder):
    """Save the current audio recording"""
    global combined_audio, is_recording, current_audio_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recording_{timestamp}.wav"
    current_audio_file = os.path.join(output_folder, filename)
    
    try:
        logging.info(f"Saving audio recording: {filename}")
        logging.debug(f"Audio duration: {len(combined_audio)}ms")
        combined_audio.export(current_audio_file, format="wav")
        logging.info("Audio saved successfully")
    except Exception as e:
        logging.error(f"Failed to save audio: {e}")
        raise
    finally:
        is_recording = False
        combined_audio = AudioSegment.empty()

def on_press(key, output_folder):
    """Handle keyboard input"""
    global is_recording, enable, combined_audio
    
    try:
        if key.char == 'w':  # Start recording
            if not is_recording:
                logging.info("Recording started")
                is_recording = True
                combined_audio = AudioSegment.empty()
        
        elif key.char == 'q':  # Stop recording
            if is_recording:
                logging.info("Recording stopped")
                save_recording(output_folder)
                enable = False
                return False  # Stop listener
                
    except AttributeError:
        pass
    return True

def transcribe_audio(audio_file, timeout=30):
    """Transcribe audio file with timeout"""
    start_time = time.time()
    
    logging.info(f"Starting audio transcription: {audio_file}")
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file) as source:
            logging.debug("Reading audio file")
            audio_data = recognizer.record(source)
        
        logging.debug("Sending audio for transcription")
        text = recognizer.recognize_google(audio_data)
        
        if time.time() - start_time > timeout:
            logging.warning("Transcription timeout exceeded")
            raise Exception("Transcription timeout")
        
        logging.info("Transcription completed successfully")
        logging.debug(f"Transcribed text: {text}")
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

def save_queries(output_folder):
    """Save queries to a JSON file"""
    queries_file = os.path.join(output_folder, "queries.json")
    try:
        logging.info("Saving queries to JSON")
        logging.debug(f"Number of queries: {len(queries)}")
        with open(queries_file, 'w') as f:
            json.dump(queries, f, indent=2)
        logging.info(f"Queries saved to {queries_file}")
    except Exception as e:
        logging.error(f"Failed to save queries: {e}")
        raise

def append_transcription(transcription_file, text):
    """Append transcription to file and update queries"""
    global current_text, queries
    
    if not text:
        logging.warning("Empty transcription text received")
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logging.debug(f"Appending transcription: {text[:50]}...")
        with open(transcription_file, "a") as f:
            f.write(f"\n[{timestamp}] {text}")
        
        current_text = text
        if current_image_path:
            query = {
                'image_path': current_image_path,
                'transcription': text,
                'timestamp': timestamp
            }
            queries.append(query)
            logging.info(f"Query added: Image: {os.path.basename(current_image_path)}")
            logging.debug(f"Full query: {query}")
            
    except Exception as e:
        logging.error(f"Failed to append transcription: {e}")
        raise

def cleanup(host, producer, sinks):
    """Cleanup resources"""
    logging.info("Starting cleanup process")
    try:
        for port in sinks:
            logging.debug(f"Detaching sink for port {port}")
            sinks[port].detach()
            producer.stop(port)
            logging.info(f'Stopped stream on port {port}')

        logging.debug("Stopping PV subsystem")
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()
        logging.info("Cleanup completed successfully")
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        raise

def initialize_streams(host):
    """Initialize both audio and video streams"""
    logging.info("Initializing streams")
    try:
        logging.debug("Starting PV subsystem")
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        
        logging.debug("Configuring producer")
        producer = hl2ss_mp.producer()
        producer.configure(
            hl2ss.StreamPort.PERSONAL_VIDEO,
            hl2ss_lnm.rx_pv(
                host,
                hl2ss.StreamPort.PERSONAL_VIDEO,
                width=PV_WIDTH,
                height=PV_HEIGHT,
                framerate=PV_FRAMERATE
            )
        )
        producer.configure(
            hl2ss.StreamPort.MICROPHONE,
            hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE)
        )

        logging.debug("Creating consumer")
        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        
        sinks = {}
        for port in [hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss.StreamPort.MICROPHONE]:
            logging.debug(f"Initializing port {port}")
            producer.initialize(port, BUFFER_ELEMENTS)
            producer.start(port)
            sinks[port] = consumer.create_sink(producer, port, manager, None)
            sinks[port].get_attach_response()
            while (sinks[port].get_buffered_frame(0)[0] != 0):
                pass
            logging.info(f'Stream started on port {port}')
            
        return producer, consumer, sinks
        
    except Exception as e:
        logging.error(f"Failed to initialize streams: {e}")
        raise

def process_inference(queries_data):
    """Process inference on either workstation or Jetson Nano"""
    logging.info("Starting inference processing")
    
    def policy_function() -> int:
        """Returns the device for inference"""
        return 0  # Always return workstation for testing
    
    inference_device = policy_function()
    logging.info(f"Selected inference device: {'Workstation' if inference_device == 0 else 'Jetson Nano'}")

    if inference_device == 0:
        return process_workstation_inference(queries_data)
    elif inference_device == 1:
        return process_jetson_inference(queries_data)
    else:
        logging.error("Invalid inference device")
        return None

def process_workstation_inference(queries_data):
    """Process inference on workstation"""
    import requests
    
    server_url = "http://10.4.16.46:5000/query"
    image_path = queries_data["image"]
    query_text = queries_data["text"]

    try:
        logging.info("Starting workstation inference")
        start_time = time.time()
        
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'query': query_text}
            logging.debug(f"Sending request to {server_url}")
            response = requests.post(server_url, files=files, data=data)

        processing_time = time.time() - start_time
        logging.info(f"Inference completed in {processing_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json().get('response', "No response field")
            logging.info("Inference successful")
            return result
        else:
            logging.error(f"Server error: {response.json()}")
            return None

    except Exception as e:
        logging.error(f"Workstation inference error: {e}")
        return None

def process_jetson_inference(queries_data):
    """Process inference on Jetson Nano"""
    import ollama
    
    try:
        logging.info("Starting Jetson Nano inference")
        start_time = time.time()
        
        jetson_ollama_response = ollama.chat(
            model="moondream",
            messages=[{
                "role": "user",
                "content": queries_data["text"],
                "images": [queries_data["image"]]
            }]
        )

        processing_time = time.time() - start_time
        logging.info(f"Inference completed in {processing_time:.2f} seconds")

        response_dict = dict(jetson_ollama_response)
        message_content = response_dict.get('message', {}).get('content', "No content")
        
        response = {
            "model": response_dict.get('model', "Unknown model"),
            "message": message_content
        }
        
        logging.debug(f"Inference response: {response}")
        return response

    except Exception as e:
        logging.error(f"Jetson inference error: {e}")
        return None

def run_recorder(host, output_folder, transcription_file):
    """Main execution function"""
    global enable, is_recording, combined_audio, queries, current_text
    
    try:
        setup_logging(output_folder)
        validate_directories(output_folder, transcription_file)
        
        logging.info(f"Starting recorder. Host: {host}")
        producer, consumer, sinks = initialize_streams(host)
        
        listener = keyboard.Listener(on_press=lambda key: on_press(key, output_folder))
        listener.start()
        logging.info("Keyboard listener started. Press 'W' to start recording, 'Q' to stop")

        while enable:
            _, video_data = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame()
            if video_data is not None:
                process_video_frame(output_folder, video_data.payload, show_image=True)

            if is_recording:
                _, audio_data = sinks[hl2ss.StreamPort.MICROPHONE].get_most_recent_frame()
                if audio_data is not None:
                    audio = audio_data.payload
                    segment = AudioSegment.from_file(
                        io.BytesIO(audio.tobytes()),
                        format="raw",
                        frame_rate=48000,
                        channels=2,
                        sample_width=2
                    )
                    combined_audio += segment

            cv2.waitKey(1)

        listener.join()
        logging.info("Keyboard listener stopped")
        
        # Process final recording if exists
        current_audio_file = "sasika_stream/recording_2025-01-07_13-24-55.wav"
        if current_audio_file:
            logging.info("Processing final recording")
            text = transcribe_audio(current_audio_file)
            if text:
                current_text = text
                append_transcription(transcription_file, text)

    except Exception as e:
        logging.error(f"Error during recording: {e}", exc_info=True)
        raise
    finally:
        cleanup(host, producer, sinks)
        save_queries(output_folder)
        logging.info("Recording session completed")
        
    return {"text": current_text, "image": current_image_path}

if __name__ == "__main__":
    try:
        # Try to load configuration
        try:
            import config
            HOLOLENS_IP = config.HOLOLENS_IP
            logging.info(f"Loaded HoloLens IP from config: {HOLOLENS_IP}")
        except ImportError:
            logging.warning("Could not load config.py")
            HOLOLENS_IP = input("Please enter HoloLens IP address: ")
            logging.info(f"Manual HoloLens IP input: {HOLOLENS_IP}")
        
        # Define output paths
        output_folder = "sasika_stream"
        transcription_file = os.path.join(output_folder, "transcription.txt")
        
        # Run recorder
        logging.info("Starting HoloLens recorder application")
        queries_data = run_recorder(
            host=HOLOLENS_IP,
            output_folder=output_folder,
            transcription_file=transcription_file
        )
        
        logging.info("Processing queries with inference")
        inference_result = process_inference(queries_data)
        
        if inference_result:
            logging.info("Inference completed successfully")
            print("Inference result:", inference_result)
        else:
            logging.error("Inference processing failed")
            print("Failed to process inference")

    except KeyboardInterrupt:
        logging.info("Application terminated by user")
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        logging.info("Application shutdown complete")
            