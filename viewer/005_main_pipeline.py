import logging
import os
from datetime import datetime
import multiprocessing as mp
import cv2
from pynput import keyboard
import time
from typing import Optional
import io

import hl2ss
import hl2ss_lnm
import hl2ss_mp
from pydub import AudioSegment
import speech_recognition as sr

# Global state variables
enable = True
is_recording = False
combined_audio = AudioSegment.empty()
current_image_file = None
current_audio_file = None
last_save_time = 0

# Stream components
producer = None
consumer = None
sinks = {}

def initialize_streams(host: str):
    """Initialize PV and microphone streams"""
    global producer, consumer, sinks
    
    # Stream settings
    ports = [
        hl2ss.StreamPort.PERSONAL_VIDEO,
        hl2ss.StreamPort.MICROPHONE
    ]
    
    # PV settings
    pv_width = 760
    pv_height = 428
    pv_framerate = 30
    buffer_elements = 150
    
    # Start PV subsystem
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    
    # Initialize producer
    producer = hl2ss_mp.producer()
    
    # Configure PV stream
    producer.configure(
        hl2ss.StreamPort.PERSONAL_VIDEO,
        hl2ss_lnm.rx_pv(
            host,
            hl2ss.StreamPort.PERSONAL_VIDEO,
            width=pv_width,
            height=pv_height,
            framerate=pv_framerate
        )
    )
    
    # Configure microphone stream
    producer.configure(
        hl2ss.StreamPort.MICROPHONE,
        hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE)
    )
    
    # Initialize consumer and sinks
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    
    # Start both streams
    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()
        while (sinks[port].get_buffered_frame(0)[0] != 0):
            pass
        logging.info(f'Started stream on port {port}')

def process_video(payload, output_folder: str):
    """Process video frames and save periodically"""
    global last_save_time, current_image_file
    
    if payload.image is not None and payload.image.size > 0:
        # Display frame
        cv2.imshow('Personal Video', payload.image)
        
        # Save frame if interval has elapsed (every 20 seconds)
        current_time = time.time()
        if current_time - last_save_time >= 20:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            current_image_file = os.path.join(output_folder, f"frame_{timestamp}.jpg")
            cv2.imwrite(current_image_file, payload.image)
            last_save_time = current_time
            logging.info(f"Saved image: {current_image_file}")

def process_audio(payload):
    """Process audio data during recording"""
    global combined_audio
    
    if is_recording:
        try:
            segment = AudioSegment.from_file(
                io.BytesIO(payload.tobytes()),
                format="raw",
                frame_rate=48000,
                channels=2,
                sample_width=2
            )
            combined_audio += segment
        except Exception as e:
            logging.error(f"Error processing audio: {e}")

def save_recording(output_folder: str):
    """Save the current audio recording"""
    global is_recording, current_audio_file, combined_audio
    
    if len(combined_audio) < 100:  # Minimum length check
        logging.warning("Recording too short, discarding")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_audio_file = os.path.join(output_folder, f"recording_{timestamp}.wav")
    
    try:
        combined_audio.export(current_audio_file, format="wav")
        logging.info(f"Audio saved as {current_audio_file}")
    except Exception as e:
        logging.error(f"Failed to save audio: {e}")
    finally:
        is_recording = False

def transcribe_audio(audio_file: str) -> Optional[str]:
    """Transcribe the audio file using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def on_press(key) -> bool:
    """Handle keyboard events"""
    global enable, is_recording, combined_audio
    
    try:
        if key.char == 'w':  # Start recording
            if not is_recording:
                logging.info("Recording started...")
                is_recording = True
                combined_audio = AudioSegment.empty()
        
        elif key.char == 'q':  # Stop recording
            if is_recording:
                save_recording(OUTPUT_FOLDER)
                enable = False
                return False  # Stop listener
                
    except AttributeError:
        pass
    return True

def cleanup():
    """Clean up resources"""
    # Stop streams
    for port in sinks:
        try:
            sinks[port].detach()
            producer.stop(port)
            logging.info(f'Stopped stream on port {port}')
        except Exception as e:
            logging.error(f"Error stopping stream {port}: {e}")

    # Stop PV subsystem
    try:
        hl2ss_lnm.stop_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO)
    except Exception as e:
        logging.error(f"Error stopping PV subsystem: {e}")

    # Close windows
    cv2.destroyAllWindows()

def main():
    try:
        # Ensure output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Initialize streams
        initialize_streams(HOST)
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logging.info("Press 'W' to start recording, 'Q' to stop")

        # Main loop
        while enable:
            # Process video
            _, video_data = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame()
            if video_data is not None:
                process_video(video_data.payload, OUTPUT_FOLDER)

            # Process audio separately
            if is_recording:
                _, audio_data = sinks[hl2ss.StreamPort.MICROPHONE].get_most_recent_frame()
                if audio_data is not None:
                    process_audio(audio_data.payload)

            cv2.waitKey(1)

        listener.join()
        
        # Process final recording if exists
        text = None
        if current_audio_file:
            text = transcribe_audio(current_audio_file)
            
        return {
            "text": text,
            "image": current_image_file
        }

    except Exception as e:
        logging.error(f"Error during recording: {e}")
        return None
    finally:
        cleanup()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load configuration
    try:
        import config
        HOST = config.HOLOLENS_IP
        OUTPUT_FOLDER = "sasika_stream"
    except ImportError:
        logging.error("Could not load config.py")
        HOST = input("Please enter HoloLens IP address: ")
        OUTPUT_FOLDER = "sasika_stream"
    
    # Run main program
    try:
        queries = main()
        print(queries)
    except Exception as e:
        logging.error(f"Fatal error: {e}")