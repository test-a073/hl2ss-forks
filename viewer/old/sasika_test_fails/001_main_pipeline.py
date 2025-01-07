# CONFIGURATION 
transcription_file_path = "sasika_stream/transcription.txt"
audio_recording_file_path = "sasika_stream/recording_2025-01-07_11-17-52.wav"

#------------------------------------------------------------------------------ 
# 1. GET THE AUDIO DATA FROM THE HOLOLENS MICROPHONE
#------------------------------------------------------------------------------ 
import logging
from pynput import keyboard
from pydub import AudioSegment
import io
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import os
from datetime import datetime
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Settings --------------------------------------------------------------------

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
profile = hl2ss.AudioProfile.RAW

#------------------------------------------------------------------------------ 

enable = True
is_recording = False
combined_audio = AudioSegment.empty()
output_folder = "sasika_stream"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
logging.info(f"Output folder '{output_folder}' is ready for storing audio files.")

def on_press(key):
    global enable, is_recording, combined_audio

    try:
        if key.char == 'w':  # Start recording
            logging.info("Recording started...")
            is_recording = True
            combined_audio = AudioSegment.empty()
        
        elif key.char == 'q':  # Stop recording
            if is_recording:
                # Generate filename with date and time
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"recording_{timestamp}.wav"
                filepath = os.path.join(output_folder, filename)
                
                # Save the audio file
                logging.info(f"Recording stopped. Saving audio to {filepath}...")
                combined_audio.export(filepath, format="wav")
                logging.info(f"Audio saved as {filepath}")
                is_recording = False
                enable = False
                return False  # Stop the listener after saving the audio

    except AttributeError:
        pass

    return True  # Continue listening

listener = keyboard.Listener(on_press=on_press)
listener.start()
logging.info("Keyboard listener started. Press 'W' to start recording, 'Q' to stop recording, 'Esc' to quit.")

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()
logging.info(f"Connected to HoloLens at {host}.")

while enable: 
    data = client.get_next_packet()
    audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload

    # Only process audio for recording when 'is_recording' is True
    if is_recording:
        segment = AudioSegment.from_file(io.BytesIO(audio.tobytes()), 
                                         format="raw",
                                         frame_rate=48000,
                                         channels=2,
                                         sample_width=2)
        combined_audio += segment

logging.info("Audio capture completed.")
client.close()
logging.info("HoloLens microphone client closed.")
listener.join()
logging.info("Keyboard listener stopped.")


#------------------------------------------------------------------------------ 
# 2. CONVERT THE AUDIO FILE TO TEXT PROMPT
#------------------------------------------------------------------------------ 
import speech_recognition as sr
import time

# Start timing from here
start_time = time.time()

# Get the audio file
audio_file_path = filepath
logging.info(f"Attempting to transcribe the audio file: {audio_file_path}")

# Create a recognizer instance
r = sr.Recognizer()

# Use speech_recognition's own audio file reading method
with sr.AudioFile(audio_file_path) as source:
    # Load the audio into speech_recognition
    audio_data = r.record(source)
    
    # Recognize the audio data
    try:
        text_prompt = r.recognize_google(audio_data)  # Changed variable name to text_prompt
        logging.info(f"Transcription successful: {text_prompt}")
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        text_prompt = ""  # Added to handle the error case
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        text_prompt = ""  # Added to handle the error case

end_time = time.time()

# Log the time taken for recognition
logging.info(f"Time taken for transcription: {end_time - start_time:.3f} seconds")

#------------------------------------------------------------------------------ 
# 3. DISPLAY THE TEXT PROMPT AND APPEND IT TO A TEXT FILE IN A NEW LINE
#------------------------------------------------------------------------------ 
if text_prompt:
    # Get current timestamp for the log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write to file if we have text to write
    with open(transcription_file_path, "a") as f:
        f.write(f"\n[{timestamp}] {text_prompt}")
        logging.info("Text prompt appended to transcription.txt")
else:
    logging.info("No text to append due to recognition error")
