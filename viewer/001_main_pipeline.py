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

# Settings --------------------------------------------------------------------

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
profile = hl2ss.AudioProfile.RAW

#------------------------------------------------------------------------------ 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Enable script and recording
enable = True
is_recording = False
combined_audio = AudioSegment.empty()
output_folder = "sasika_stream"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def on_press(key):
    global enable, is_recording, combined_audio

    if key == keyboard.Key.esc:
        enable = False  # Stop the script
        return False    # Exit listener

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

    except AttributeError:
        pass

    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Connect to HoloLens
logging.info(f"Connecting to HoloLens at IP: {host}")
client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()
logging.info(f"Connected to HoloLens at {host}.")

# Recording loop
logging.info('Press W to start recording, Q to stop recording, and Esc to quit.')
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

# Close the client connection
client.close()

# Stop the listener after recording is stopped
listener.stop()

#------------------------------------------------------------------------------ 
# 2. CONVERT THE AUDIO FILE TO TEXT PROMPT
#------------------------------------------------------------------------------ 
import speech_recognition as sr
import time

# Start timing from here
start_time = time.time()

# Get the audio file
audio_file_path = "sasika_stream/recording_2025-01-07_10-04-37.wav"  # Example file, change as needed
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
    with open("transcription.txt", "a") as f:
        f.write(f"\n[{timestamp}] {text_prompt}")
        logging.info("Text prompt appended to transcription.txt")
else:
    logging.info("No text to append due to recognition error")
