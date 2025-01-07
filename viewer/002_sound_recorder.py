#------------------------------------------------------------------------------
# This script captures microphone audio from the HoloLens.
# Press 'W' to start recording and 'Q' to stop recording and save the audio to the "sasika_stream" folder.
# The saved file will be named in the format recording_<date_time>.wav.
# Audio stream configuration is fixed to 2 channels, 48000 Hz.
# Press 'Esc' to stop the script entirely.
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

    if key == keyboard.Key.esc:
        enable = False  # Stop the script
        logging.info("Esc key pressed. Stopping the script.")
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
