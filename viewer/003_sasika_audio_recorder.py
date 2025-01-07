import io
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import pyaudio
from pynput import keyboard
import config
import datetime
import os
import wave
import numpy as np
import sys

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
profile = hl2ss.AudioProfile.RAW

# Variables to manage recording
enable = True
recording = False
audio_data = bytearray()

def on_press(key):
    global enable, recording
    try:
        if key.char == 's':
            print("Recording started...")
            recording = True
        elif key.char == 'q':
            print("Recording stopped. Saving file...")
            recording = False
            enable = False
            
            # Save the audio
            if len(audio_data) > 0:
                os.makedirs('sasika_stream', exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('sasika_stream', f'recording_{timestamp}.wav')
                
                with wave.open(filename, 'wb') as wave_file:
                    wave_file.setnchannels(2)
                    wave_file.setsampwidth(2)
                    wave_file.setframerate(48000)
                    wave_file.writeframes(audio_data)
                
                print(f"Audio saved as '{filename}'")
                print(f"File size: {os.path.getsize(filename)} bytes")
            else:
                print("No audio recorded.")
            
            # Force exit the program
            sys.exit(0)
    except AttributeError:
        pass
    except Exception as e:
        print(f"Error in key handler: {e}")
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    print(f"Connecting to HoloLens at {host}...")
    client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
    client.open()
    print("Successfully connected to HoloLens")
    print("Press 's' to start recording, 'q' to stop and save.")

    while enable:
        if recording:
            try:
                print("Waiting for audio packet...")  # Debug print
                data = client.get_next_packet()
                print(f"Received packet with payload size: {len(data.payload)}")  # Debug print
                
                audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
                print(f"Processed audio chunk size: {len(audio)} bytes")  # Debug print
                
                # If audio is numpy array, convert to bytes
                if isinstance(audio, np.ndarray):
                    audio_bytes = audio.tobytes()
                else:
                    audio_bytes = audio
                    
                audio_data.extend(audio_bytes)
                print(f"Total audio data collected: {len(audio_data)} bytes")  # Debug print
                
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
        else:
            # Small sleep when not recording to prevent CPU spinning
            import time
            time.sleep(0.1)

except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    client.close()
    listener.join(timeout=1)  # Wait for listener with timeout
    print("Cleanup complete")
