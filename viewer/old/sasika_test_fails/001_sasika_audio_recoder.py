import io
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import pyaudio
from pynput import keyboard
from pydub import AudioSegment
import config
import datetime
import os
import numpy as np

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
profile = hl2ss.AudioProfile.RAW
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32

# Variables to manage recording
enable = True
recording = False
combined = AudioSegment.empty()

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
            if not combined.empty():
                print(f"Audio length: {len(combined)} ms")  # Debug print
                os.makedirs('sasika_stream', exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('sasika_stream', f'recording_{timestamp}.wav')
                combined.export(filename, format="wav")
                print(f"Audio saved as '{filename}'")
            else:
                print("No audio recorded.")
    except AttributeError:
        pass  # Ignore special keys
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()
print("Press 's' to start recording, 'q' to stop and save.")

buffer_count = 0  # Debug counter
while enable:
    if recording:
        data = client.get_next_packet()
        audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
        
        # Debug prints
        print(f"Received audio chunk size: {len(audio)}")
        print(f"Audio data type: {type(audio)}")
        
        # Convert numpy array to bytes properly
        if isinstance(audio, np.ndarray):
            audio_bytes = audio.tobytes()
        else:
            audio_bytes = audio

        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="raw",
                frame_rate=48000,
                channels=2,
                sample_width=2
            )
            combined += audio_segment
            buffer_count += 1
            if buffer_count % 10 == 0:  # Print every 10 buffers
                print(f"Processed {buffer_count} buffers. Current length: {len(combined)} ms")
        except Exception as e:
            print(f"Error processing audio chunk: {e}")

client.close()
listener.join()
