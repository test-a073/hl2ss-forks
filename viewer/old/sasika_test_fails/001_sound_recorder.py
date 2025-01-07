#------------------------------------------------------------------------------
# This script captures microphone audio from the HoloLens and plays it.
# Press 'W' to start recording and 'Q' to stop recording and save the audio to a .wav file.
# Audio stream configuration is fixed to 2 channels, 48000 Hz.
# Press 'Esc' to stop the script entirely.
#------------------------------------------------------------------------------

from pynput import keyboard
from pydub import AudioSegment

import io
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import pyaudio
import queue
import threading
import config

# Settings --------------------------------------------------------------------

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
profile = hl2ss.AudioProfile.RAW

#------------------------------------------------------------------------------ 

audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
enable = True
is_recording = False
combined_audio = AudioSegment.empty()

def pcmworker(pcmqueue):
    global enable
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=hl2ss.Parameters_MICROPHONE.CHANNELS, 
                    rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    
    while enable:
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def on_press(key):
    global enable, is_recording, combined_audio

    if key == keyboard.Key.esc:
        enable = False  # Stop the script
        return False    # Exit listener

    try:
        if key.char == 'w':  # Start recording
            print("Recording started...")
            is_recording = True
            combined_audio = AudioSegment.empty()
        
        elif key.char == 'q':  # Stop recording
            if is_recording:
                print("Recording stopped. Saving audio...")
                combined_audio.export("recorded_audio.wav", format="wav")
                print("Audio saved as 'recorded_audio.wav'")
                is_recording = False

    except AttributeError:
        pass

    return enable

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()

print('Press W to start recording, Q to stop recording, and Esc to quit.')
while enable: 
    data = client.get_next_packet()
    audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
    pcmqueue.put(audio.tobytes())

    # Only process audio for recording when 'is_recording' is True
    if is_recording:
        segment = AudioSegment.from_file(io.BytesIO(audio.tobytes()), 
                                         format="raw",
                                         frame_rate=48000,
                                         channels=2,
                                         sample_width=2)
        combined_audio += segment

client.close()
enable = False
pcmqueue.put(b'')
thread.join()
listener.join()
