#------------------------------------------------------------------------------
# This script receives microphone audio from the HoloLens and plays it. The 
# main thread receives the data, decodes it, and puts the decoded audio samples
# in a queue. A second thread gets the samples from the queue and plays them.
# Audio stream configuration is fixed to 2 channels, 48000 Hz.
# Press esc to stop.
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
import speech_recognition as sr
import config
# Settings --------------------------------------------------------------------

# HoloLens address
host = config.HOLOLENS_IP

# Audio encoding profile
# profile = hl2ss.AudioProfile.AAC_24000
profile = hl2ss.AudioProfile.RAW

#------------------------------------------------------------------------------

# RAW format is s16 packed, AAC decoded format is f32 planar
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
enable = True
r = sr.Recognizer()

def pcmworker(pcmqueue):
    global enable
    global audio_format
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=hl2ss.Parameters_MICROPHONE.CHANNELS, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    
    while (enable):
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
# thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()

print('Press esc to stop')
combined = AudioSegment.empty()
n = 0
while (enable): 
    data = client.get_next_packet()
    # RAW format is s16 packed, AAC decoded format is f32 planar
    audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
    pcmqueue.put(audio.tobytes())
    audio = AudioSegment.from_file(io.BytesIO(audio.tobytes()), 
                                format="raw",
                                frame_rate=48000,
                                channels=2,
                                sample_width=2)
    # audio = audio.set_frame_rate(16000).set_channels(1)

    combined += audio


    # wav_data = io.BytesIO()
    # wav_data = 'audio.wav'
    # audio.export(wav_data, format="wav")
    # wav_data.seek(0)

    if (n == 100):
        combined.export("audio.wav", format="wav")
        print('exported')
        

        # with sr.AudioFile('audio.wav') as source:
            # Read the audio file
            # audio_file = r.record(source)
            # try:
                # text = r.recognize_google(audio_file)
                # print(text)
            # except sr.UnknownValueError:
                # print('unknown error')
                # continue
            # text = r.recognize_google(audio_file)
            # print(text)
        n = 0
        combined = AudioSegment.empty()
    n +=1


client.close()

enable = False
pcmqueue.put(b'')
thread.join()
listener.join()
