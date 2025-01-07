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
from pymemcache.client import base

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.10.149"

# Audio encoding profile
# profile = hl2ss.AudioProfile.AAC_24000
profile = hl2ss.AudioProfile.RAW
mem_client = base.Client(('localhost', 11211))
#------------------------------------------------------------------------------

# RAW format is s16 packed, AAC decoded format is f32 planar
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
enable = True
r = sr.Recognizer()




def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    mem_client.set('instruction', '')
    mem_client.set('process', 0)

    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()

print('Press esc to stop')
combined = AudioSegment.empty()
state = 0
transcribe = False

while (enable):
    state = mem_client.get('state').decode('utf-8')
    # state byte to string
    
    if (state == '1' ):
        transcribe = True
        data = client.get_next_packet()
        # RAW format is s16 packed, AAC decoded format is f32 planar

        audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload

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

    else:
        # check if there is any audio to be transcribed

        if (transcribe):
            combined.export("audio.wav", format="wav")
            with sr.AudioFile('audio.wav') as source:
                # Read the audio file
                audio_file = r.record(source)
                try:
                    text = r.recognize_google(audio_file)
                    # replace start with empty string
                    text = text.replace('start', '')
                    text = text.replace('done', '')
                    text = text.replace('yes', '')

                    print("Your instruction is: ",text)
                except sr.UnknownValueError:
                    print('unknown error')
                    continue
                # text = r.recognize_google(audio_file)
                # print(text)
            combined = AudioSegment.empty()
            mem_client.set('instruction', text) 
            mem_client.set('process', 1)

            transcribe = False

        # if (n == 100):
        #     combined.export("audio.wav", format="wav")
        #     print('exported')
            

        #     # with sr.AudioFile('audio.wav') as source:
        #         # Read the audio file
        #         # audio_file = r.record(source)
        #         # try:
        #             # text = r.recognize_google(audio_file)
        #             # print(text)
        #         # except sr.UnknownValueError:
        #             # print('unknown error')
        #             # continue
        #         # text = r.recognize_google(audio_file)
        #         # print(text)
        #     n = 0
        #     combined = AudioSegment.empty()
        # n +=1


client.close()

enable = False
# pcmqueue.put(b'')
# thread.join()
listener.join()
