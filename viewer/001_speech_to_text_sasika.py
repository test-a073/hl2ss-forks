# AudioToText using SpeechRecognition library

# IMPORT LIBRARIES
from pydub import AudioSegment
import speech_recognition as sr
import time

# Start timing from here
start_time = time.time()

# Get the audio file
# audio_file_path = "sasika_stream/sample_00.wav"
audio_file_path = "sasika_stream/recording_2025-01-07_10-04-37.wav"

# Create a recognizer instance
r = sr.Recognizer()

# Use speech_recognition's own audio file reading method
with sr.AudioFile(audio_file_path) as source:
    # Load the audio into speech_recognition
    audio_data = r.record(source)
    
    # Recognize the audio data
    try:
        text = r.recognize_google(audio_data)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

end_time = time.time()

# Print the time taken in 3 decimal places
print(f"Time taken: {end_time - start_time:.3f} seconds\n")

# Print the recognized text
print(text)