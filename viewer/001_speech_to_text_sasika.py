# AudioToText using SpeechRecognition library

# IMPORT LIBRARIES
import logging
from pydub import AudioSegment
import speech_recognition as sr
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Start timing from here
start_time = time.time()

# Get the audio file
audio_file_path = "sasika_stream/recording_2025-01-07_10-04-37.wav"  # Example file, change as needed
logging.info(f"Attempting to transcribe the audio file: {audio_file_path}")

# Create a recognizer instance
r = sr.Recognizer()

# Use speech_recognition's own audio file reading method
try:
    with sr.AudioFile(audio_file_path) as source:
        # Load the audio into speech_recognition
        audio_data = r.record(source)
        
        # Recognize the audio data
        text = r.recognize_google(audio_data)
        logging.info("Transcription successful")
except sr.UnknownValueError:
    logging.error("Google Speech Recognition could not understand audio")
    text = ""  # Handle error case
except sr.RequestError as e:
    logging.error(f"Could not request results from Google Speech Recognition service; {e}")
    text = ""  # Handle error case

end_time = time.time()

# Log the time taken for recognition
logging.info(f"Time taken for transcription: {end_time - start_time:.3f} seconds")

# Print the recognized text
if text:
    logging.info(f"Recognized Text: {text}")
else:
    logging.info("No text recognized")

# Optionally, print the recognized text (if needed for your use case)
print(text)
