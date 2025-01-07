from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import wave
import os
import datetime
import config

# Settings --------------------------------------------------------------------
# HoloLens address
host = config.HOLOLENS_IP
# Audio encoding profile
profile = hl2ss.AudioProfile.AAC_24000
#------------------------------------------------------------------------------

enable = True
recording = False
audio_data = bytearray()

def on_press(key):
    global enable, recording
    try:
        if hasattr(key, 'char'):
            if key.char == 's':
                print("Recording started...")
                recording = True
            elif key.char == 'q':
                print("Recording stopped. Saving file...")
                recording = False
                
                # Save the audio
                if len(audio_data) > 0:
                    os.makedirs('sasika_stream', exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join('sasika_stream', f'recording_{timestamp}.wav')
                    
                    with wave.open(filename, 'wb') as wave_file:
                        wave_file.setnchannels(hl2ss.Parameters_MICROPHONE.CHANNELS)
                        wave_file.setsampwidth(2)  # 16-bit audio
                        wave_file.setframerate(hl2ss.Parameters_MICROPHONE.SAMPLE_RATE)
                        wave_file.writeframes(audio_data)
                    
                    print(f"Audio saved as '{filename}'")
                    print(f"File size: {os.path.getsize(filename)} bytes")
                else:
                    print("No audio recorded.")
                    
                enable = False
                return False
                
        elif key == keyboard.Key.esc:
            enable = False
            return False
            
    except AttributeError:
        pass
    return True

print("Initializing audio recording...")
print("Controls:")
print("- Press 's' to start recording")
print("- Press 'q' to stop recording and save")
print("- Press 'esc' to exit without saving")

listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
    client.open()
    print("Connected to HoloLens microphone")

    while enable:
        try:
            if recording:
                data = client.get_next_packet()
                # RAW format is s16 packed, AAC decoded format is f32 planar
                audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
                audio_data.extend(audio.tobytes())
                print(f"Recording... Current size: {len(audio_data)} bytes", end='\r')
        except Exception as e:
            print(f"Error processing audio: {e}")

finally:
    print("\nCleaning up...")
    client.close()
    listener.join()
    print("Done!")
