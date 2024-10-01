import os
import pyaudio
import wave
import numpy as np
import scipy.io.wavfile
import time
from datetime import datetime
import speech_recognition as sr
from google.cloud import dialogflow
import serial

credential_path = (r'./key.json')  #change the file if needed
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Settings
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Single channel for microphone
RATE = 44100  # Sample rate (samples per second)
THRESHOLD = 500  # Sound threshold for detecting speech
SILENCE_DURATION = 2  # Silence duration (in seconds) to consider the end of speech

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start a new stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

arduino = serial.Serial(port='/dev/ttyACM0',  baudrate=9600, timeout=.1)  # check your serial port


def is_silent(data):
    """Check if the data chunk is silent."""
    return np.abs(np.frombuffer(data, dtype=np.int16)).mean() < THRESHOLD

def record_audio(filename):
    """Record audio until silence is detected."""
    frames = []
    silence_start_time = None

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        if is_silent(data):
            if silence_start_time is None:
                silence_start_time = time.time()
            elif time.time() - silence_start_time > SILENCE_DURATION:
                break
        else:
            silence_start_time = None

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    
    try:
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio)
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio.")
        return 13
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return 13



def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print("Session path: {}\n".format(session))

    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)

        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        print("=" * 20)
        print("Query text: {}".format(response.query_result.query_text))
        print(
            "Detected intent: {} (confidence: {})\n".format(
                response.query_result.intent.display_name,
                response.query_result.intent_detection_confidence,
            )
        )
        print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))



def main():
    while True:
        print("Listening for speech...")
        data = stream.read(CHUNK)

        if not is_silent(data):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./audios/audio_{timestamp}.wav"
            print(f"Recording to {filename}...")
            record_audio(filename)
            print("Recording stopped.")
            txt = transcribe_audio(filename)
            if (txt == 13):  # Speech not understandable or API exploded
                time.sleep(1)
                continue
            detect_intent_texts("humanrobotcommunication", "1234", [txt], "en-US")
            # arduino.write(bytes(f"{value}\n".encode(),  'utf-8')) # send intent to arduino 
            time.sleep(1)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        arduino.close()