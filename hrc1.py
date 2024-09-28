'''
import os
from google.cloud import dialogflow

credential_path = (r'./key.json')  #change the file if needed
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def detect_intent_stream(project_id, session_id, audio_file_path, language_code):
    """Returns the result of detect intent with streaming audio as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 16000

    session_path = session_client.session_path(project_id, session_id)
    print("Session path: {}\n".format(session_path))

    def request_generator(audio_config, audio_file_path):
        query_input = dialogflow.QueryInput(audio_config=audio_config)

        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            session=session_path, query_input=query_input
        )

        # Here we are reading small chunks of audio data from a local
        # audio file.  In practice these chunks should come from
        # an audio input device.
        with open(audio_file_path, "rb") as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                # The later requests contains audio data.
                yield dialogflow.StreamingDetectIntentRequest(input_audio=chunk)

    audio_config = dialogflow.InputAudioConfig(
        audio_encoding=audio_encoding,
        language_code=language_code,
        sample_rate_hertz=sample_rate_hertz,
    )

    requests = request_generator(audio_config, audio_file_path)
    responses = session_client.streaming_detect_intent(requests=requests)

    print("=" * 20)
    for response in responses:
        print(
            'Intermediate transcript: "{}".'.format(
                response.recognition_result.transcript
            )
        )
        # Note: Since Python gRPC doesn't have closeSend method, to stop processing the audio after result is recognized,
        # you may close the channel manually to prevent further iteration.
        # Keep in mind that if there is a silence chunk in the audio, part after it might be missed because of early teardown.
        # https://cloud.google.com/dialogflow/es/docs/how/detect-intent-stream#streaming_basics
        if response.recognition_result.is_final:
            session_client.transport.close()
            break

    # Note: The result from the last response is the final transcript along
    # with the detected content.
    query_result = response.query_result

    print("=" * 20)
    print("Query text: {}".format(query_result.query_text))
    print(
        "Detected intent: {} (confidence: {})\n".format(
            query_result.intent.display_name, query_result.intent_detection_confidence
        )
    )
    print("Fulfillment text: {}\n".format(query_result.fulfillment_text))




detect_intent_stream("innate-vigil-434613-v1", "12345", "What time is it", "en-US")  # change project id
'''


import os
import pyaudio
import wave
import numpy as np
import scipy.io.wavfile
import time
from datetime import datetime
import speech_recognition as sr
from google.cloud import dialogflow

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
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")



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



#detect_intent_texts("innate-vigil-434613-v1", "1234", ["hey"], "en-US")


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
            detect_intent_texts("innate-vigil-434613-v1", "1234", [txt], "en-US")
            break



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()