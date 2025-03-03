import speech_recognition as sr
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env file

# Gemini API setup (replace with your actual API key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL") # Replace with the correct endpoint

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp')

for model_name in genai.list_models():
  if 'generateContent' in model_name.supported_generation_methods:
    print(model_name)
          
          
import os
import google.generativeai as genai


def load_file_content(file_path):
  try:
    with open(file_path, 'r') as f:
      return f.read()
  except Exception as e:
      print(f"Error loading file: {e}")
      return None


def interact_with_gemini_model_with_file(file_content, api_key, model_name="models/gemini-2.0-flash-thinking-exp"):
    global model
    try:

        # You will need to adjust this part based on the exact method your desired
        #  model has and whether it supports a structured context as a message or
        # if it accepts text directly
        response = model.generate_content(
            contents = [
            { "role": "user", "parts": [file_content]}
            ]
        )
        return response.text

    except Exception as e:
        print(f"Error communicating with genai: {e}")
        return None
  


def record_audio():
    """Records audio from the microphone and returns the audio data."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    return audio

def transcribe_audio(audio):
    """Transcribes audio to text using Google Web Speech API."""
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return None

def send_to_gemini(text):
    try:
        response = model.generate_content(text)
        #response = requests.post(GEMINI_API_URL, headers=headers, json=data) # Use json=data to send JSON
        print("Gemini Response:", response.text)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        if response is not None:
            print(f"Response content: {response.text}") # Print the error response from the API
        return None

if __name__ == "__main__":
    while 1==1:
        audio = record_audio()
        if audio:
            transcribed_text = transcribe_audio(audio)
            if transcribed_text:
                send_to_gemini(transcribed_text)