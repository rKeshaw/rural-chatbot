# tts_handler.py (Improved Configuration)
import os
import random
import io
import httpx
import pysbd
import numpy as np
import soundfile as sf
from threading import Thread
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from piper.voice import PiperVoice

load_dotenv()

# --- ⚙️ LOCAL TTS CONFIGURATION ---
# Change these values to switch local voices.
# Ensure the filenames in the 'local_tts_models' folder match EXACTLY.
SPEAKER = "rohan"
QUALITY = "medium"
# ------------------------------------


# --- BaseTTSProvider, ElevenLabsProvider, OpenAITTSProvider Classes (Unchanged) ---
class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes: pass
class ElevenLabsProvider(BaseTTSProvider):
    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"
    def synthesize(self, text: str) -> bytes:
        if not self.api_key: raise ValueError("ElevenLabs key not found.")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = { "Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": self.api_key }
        data = { "text": text, "model_id": "eleven_multilingual_v2" }
        response = httpx.post(url, json=data, headers=headers, timeout=20.0)
        response.raise_for_status()
        return response.content
class OpenAITTSProvider(BaseTTSProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def synthesize(self, text: str) -> bytes:
        if not self.client.api_key: raise ValueError("OpenAI key not found.")
        response = self.client.audio.speech.create(model="tts-1", voice="nova", input=text)
        return response.content
# ---------------------------------------------------------------------------------


# --- New Local Piper Provider (Unchanged) ---
class PiperProvider(BaseTTSProvider):
    def __init__(self, model_path, config_path):
        print("Initializing local Piper TTS Provider...")
        self.voice = PiperVoice.load(model_path=model_path, config_path=config_path)
        self.sample_rate = self.voice.config.sample_rate
        print("✅ Piper TTS Provider initialized successfully.")
    def synthesize(self, text: str) -> bytes:
        audio_chunks = self.voice.synthesize(text)
        audio_samples = np.concatenate([chunk.audio_float_array for chunk in audio_chunks])
        buffer = io.BytesIO()
        sf.write(buffer, audio_samples, self.sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()

# --- Unified TTS Handler (Now uses config from top of file) ---
class TTSHandler:
    def __init__(self):
        self.providers = self._initialize_providers()
        if not self.providers:
            print("⚠️ No TTS providers could be initialized.")
        else:
            print(f"✅ TTS Handler initialized with {len(self.providers)} providers. Default is '{self.providers[0].__class__.__name__}'.")

    def _initialize_providers(self):
        provider_instances = []
        try:
            MODEL_FOLDER = "local_tts_models"
            # This line now uses the variables from the top of the file
            MODEL_FILE = f"hi_IN-{SPEAKER}-{QUALITY}.onnx"
            MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILE)
            JSON_PATH = f"{MODEL_PATH}.json"
            
            if os.path.exists(MODEL_PATH) and os.path.exists(JSON_PATH):
                provider_instances.append(PiperProvider(MODEL_PATH, JSON_PATH))
            else:
                print(f"⚠️ Piper model files not found for '{SPEAKER}'. Expected at {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Could not initialize PiperProvider: {e}")

        if os.environ.get("ELEVENLABS_API_KEY"):
            provider_instances.append(ElevenLabsProvider())
        if os.environ.get("OPENAI_API_KEY"):
            provider_instances.append(OpenAITTSProvider())
            
        return provider_instances

    def speak(self, text: str):
        """Generates and plays audio sentence-by-sentence."""
        if not self.providers:
            print("⚠️ No TTS providers configured. Cannot speak.")
            return

        # Use a thread to prevent the UI from freezing while audio plays
        thread = Thread(target=self._speak_thread, args=(text,))
        thread.start()

    def _speak_thread(self, text: str):
        """The actual speech synthesis and playback, run in a separate thread."""
        # Initialize the sentence segmenter for Hindi
        segmenter = pysbd.Segmenter(language="hi", clean=False)
        sentences = segmenter.segment(text)

        print(f"Segmented into {len(sentences)} sentences for playback.")

        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            audio_content = None
            # Find the first available provider that can synthesize this sentence
            for provider in self.providers:
                provider_name = provider.__class__.__name__
                try:
                    audio_content = provider.synthesize(sentence)
                    print(f"✅ Synthesized sentence with {provider_name}.")
                    break 
                except Exception as e:
                    print(f"⚠️ Provider {provider_name} failed for sentence: {e}")

            # Play the synthesized audio if successful
            if audio_content:
                output_path = "temp_audio.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_content)
                os.system(f"ffplay -autoexit -nodisp -loglevel quiet {output_path}")
            else:
                print(f"❌ All providers failed for sentence: '{sentence}'")

tts_handler = TTSHandler()
