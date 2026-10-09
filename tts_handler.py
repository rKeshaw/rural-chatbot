# tts_handler.py (Improved Configuration)
import os
import io
import httpx
from threading import Thread
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

SPEAKER = "rohan"
QUALITY = "medium"


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
        headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": self.api_key}
        data = {"text": text, "model_id": "eleven_multilingual_v2"}
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


class PiperProvider(BaseTTSProvider):
    def __init__(self, model_path, config_path):
        import numpy as np
        import soundfile as sf
        from piper.voice import PiperVoice
        self.np = np
        self.sf = sf
        self.voice = PiperVoice.load(model_path=model_path, config_path=config_path)
        self.sample_rate = self.voice.config.sample_rate
    def synthesize(self, text: str) -> bytes:
        audio_chunks = self.voice.synthesize(text)
        audio_samples = self.np.concatenate([chunk.audio_float_array for chunk in audio_chunks])
        buffer = io.BytesIO()
        self.sf.write(buffer, audio_samples, self.sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()


class TTSHandler:
    def __init__(self):
        self.providers = self._initialize_providers()
        if not self.providers:
            print("⚠️ No TTS providers could be initialized. Audio playback will be unavailable.")
        else:
            print(f"✅ TTS Handler initialized with {len(self.providers)} providers. Default is '{self.providers[0].__class__.__name__}'.")

    def _initialize_providers(self):
        provider_instances = []
        try:
            MODEL_FOLDER = "local_tts_models"
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
            try:
                provider_instances.append(ElevenLabsProvider())
            except Exception as e:
                print(f"⚠️ Could not initialize ElevenLabsProvider: {e}")
        if os.environ.get("OPENAI_API_KEY"):
            try:
                provider_instances.append(OpenAITTSProvider())
            except Exception as e:
                print(f"⚠️ Could not initialize OpenAITTSProvider: {e}")

        return provider_instances

    def speak(self, text: str):
        """Generates and plays audio sentence-by-sentence."""
        if not self.providers:
            print("⚠️ No TTS providers configured. Cannot speak.")
            return

        thread = Thread(target=self._speak_thread, args=(text,))
        thread.start()

    def _speak_thread(self, text: str):
        import pysbd
        segmenter = pysbd.Segmenter(language="hi", clean=False)
        sentences = segmenter.segment(text)

        print(f"Segmented into {len(sentences)} sentences for playback.")

        for sentence in sentences:
            if not sentence.strip():
                continue

            audio_content = None
            for provider in self.providers:
                provider_name = provider.__class__.__name__
                try:
                    audio_content = provider.synthesize(sentence)
                    print(f"✅ Synthesized sentence with {provider_name}.")
                    break
                except Exception as e:
                    print(f"⚠️ Provider {provider_name} failed for sentence: {e}")

            if audio_content:
                output_path = "temp_audio.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_content)
                os.system(f"ffplay -autoexit -nodisp -loglevel quiet {output_path}")
            else:
                print(f"❌ All providers failed for sentence: '{sentence}'")


tts_handler = TTSHandler()
