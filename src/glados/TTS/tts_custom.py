import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Import your pipeline and model from the Kokoro package.
from kokoro import KPipeline
from kokoro.model import KModel

# Define the sample rate – must match your TTS pipeline settings.
SAMPLE_RATE = 24000

# Global dictionary to cache pipelines by language code.
pipelines = {}

# Optionally, preload a global model if desired.
global_model = None

# Map voice name prefixes to language codes.
LANG_MAP = {
    "af_": "a",
    "am_": "a",
    "bf_": "b",
    "bm_": "b",
    "jf_": "j",
    "jm_": "j",
    "zf_": "z",
    "zm_": "z",
    "ef_": "e",
    "em_": "e",
    "ff_": "f",
    "hf_": "h",
    "hm_": "h",
    "if_": "i",
    "im_": "i",
    "pf_": "p",
    "pm_": "p",
}

def list_available_voices() -> list[str]:
    """
    Returns a list of available voice names.
    """
    return [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "am_adam", 
        "bf_alice", "bf_emma", "bm_daniel", "bm_george",
        "jf_alpha", "jm_kumo", "zf_xiaobei", "zm_yunjian",
        "ef_dora", "em_alex", "ff_siwis", "hf_alpha", "hm_omega",
        "if_sara", "im_nicola", "pf_dora", "pm_santa",
    ]

def get_pipeline_for_voice(voice_name: str) -> KPipeline:
    """
    Determines the language code from the voice name and returns or creates the corresponding pipeline.
    """
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")
    if lang_code not in pipelines:
        if global_model is not None:
            pipelines[lang_code] = KPipeline(lang_code=lang_code, model=global_model)
        else:
            # Let the pipeline load its own model.
            pipelines[lang_code] = KPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]

def generate_speech_audio(voice_name: str, text: str) -> np.ndarray:
    """
    Given a voice name and text, runs the TTS pipeline to produce speech audio as a NumPy array.
    """
    pipeline = get_pipeline_for_voice(voice_name)
    results_gen = pipeline(
        text,
        voice=voice_name,  # e.g., "af_heart"
        speed=1.0,
        split_pattern=r"\n+",
    )
    audio_segments = []
    for result in results_gen:
        audio_tensor = result.audio
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
        audio_tensor = audio_tensor.float().cpu()
        audio_segments.append(audio_tensor)
    if not audio_segments:
        raise RuntimeError("No audio was generated.")
    # Concatenate all segments and return as a NumPy array.
    final_audio = torch.cat(audio_segments, dim=0)
    return final_audio.numpy()

class Synthesizer:
    """
    Custom TTS synthesizer that wraps your multi-lingual pipeline.
    """
    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self, voice: str = "af_heart") -> None:
        self.voice = voice
        self.sample_rate = self.SAMPLE_RATE  # Ensure instance has a sample_rate attribute

    def generate_speech_audio(self, text: str) -> np.ndarray:
        """
        Generate speech audio for the provided text using the custom pipeline.
        """
        return generate_speech_audio(self.voice, text)
