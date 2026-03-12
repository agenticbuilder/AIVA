"""
TTS Engine Abstraction Layer
=============================
[Proposed Implementation]

The paper describes a Text-to-Speech module for delivering AIVA's responses
as speech. The specific TTS system used is not identified. This module provides
an abstract interface with swappable backends.

Supported backends:
  - coqui:       Coqui XTTS v2 (local, open-source, multilingual)
  - bark:        Bark by Suno AI (local, expressive, open-source)
  - elevenlabs:  ElevenLabs API (cloud, high-quality, requires API key)
  - none:        Returns None; no audio is synthesized
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TTSEngine:
    """
    Abstract TTS engine with pluggable backend support.

    Args:
        backend: One of "coqui", "bark", "elevenlabs", "none".
        config:  Backend-specific configuration dict.
    """

    def __init__(self, backend: str = "none", config: dict | None = None) -> None:
        self.backend = backend.lower()
        self.config = config or {}
        self._engine = None

        if self.backend not in {"coqui", "bark", "elevenlabs", "none"}:
            raise ValueError(f"Unknown TTS backend: '{self.backend}'")

        if self.backend != "none":
            self._load_engine()

    def _load_engine(self) -> None:
        if self.backend == "coqui":
            self._load_coqui()
        elif self.backend == "bark":
            self._load_bark()
        elif self.backend == "elevenlabs":
            self._load_elevenlabs()

    def _load_coqui(self) -> None:
        try:
            from TTS.api import TTS
            model = self.config.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
            logger.info("Loading Coqui TTS model: %s", model)
            self._engine = TTS(model_name=model, progress_bar=False)
            logger.info("Coqui TTS loaded successfully.")
        except ImportError:
            raise ImportError("Coqui TTS is not installed. Run: pip install TTS")

    def _load_bark(self) -> None:
        try:
            from bark import generate_audio, preload_models
            logger.info("Preloading Bark models...")
            preload_models()
            self._engine = generate_audio
            logger.info("Bark loaded successfully.")
        except ImportError:
            raise ImportError("Bark is not installed. Run: pip install bark")

    def _load_elevenlabs(self) -> None:
        try:
            from elevenlabs.client import ElevenLabs
            api_key = self.config.get("api_key") or os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ElevenLabs API key not found. Set ELEVENLABS_API_KEY.")
            self._engine = ElevenLabs(api_key=api_key)
            logger.info("ElevenLabs client initialized.")
        except ImportError:
            raise ImportError("ElevenLabs SDK not installed. Run: pip install elevenlabs")

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        speaker_wav: str | None = None,
        language: str = "en",
    ) -> Path | None:
        """
        Synthesize speech from text and write to a WAV file.

        Args:
            text:         Text to synthesize.
            output_path:  Destination path for the output WAV file.
            speaker_wav:  Reference speaker WAV for voice cloning (Coqui only).
            language:     Language code (Coqui only).

        Returns:
            Path to the output WAV file, or None if backend is "none".
        """
        if self.backend == "none" or self._engine is None:
            logger.debug("TTS backend is 'none'. Skipping synthesis.")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "coqui":
            return self._synthesize_coqui(text, output_path, speaker_wav, language)
        elif self.backend == "bark":
            return self._synthesize_bark(text, output_path)
        elif self.backend == "elevenlabs":
            return self._synthesize_elevenlabs(text, output_path)

        return None

    def _synthesize_coqui(
        self,
        text: str,
        output_path: Path,
        speaker_wav: str | None,
        language: str,
    ) -> Path:
        kwargs: dict = {"text": text, "file_path": str(output_path), "language": language}
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        self._engine.tts_to_file(**kwargs)
        logger.info("Coqui TTS output saved to %s", output_path)
        return output_path

    def _synthesize_bark(self, text: str, output_path: Path) -> Path:
        import numpy as np
        import soundfile as sf

        audio_array = self._engine(text, history_prompt="v2/en_speaker_6")
        sample_rate = 24000
        sf.write(str(output_path), audio_array, sample_rate)
        logger.info("Bark TTS output saved to %s", output_path)
        return output_path

    def _synthesize_elevenlabs(self, text: str, output_path: Path) -> Path:
        voice_id = self.config.get("voice_id", "Rachel")
        audio = self._engine.generate(text=text, voice=voice_id)
        with open(output_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        logger.info("ElevenLabs TTS output saved to %s", output_path)
        return output_path

    @classmethod
    def from_config(cls, cfg) -> "TTSEngine":
        """Instantiate TTSEngine from the tts section of the system config."""
        backend = cfg.get("backend", "none")
        config = {
            "model": cfg.get("model"),
            "language": cfg.get("language", "en"),
            "speaker_wav": cfg.get("speaker_wav"),
            "output_sample_rate": cfg.get("output_sample_rate", 22050),
        }
        return cls(backend=backend, config=config)
