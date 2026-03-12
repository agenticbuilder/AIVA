"""
AIVA Inference Pipeline
========================
Full end-to-end inference pipeline as described in arXiv:2509.03212.

Pipeline stages:
  1. Frame extraction from video input
  2. MSPN: visual + text encoding, cross-attention, CMFT, sentiment head
  3. EPE: construct emotion-conditioned prompt prefix
  4. LLM: generate empathetic response
  5. TTS: synthesize speech
  6. Avatar: trigger animated delivery

Each stage is independently configurable and can be disabled for ablation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.avatar.avatar_controller import AvatarController
from src.fusion.mspn import MSPN, SentimentCue
from src.prompting.epe import EmotionAwarePromptEngineer
from src.tts.tts_engine import TTSEngine
from src.utils.logging import get_logger
from src.utils.video import extract_frames

logger = get_logger(__name__)


@dataclass
class AIVAResult:
    """Full output from a single AIVA inference pass."""
    user_text: str
    sentiment_label: str
    valence: float
    arousal: float
    dominance: float
    sentiment_confidence: float
    emotion_prefix: str
    llm_response: str
    audio_path: Path | None = None
    avatar_command: dict | None = None
    latency_ms: dict = field(default_factory=dict)


class AIVAPipeline:
    """
    Full AIVA inference pipeline.

    Composes all system modules in sequence:
      MSPN -> EPE -> LLM -> TTS -> Avatar

    Args:
        cfg: Full system DictConfig (loaded from configs/default.yaml).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.system.get("device", "cpu"))
        logger.info("Initializing AIVA pipeline on device: %s", self.device)

        # Load MSPN
        logger.info("Loading MSPN...")
        self.mspn = MSPN(cfg)
        self.mspn.to(self.device)
        self.mspn.eval()

        if cfg.get("mspn_checkpoint"):
            self._load_mspn_checkpoint(cfg.mspn_checkpoint)

        # Emotion-Aware Prompt Engineering
        self.epe = EmotionAwarePromptEngineer.from_config(cfg.epe)

        # LLM client
        self.llm_provider = cfg.llm.provider
        self.llm_model = cfg.llm.model
        self.llm_temperature = cfg.llm.get("temperature", 0.7)
        self.llm_max_tokens = cfg.llm.get("max_tokens", 512)
        self._system_prompt = self._load_system_prompt(cfg.llm.get("system_prompt_path"))
        self._llm_client = self._init_llm_client()

        # TTS
        logger.info("Initializing TTS engine (backend=%s)...", cfg.tts.backend)
        self.tts = TTSEngine.from_config(cfg.tts)

        # Avatar
        logger.info("Initializing avatar controller (backend=%s)...", cfg.avatar.backend)
        self.avatar = AvatarController.from_config(cfg.avatar)

        # Output directory
        self.output_dir = Path(cfg.system.get("output_dir", "outputs/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AIVA pipeline ready.")

    def _load_system_prompt(self, path: str | None) -> str:
        if path and Path(path).exists():
            return Path(path).read_text().strip()
        return "You are AIVA, an empathetic AI companion."

    def _init_llm_client(self):
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        else:
            logger.warning("LLM provider '%s' not recognized. LLM calls will return stubs.", self.llm_provider)
            return None

    def _load_mspn_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("MSPN checkpoint not found at %s. Using random weights.", path)
            return
        state = torch.load(path, map_location=self.device)
        self.mspn.load_state_dict(state["model_state_dict"])
        logger.info("Loaded MSPN checkpoint from %s", path)

    def _run_mspn(self, video_path: str | None, user_text: str) -> SentimentCue:
        """Stage 1-2: Extract features and run MSPN."""
        frames = None
        if video_path:
            t0 = time.perf_counter()
            raw_frames = extract_frames(
                video_path,
                num_frames=self.cfg.visual_encoder.get("frames_per_sample", 8),
                resize=(self.cfg.visual_encoder.input_size, self.cfg.visual_encoder.input_size),
            )
            # Convert to float tensor (B=1, T, C, H, W) in [0, 1]
            import numpy as np
            frames_np = raw_frames.astype("float32") / 255.0
            frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2)   # (T, C, H, W)
            frames = frames.unsqueeze(0).to(self.device)               # (1, T, C, H, W)
            logger.debug("Frame extraction: %.1f ms", (time.perf_counter() - t0) * 1000)

        with torch.no_grad():
            cues = self.mspn.predict(frames=frames, texts=[user_text])

        return cues[0]

    def _run_llm(self, messages: list[dict]) -> str:
        """Stage 4: Call LLM and return response text."""
        if self._llm_client is None:
            return "[LLM not configured. Set provider and API key in config.]"

        if self.llm_provider == "openai":
            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.llm_provider == "anthropic":
            # Anthropic separates system from user messages
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            response = self._llm_client.messages.create(
                model=self.llm_model,
                system=system_msg,
                messages=user_msgs,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
            return response.content[0].text.strip()

        return "[LLM provider not implemented.]"

    def run(
        self,
        video_path: str | None,
        user_text: str,
        conversation_history: list[dict] | None = None,
        output_audio_name: str = "response.wav",
    ) -> AIVAResult:
        """
        Run the full AIVA pipeline for a single turn.

        Args:
            video_path:           Path to user video input, or None for text-only.
            user_text:            User's text utterance.
            conversation_history: Prior turns for multi-turn context.
            output_audio_name:    Filename for the output audio WAV.

        Returns:
            AIVAResult with all intermediate and final outputs.
        """
        latency: dict[str, float] = {}

        # Stage 1-2: MSPN sentiment inference
        t0 = time.perf_counter()
        cue = self._run_mspn(video_path, user_text)
        latency["mspn_ms"] = (time.perf_counter() - t0) * 1000
        logger.info("MSPN: %s (V=%.2f, A=%.2f, conf=%.2f)", cue.emotion_label, cue.valence, cue.arousal, cue.confidence)

        # Stage 3: EPE - build emotion-conditioned messages
        t0 = time.perf_counter()
        messages = self.epe.build_messages(
            cue=cue,
            user_text=user_text,
            system_prompt=self._system_prompt,
            conversation_history=conversation_history,
        )
        emotion_prefix = self.epe.build_prefix(cue)
        latency["epe_ms"] = (time.perf_counter() - t0) * 1000

        # Stage 4: LLM generation
        t0 = time.perf_counter()
        llm_response = self._run_llm(messages)
        latency["llm_ms"] = (time.perf_counter() - t0) * 1000
        logger.info("LLM response generated (%d chars, %.0f ms)", len(llm_response), latency["llm_ms"])

        # Stage 5: TTS synthesis
        t0 = time.perf_counter()
        audio_path = None
        if self.tts.backend != "none":
            audio_out = self.output_dir / output_audio_name
            audio_path = self.tts.synthesize(
                text=llm_response,
                output_path=audio_out,
                speaker_wav=self.cfg.tts.get("speaker_wav"),
                language=self.cfg.tts.get("language", "en"),
            )
        latency["tts_ms"] = (time.perf_counter() - t0) * 1000

        # Stage 6: Avatar animation
        t0 = time.perf_counter()
        avatar_command = None
        if self.avatar.backend != "none":
            avatar_command = self.avatar.animate(
                response_text=llm_response,
                audio_path=audio_path,
                emotion_label=cue.emotion_label,
            )
        latency["avatar_ms"] = (time.perf_counter() - t0) * 1000

        total_ms = sum(latency.values())
        logger.info("Pipeline complete. Total latency: %.0f ms", total_ms)

        return AIVAResult(
            user_text=user_text,
            sentiment_label=cue.emotion_label,
            valence=cue.valence,
            arousal=cue.arousal,
            dominance=cue.dominance,
            sentiment_confidence=cue.confidence,
            emotion_prefix=emotion_prefix,
            llm_response=llm_response,
            audio_path=audio_path,
            avatar_command=avatar_command,
            latency_ms=latency,
        )
