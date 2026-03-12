"""
Avatar Controller
==================
[Proposed Implementation]

The paper describes an animated avatar module that delivers AIVA's responses
with synchronized lip motion and expressive gesture. Technical implementation
details for the avatar system are not specified in the paper.

This module provides an abstract interface that can be connected to:
  - A local 2D/3D renderer
  - Ready Player Me (readyplayer.me)
  - VRoid Hub character
  - A custom WebSocket-based avatar service

The controller's role is to:
  1. Accept response text and associated audio
  2. Generate lip sync data from the audio waveform
  3. Map the emotional tone to a facial expression preset
  4. Send animation commands to the rendering backend
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPRESSION_PRESETS = {
    "neutral":  {"blend_shape": "neutral",  "intensity": 1.0},
    "joy":      {"blend_shape": "happy",    "intensity": 0.85},
    "sadness":  {"blend_shape": "sad",      "intensity": 0.75},
    "anger":    {"blend_shape": "angry",    "intensity": 0.70},
    "fear":     {"blend_shape": "fearful",  "intensity": 0.80},
    "disgust":  {"blend_shape": "disgusted","intensity": 0.65},
    "surprise": {"blend_shape": "surprised","intensity": 0.90},
    "anxiety":  {"blend_shape": "worried",  "intensity": 0.75},
}


class AvatarController:
    """
    Abstract avatar animation controller.

    In stub mode (backend='none'), logs what would be sent to a renderer.
    With a real backend, sends animation commands over HTTP/WebSocket.

    Args:
        backend: One of "none", "websocket", "ready_player_me", "custom".
        service_url: URL of the avatar rendering service (if applicable).
        avatar_id: Identifier for the specific avatar character.
        enable_lip_sync: Whether to compute and send lip sync data.
        enable_gestures: Whether to trigger idle gesture animations.
    """

    def __init__(
        self,
        backend: str = "none",
        service_url: str | None = None,
        avatar_id: str | None = None,
        enable_lip_sync: bool = True,
        enable_gestures: bool = False,
    ) -> None:
        self.backend = backend.lower()
        self.service_url = service_url
        self.avatar_id = avatar_id
        self.enable_lip_sync = enable_lip_sync
        self.enable_gestures = enable_gestures

        if self.backend not in {"none", "websocket", "ready_player_me", "custom"}:
            raise ValueError(f"Unknown avatar backend: '{self.backend}'")

        logger.info("AvatarController initialized with backend='%s'", self.backend)

    @classmethod
    def from_config(cls, cfg) -> "AvatarController":
        return cls(
            backend=cfg.get("backend", "none"),
            service_url=cfg.get("service_url"),
            avatar_id=cfg.get("avatar_id"),
            enable_lip_sync=cfg.get("enable_lip_sync", True),
            enable_gestures=cfg.get("enable_gestures", False),
        )

    def animate(
        self,
        response_text: str,
        audio_path: Path | None,
        emotion_label: str = "neutral",
    ) -> dict | None:
        """
        Trigger avatar animation for a given response.

        Args:
            response_text: The text of AIVA's response.
            audio_path:    Path to the synthesized audio WAV file.
            emotion_label: Detected user emotion, used to select expression preset.

        Returns:
            Animation command dict sent to the backend, or None if backend is 'none'.
        """
        expression = EXPRESSION_PRESETS.get(emotion_label, EXPRESSION_PRESETS["neutral"])

        command = {
            "type": "animate_response",
            "avatar_id": self.avatar_id,
            "expression": expression,
            "lip_sync": {
                "enabled": self.enable_lip_sync,
                "audio_path": str(audio_path) if audio_path else None,
            },
            "gestures": {
                "enabled": self.enable_gestures,
                "type": "idle_nod" if emotion_label in {"sadness", "neutral"} else "open",
            },
            "text": response_text,
        }

        if self.backend == "none":
            logger.debug("Avatar stub: %s", json.dumps(command, indent=2))
            return command

        elif self.backend == "websocket":
            return self._send_websocket(command)

        elif self.backend in {"ready_player_me", "custom"}:
            return self._send_http(command)

        return None

    def _send_websocket(self, command: dict) -> dict | None:
        """
        Send animation command over WebSocket.
        [Proposed Implementation] - requires a running avatar service.
        """
        try:
            import asyncio
            import websockets

            async def _send():
                async with websockets.connect(self.service_url) as ws:
                    await ws.send(json.dumps(command))
                    response = await ws.recv()
                    return json.loads(response)

            return asyncio.run(_send())
        except Exception as exc:
            logger.warning("Avatar WebSocket send failed: %s", exc)
            return None

    def _send_http(self, command: dict) -> dict | None:
        """
        Send animation command via HTTP POST.
        [Proposed Implementation] - requires a running avatar service.
        """
        try:
            import requests
            resp = requests.post(
                f"{self.service_url}/animate",
                json=command,
                timeout=5.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Avatar HTTP send failed: %s", exc)
            return None
