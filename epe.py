"""
Emotion-Aware Prompt Engineering (EPE)
========================================
As described in arXiv:2509.03212.

EPE translates the structured sentiment cue produced by MSPN into a natural
language prompt prefix that is injected into the LLM context prior to the
user message. This conditions the LLM's generation toward emotionally
appropriate responses without requiring any model fine-tuning.

[Proposed Implementation]: Template schemas and injection strategy are
proposed implementations. The paper describes EPE as the mechanism for
sentiment-conditioned prompting but does not define specific template formats.
"""

from __future__ import annotations

from src.fusion.mspn import SentimentCue
from src.prompting.templates import get_template
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmotionAwarePromptEngineer:
    """
    Constructs emotion-conditioned LLM prompts from MSPN sentiment cues.

    Args:
        template_name: Name of the prompt template to use (see templates.py).
        include_valence: Whether to include valence score in the prefix.
        include_arousal: Whether to include arousal score in the prefix.
        include_dominance: Whether to include dominance score in the prefix.
        max_prefix_tokens: Approximate token budget for the prefix (soft limit).
    """

    def __init__(
        self,
        template_name: str = "default",
        include_valence: bool = True,
        include_arousal: bool = True,
        include_dominance: bool = False,
        max_prefix_tokens: int = 200,
    ) -> None:
        self.template = get_template(template_name)
        self.include_valence = include_valence
        self.include_arousal = include_arousal
        self.include_dominance = include_dominance
        self.max_prefix_tokens = max_prefix_tokens

    @classmethod
    def from_config(cls, cfg) -> "EmotionAwarePromptEngineer":
        """Instantiate EPE from the epe section of the system config."""
        return cls(
            template_name=cfg.get("template", "default"),
            include_valence=cfg.get("include_valence", True),
            include_arousal=cfg.get("include_arousal", True),
            include_dominance=cfg.get("include_dominance", False),
            max_prefix_tokens=cfg.get("max_prefix_tokens", 200),
        )

    def build_prefix(self, cue: SentimentCue) -> str:
        """
        Build the emotion-aware prompt prefix from a SentimentCue.

        Args:
            cue: Structured sentiment output from MSPN.

        Returns:
            Prompt prefix string to be prepended to the LLM system/user context.
        """
        prefix = self.template.build(
            emotion_label=cue.emotion_label,
            valence=cue.valence,
            arousal=cue.arousal,
            dominance=cue.dominance,
            include_valence=self.include_valence,
            include_arousal=self.include_arousal,
            include_dominance=self.include_dominance,
        )

        logger.debug("EPE prefix built for emotion '%s' (confidence=%.2f)", cue.emotion_label, cue.confidence)
        return prefix

    def build_messages(
        self,
        cue: SentimentCue,
        user_text: str,
        system_prompt: str = "",
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Construct the full messages list for the LLM API call.

        The emotion prefix is injected as part of the system prompt.
        Conversation history (if any) is appended before the current
        user message.

        Args:
            cue: Sentiment cue from MSPN.
            user_text: Raw text input from the user.
            system_prompt: Base system prompt (from configs/system_prompt.txt).
            conversation_history: Previous turns as list of {role, content} dicts.

        Returns:
            List of message dicts for OpenAI / Anthropic API.
        """
        emotion_prefix = self.build_prefix(cue)

        # Combine base system prompt with emotion prefix
        full_system = f"{system_prompt}\n\n{emotion_prefix}".strip()

        messages: list[dict] = [{"role": "system", "content": full_system}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_text})

        return messages
