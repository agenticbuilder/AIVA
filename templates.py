"""
Emotion-Aware Prompt Templates
================================
[Proposed Implementation]

The paper describes Emotion-aware Prompt Engineering (EPE) as the mechanism
for injecting sentiment cues into the LLM context. Specific template schemas
are not defined in the paper. The templates below are proposed implementations
designed to be interpretable, consistent, and LLM-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PromptTemplate:
    name: str
    build: Callable[..., str]
    description: str


def _default_template(
    emotion_label: str,
    valence: float,
    arousal: float,
    dominance: float | None = None,
    include_valence: bool = True,
    include_arousal: bool = True,
    include_dominance: bool = False,
    **kwargs,
) -> str:
    """
    Standard EPE prompt prefix.

    Encodes the inferred emotional state and recommended response posture
    as a structured natural language prefix injected before the user message.
    """
    lines = [
        "[EMOTIONAL CONTEXT - INTERNAL USE ONLY]",
        f"Detected user emotion: {emotion_label}",
    ]

    if include_valence:
        valence_desc = _valence_descriptor(valence)
        lines.append(f"Emotional valence: {valence_desc} ({valence:+.2f})")

    if include_arousal:
        arousal_desc = _arousal_descriptor(arousal)
        lines.append(f"Arousal level: {arousal_desc} ({arousal:+.2f})")

    if include_dominance and dominance is not None:
        lines.append(f"Sense of control: {'high' if dominance > 0 else 'low'} ({dominance:+.2f})")

    lines.append("")
    lines.append("Response guidance:")
    lines += _response_guidance(emotion_label, valence, arousal)
    lines.append("[END CONTEXT]")

    return "\n".join(lines)


def _valence_descriptor(v: float) -> str:
    if v > 0.5:
        return "very positive"
    elif v > 0.1:
        return "mildly positive"
    elif v > -0.1:
        return "neutral"
    elif v > -0.5:
        return "mildly negative"
    else:
        return "strongly negative"


def _arousal_descriptor(a: float) -> str:
    if a > 0.6:
        return "highly activated (agitated, excited, or distressed)"
    elif a > 0.2:
        return "moderately activated"
    elif a > -0.2:
        return "calm and steady"
    else:
        return "low energy (withdrawn or fatigued)"


def _response_guidance(emotion: str, valence: float, arousal: float) -> list[str]:
    """
    Generate response style guidelines based on detected emotion.
    [Proposed Implementation]
    """
    guidance_map: dict[str, list[str]] = {
        "sadness": [
            "- Acknowledge the user's feelings directly and without minimizing them.",
            "- Use a warm, gentle, and unhurried tone.",
            "- Avoid unsolicited problem-solving or silver-lining framing.",
            "- Invite the user to share more if they wish, without pressuring them.",
        ],
        "anxiety": [
            "- Validate that the user's concerns are understandable.",
            "- Use a calm, steady tone that models emotional regulation.",
            "- Avoid dismissive phrases like 'don't worry' or 'it'll be fine'.",
            "- Offer grounding or perspective when appropriate, not minimization.",
        ],
        "anger": [
            "- Do not become defensive or escalate.",
            "- Acknowledge the frustration before anything else.",
            "- Keep responses concise; avoid over-explaining.",
            "- Offer to understand more about the source of frustration.",
        ],
        "fear": [
            "- Respond with steadiness and reassurance.",
            "- Validate the fear without amplifying it.",
            "- Avoid catastrophizing language.",
            "- Offer information or perspective that could reduce uncertainty.",
        ],
        "joy": [
            "- Match the user's positive energy authentically.",
            "- Celebrate and affirm their experience.",
            "- Keep the tone light and engaged.",
        ],
        "neutral": [
            "- Respond in a clear, balanced, and conversational tone.",
            "- Neither overly warm nor overly clinical.",
        ],
        "surprise": [
            "- Acknowledge the unexpected nature of the situation.",
            "- Provide grounding information if relevant.",
            "- Match curiosity or concern depending on context.",
        ],
        "disgust": [
            "- Validate the response without amplifying aversion.",
            "- Avoid forcing positivity.",
            "- Acknowledge what the user finds objectionable.",
        ],
    }
    return guidance_map.get(emotion, ["- Respond with empathy and care."])


def _minimal_template(emotion_label: str, valence: float, arousal: float, **kwargs) -> str:
    """Compact EPE prefix for models with tight context windows."""
    return (
        f"[Context: user appears {emotion_label}, valence={valence:+.2f}, "
        f"arousal={arousal:+.2f}. Respond with empathy.]"
    )


TEMPLATES: dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        name="default",
        build=_default_template,
        description="Full EPE prefix with valence/arousal descriptors and response guidance.",
    ),
    "minimal": PromptTemplate(
        name="minimal",
        build=_minimal_template,
        description="Compact single-line prefix for constrained context windows.",
    ),
}


def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown EPE template '{name}'. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]
