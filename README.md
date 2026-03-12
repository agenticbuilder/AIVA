# AIVA: Emotion-Aware Virtual Companion

<p align="center">
  <img src="AIVA.png" width="300"/>
</p>

---

## Overview

AIVA (AI-based Virtual Companion for Emotion-aware Interaction) is a multimodal AI system designed to perceive, interpret, and respond to human emotional states during natural conversation. This repository provides a research-to-prototype implementation of the framework described in the paper *"AIVA: An AI-based Virtual Companion for Emotion-aware Interaction"* (arXiv:2509.03212).

The system integrates a Multimodal Sentiment Perception Network (MSPN) with a large language model (LLM), an emotion-aware prompt engineering layer, text-to-speech synthesis, and an animated avatar interface. Together, these components form a pipeline capable of generating contextually and emotionally appropriate responses for human-computer interaction.

This repository is a faithful prototype. Where exact implementation details are absent from the paper, engineering decisions are clearly labeled as **[Proposed Implementation]** to distinguish them from paper-described design choices.

---

## Why This Project Exists

Most conversational AI systems treat user input as purely semantic. They process what a person says but largely ignore how they say it. Paralinguistic signals, facial expressions, and sentiment-laden word choices carry critical information about a user's emotional state. Ignoring these signals leads to responses that are factually correct but emotionally tone-deaf.

AIVA addresses this gap by building sentiment perception directly into the interaction pipeline. The system infers emotional context from both visual and textual inputs, injects that context into an LLM at the prompt level, and delivers responses through expressive voice and an animated avatar.

Potential applications include:

- Social companion systems for elderly or isolated individuals
- Mental health support and emotional check-in interfaces
- Human-centered tutoring and educational assistants
- Empathetic customer service and virtual agent systems
- Research testbeds for affective computing and HCI

---

## Problem Statement

Emotion-aware HCI requires solving three problems simultaneously:

1. **Perception**: How do we reliably infer emotional state from multimodal inputs (face, text)?
2. **Injection**: How do we communicate that emotional context to an LLM without disrupting generation quality?
3. **Expression**: How do we deliver a response that feels natural and empathetic?

AIVA proposes an integrated answer. MSPN handles perception. Emotion-aware Prompt Engineering (EPE) handles injection. TTS and avatar synthesis handle expression.

---

## System Architecture

```
Input Layer
    Video Frame (webcam / recorded)        Text Utterance (typed / ASR)
         |                                         |
         v                                         v
  Visual Encoder (ViT)                   Text Encoder (BERT)
         |                                         |
         +--------------> Cross-Attention <--------+
                                |
                   Cross-Modal Fusion Transformer
                                |
                       Sentiment Cue Vector
                                |
                  Emotion-Aware Prompt Engineering
                                |
                        LLM (GPT / LLaMA)
                                |
                       Generated Response Text
                          /             \
                   TTS Engine       Avatar Controller
                   (speech)          (animation)
                          \             /
                       Multimodal Output
```

---

## Component Explanations

### MSPN: Multimodal Sentiment Perception Network

MSPN is the core perceptual module. It processes two input streams:

**Visual Encoder**
Accepts video frames or static images. Extracts spatial features encoding facial expressions, head pose, and appearance-level cues. **[Proposed Implementation]**: A Vision Transformer (ViT-B/16) pretrained on ImageNet, fine-tuned on AffectNet or FER2013, is used as the visual backbone. Features are projected into a 512-dim shared embedding space.

**Text Encoder**
Accepts the user's verbal input as tokenized text. **[Proposed Implementation]**: BERT-base-uncased with a sentiment-aware projection head maps text to the same 512-dim space.

**Cross-Attention Fusion**
A cross-attention layer allows each modality to attend to the other. Visual features attend to text features and vice versa. This enables the model to, for example, weight visual cues more heavily when speech is ambiguous.

**Cross-Modal Fusion Transformer (CMFT)**
A small transformer that takes cross-attended representations and produces a unified sentiment embedding. The CMFT captures interaction effects between modalities that cross-attention alone may miss.

**Sentiment Cue Generation**
The fused embedding is passed through a classification head producing a structured sentiment cue: discrete emotion label + valence/arousal scores.

---

### Emotion-Aware Prompt Engineering (EPE)

EPE translates the structured sentiment cue into a natural language prompt prefix injected at the start of the LLM context. It encodes:

- Inferred emotional state (e.g., "the user appears anxious and uncertain")
- Recommended response style (e.g., "respond with warmth, patience, and reassurance")
- Conversational guardrails (e.g., avoid humor, avoid direct contradiction)

This approach is model-agnostic and requires no LLM fine-tuning.

> **[Proposed Implementation]**: The exact prompt template schema is not specified in the paper. Templates in `src/prompting/epe.py` are proposed based on established practices in empathetic dialogue systems.

---

### TTS Module

Converts LLM text responses to speech. Supports multiple backends:
- **Coqui XTTS**: Local, open-source, emotion-conditioned
- **Bark**: Expressive open-source synthesis
- **ElevenLabs API**: High-quality cloud option

> **[Proposed Implementation]**: The paper's specific TTS backend is not identified. An abstracted interface supporting swappable backends is provided.

---

### Avatar Controller

Renders a virtual character delivering spoken responses with synchronized animation including lip sync, facial expression overlays, and gesture generation.

> **[Proposed Implementation]**: Avatar rendering is not technically detailed in the paper. This module provides an interface compatible with Ready Player Me, VRoid, or a custom 2D renderer.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- ffmpeg (for video preprocessing)

### Setup

```bash
git clone https://github.com/your-org/aiva-companion.git
cd aiva-companion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys (OpenAI, ElevenLabs, etc.)
```

### Optional: GPU acceleration

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Run the demo pipeline

```bash
python scripts/run_demo.py \
  --video data/samples/sample_video.mp4 \
  --text "I've been feeling really overwhelmed lately." \
  --config configs/default.yaml
```

### Run MSPN standalone (emotion inference only)

```bash
python scripts/evaluate_mspn.py \
  --video data/samples/sample_video.mp4 \
  --text "I don't know what to do anymore." \
  --output results/sentiment_output.json
```

### Run the full interactive session

```bash
python src/main.py --mode interactive --config configs/default.yaml
```

---

## Example Usage

```python
from src.inference.pipeline import AIVAPipeline
from src.utils.config import load_config

config = load_config("configs/default.yaml")
pipeline = AIVAPipeline(config)

result = pipeline.run(
    video_path="data/samples/sample_video.mp4",
    user_text="I've been struggling to sleep and I feel really anxious."
)

print(result.sentiment_label)       # "anxiety"
print(result.valence)               # -0.62
print(result.arousal)               # 0.71
print(result.llm_response)          # "I hear you. Feeling anxious and sleep-deprived..."
print(result.audio_path)            # "outputs/response_audio.wav"
```

---

## Sample Input / Output

**Input video**: User with furrowed brow, slightly averted gaze
**Input text**: "I just feel like nothing is going right for me."

**MSPN Output**:
```json
{
  "emotion_label": "sadness",
  "valence": -0.74,
  "arousal": 0.31,
  "dominance": 0.22,
  "confidence": 0.87
}
```

**EPE Prompt Prefix** (injected into LLM context):
```
[SYSTEM CONTEXT]
The user is experiencing sadness with low arousal and low sense of control.
Respond with: warmth, validation, gentle encouragement.
Avoid: minimizing language, unsolicited advice, humor.
Tone: calm, empathetic, unhurried.
[END CONTEXT]
```

**LLM Response**:
```
It sounds like you're carrying a lot right now, and that kind of weight
is exhausting. It makes sense that things feel heavy. You don't have to
have it all figured out at once. I'm here if you want to talk through
what's been happening.
```

---

## Training Approach

### MSPN Training

**[Proposed Implementation]** - Training details are not specified in the paper. The following approach is recommended:

1. Pretrain visual and text encoders independently on standard sentiment/emotion datasets
2. Fine-tune the cross-attention fusion layer on aligned multimodal pairs
3. End-to-end fine-tune the full MSPN on a target dataset

**Recommended Datasets**:
- CMU-MOSI: Multimodal sentiment (video + text + audio)
- CMU-MOSEI: Large-scale multimodal opinion
- AffectNet: Facial expression recognition (visual pretraining)
- SST-2 / SemEval: Text sentiment (text encoder pretraining)
- IEMOCAP: Multimodal emotion in dyadic conversations

### Training Script

```bash
python scripts/train_mspn.py \
  --dataset cmu_mosi \
  --data_dir data/processed/cmu_mosi \
  --config configs/default.yaml \
  --epochs 30 \
  --batch_size 32 \
  --lr 2e-4 \
  --output checkpoints/mspn_v1/
```

---

## Evaluation

**[Proposed Implementation]** - The paper does not report quantitative benchmarks in its current form. Suggested evaluation protocol:

| Metric | Task | Dataset |
|---|---|---|
| Accuracy / F1 | Emotion classification | CMU-MOSI, IEMOCAP |
| MAE | Valence/arousal regression | CMU-MOSEI |
| Weighted F1 | 7-class sentiment | CMU-MOSI |
| Human eval | Response empathy rating | Collected ratings |
| MOS | TTS naturalness | Crowdsourced |

```bash
python scripts/evaluate_mspn.py \
  --split test \
  --dataset cmu_mosi \
  --checkpoint checkpoints/mspn_v1/best.pt
```

---

## Inference Flow

```
1. Frame extraction from video input
2. Visual feature extraction (ViT encoder)
3. Text tokenization and encoding (BERT encoder)
4. Cross-attention fusion (visual attends to text, text attends to visual)
5. CMFT produces unified sentiment embedding
6. Sentiment head produces emotion label + valence/arousal scores
7. EPE constructs prompt prefix from sentiment cue
8. Prefix + user message passed to LLM
9. LLM generates empathetic response text
10. TTS synthesizes speech from response text
11. Avatar controller renders animated delivery
12. Audio + animation delivered to user
```

---

## Limitations

- MSPN performance degrades under poor lighting, occlusion, or low image resolution
- EPE relies on the LLM's instruction-following quality; weaker models may ignore emotional guidance
- The system does not currently process audio prosody as a third modality (paper scope)
- Avatar rendering quality depends heavily on the chosen backend
- No real-time streaming inference is implemented in this prototype
- Cross-cultural emotion interpretation is not addressed

---

## Ethical Considerations

Emotion-aware AI systems carry significant ethical responsibility:

- **Consent and transparency**: Users must be informed that their facial expressions and speech are being analyzed for emotional content
- **Accuracy limitations**: Emotion recognition systems can misclassify, especially across cultural backgrounds, ages, and neurological profiles
- **Sensitive use cases**: Deployment in mental health contexts requires clinical oversight; this system is not a therapeutic tool
- **Data privacy**: Video and audio inputs are sensitive biometric data. This system should not store or transmit such data without explicit user consent and appropriate legal compliance
- **Bias**: Pretrained models may encode demographic biases present in their training data. Evaluation across diverse populations is essential before deployment

---

## Research-to-Engineering Notes

The AIVA paper is a system-level proposal describing component interactions at an architectural level. It does not provide training code, dataset splits, hyperparameters, or quantitative benchmarks. As a result:

- All module implementations in this repository are **proposed implementations** consistent with the paper's architectural description
- The CMFT is implemented as a standard 4-layer transformer with cross-modal token mixing, following common practice in multimodal fusion literature
- EPE prompt templates are designed based on established empathetic dialogue research
- TTS and avatar backends are abstracted to allow substitution

Contributors seeking closer paper fidelity are encouraged to contact the original authors for implementation details.

---

## From Paper to Prototype

Translating a research paper into a working MVP requires filling in the gap between architectural description and executable code. For AIVA, that process involved the following steps:

**Step 1: Identify specified components**
The paper clearly specifies: a visual encoder, a text encoder, cross-attention fusion, a cross-modal fusion transformer, sentiment cue output, prompt injection into an LLM, TTS output, and an avatar module. These form the non-negotiable scaffold.

**Step 2: Select concrete implementations for each**
Where the paper names a component without specifying it (e.g., "visual encoder"), we select a well-established architecture consistent with the described function (ViT-B/16 for visual encoding, BERT-base for text encoding).

**Step 3: Define the data contract between components**
Each module's input and output tensor shapes are defined before implementation begins. This allows modules to be developed and tested independently.

**Step 4: Build a minimal end-to-end path first**
Before optimizing any individual module, establish a working pipeline from raw input to text output. This catches integration issues early.

**Step 5: Replace stubs with real implementations incrementally**
Start with placeholder modules that return dummy outputs of the correct shape. Replace each stub with a real implementation one at a time.

**Step 6: Evaluate against established benchmarks**
Even without paper-reported metrics, MSPN performance can be evaluated on CMU-MOSI and IEMOCAP to establish a baseline.

---

## Assumed Engineering Decisions

### From the paper directly
- Visual encoder and text encoder as separate streams feeding a fusion module
- Cross-attention as the fusion mechanism between modalities
- A cross-modal fusion transformer downstream of cross-attention
- Sentiment cue output used to condition LLM prompts (Emotion-aware Prompt Engineering)
- TTS and animated avatar as final output modalities
- Target applications: empathetic HCI, virtual companions, social care

### Proposed for this implementation
- ViT-B/16 as the visual backbone (pretrained on ImageNet, fine-tuned on AffectNet)
- BERT-base-uncased as the text backbone
- 512-dim shared embedding space for both modalities
- 4-layer transformer for CMFT with 8 attention heads
- Valence-Arousal-Dominance regression head alongside emotion classification
- EPE prompt template schema (emotion label, VAD scores, tone instructions)
- Coqui XTTS as the default TTS backend
- Avatar interface abstraction supporting Ready Player Me and VRoid
- CMU-MOSI and CMU-MOSEI as primary training and evaluation datasets

### Left as future work
- Audio/prosody as a third input modality
- Real-time streaming inference
- Fine-tuning the LLM on affective dialogue data
- Cross-cultural emotion calibration
- Clinical validation for mental health deployment contexts

---

## Roadmap

- [ ] MSPN training pipeline with CMU-MOSI
- [ ] Evaluation suite with standard sentiment benchmarks
- [ ] Real-time webcam inference mode
- [ ] Gradio web demo
- [ ] Audio prosody as third input modality
- [ ] Streaming LLM inference
- [ ] Docker container for reproducible deployment
- [ ] HuggingFace model card and weights upload
- [ ] Dataset preprocessing scripts for IEMOCAP and CMU-MOSEI

---

## Citation

If you use this repository in your research, please cite the original paper:

```bibtex
@article{aiva2025,
  title     = {AIVA: An AI-based Virtual Companion for Emotion-aware Interaction},
  author    = {[Authors as listed on arXiv]},
  journal   = {arXiv preprint arXiv:2509.03212},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.03212}
}
```

---

## License

This repository is released under the MIT License. See `LICENSE` for details.

The paper this work is based on is the intellectual property of its original authors. This repository is an independent prototype implementation and is not affiliated with or endorsed by the original authors.

---

## Acknowledgments

This prototype implementation draws on the architectural description in arXiv:2509.03212. The implementation choices for individual modules are informed by the broader multimodal sentiment analysis literature, including work on CMU-MOSI, IEMOCAP, and cross-modal transformer fusion.

Dependencies include PyTorch, HuggingFace Transformers, timm, Coqui TTS, and OpenAI Python SDK. See `requirements.txt` for the full list.
