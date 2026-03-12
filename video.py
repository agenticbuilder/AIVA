"""Video frame extraction utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extract_frames(
    video_path: str | Path,
    num_frames: int = 8,
    resize: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.
        resize: Target (width, height) for each frame.

    Returns:
        Array of shape (num_frames, H, W, 3) in RGB, dtype uint8.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Could not read frames from {path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Repeat the last valid frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((*resize, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames])
