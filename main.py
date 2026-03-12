"""
AIVA entry point.

Supports two modes:
  - interactive: live session from webcam + microphone / keyboard
  - single:      process a single video + text input and exit
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.inference.pipeline import AIVAPipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def cli() -> None:
    """AIVA: Emotion-Aware Virtual Companion"""


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to config file.")
@click.option("--video", default=None, help="Path to input video file.")
@click.option("--text", default=None, help="User text input.")
@click.option("--output-dir", default="outputs/", help="Directory for output files.")
def single(config: str, video: str | None, text: str | None, output_dir: str) -> None:
    """Process a single video + text input and generate a response."""
    cfg = load_config(config)
    cfg.system.output_dir = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not text:
        console.print("[red]Error:[/red] --text is required for single mode.")
        sys.exit(1)

    console.print(Panel("[bold cyan]AIVA Single Inference Mode[/bold cyan]", expand=False))

    pipeline = AIVAPipeline(cfg)
    result = pipeline.run(video_path=video, user_text=text)

    console.print(f"\n[bold]Detected emotion:[/bold] {result.sentiment_label}")
    console.print(f"[bold]Valence / Arousal:[/bold] {result.valence:.2f} / {result.arousal:.2f}")
    console.print(f"\n[bold]AIVA response:[/bold]\n{result.llm_response}")

    if result.audio_path:
        console.print(f"\n[bold]Audio saved to:[/bold] {result.audio_path}")


@cli.command()
@click.option("--config", default="configs/default.yaml", help="Path to config file.")
def interactive(config: str) -> None:
    """
    Start an interactive AIVA session.

    In each turn, the user provides text (and optionally a webcam frame is captured).
    AIVA infers emotional context, generates a response, and speaks it.
    """
    cfg = load_config(config)
    pipeline = AIVAPipeline(cfg)

    console.print(Panel("[bold cyan]AIVA Interactive Session[/bold cyan]\nType 'quit' to exit.", expand=False))

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        result = pipeline.run(video_path=None, user_text=user_input)

        console.print(f"\n[dim]Emotion: {result.sentiment_label} | V={result.valence:.2f} A={result.arousal:.2f}[/dim]")
        console.print(f"\n[bold blue]AIVA:[/bold blue] {result.llm_response}")


if __name__ == "__main__":
    cli()
