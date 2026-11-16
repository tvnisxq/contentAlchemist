"""
Audio extraction module.

Extracts audio from downloaded video files using FFmpeg.
The output is mono, 16 kHz WAV, suitable for speech recognition.
"""

import subprocess
from pathlib import Path


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""
    pass


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract audio from a video and save it as a WAV file.

    Args:
        video_path: Path to the downloaded video.
        output_dir: Directory where extracted audio is saved.

    Returns:
        Absolute path of the extracted WAV file.
    """

    # Convert paths into Path objects for easier filesystem handling.
    source = Path(video_path).resolve()
    destination_dir = Path(output_dir).resolve()

    # Verify that the input video exists.
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    # Create the output directory if necessary.
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Generate a predictable output filename.
    output_file = destination_dir / f"{source.stem}.wav"

    # FFmpeg command:
    # -y          : overwrite an existing output file.
    # -i          : specify the input video.
    # -vn         : ignore the video stream.
    # -ac 1       : convert audio to mono.
    # -ar 16000   : resample audio to 16 kHz.
    # -c:a pcm_s16le : encode as 16-bit PCM WAV.
    command = [
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_file),
    ]

    try:
        # Execute FFmpeg without invoking a shell.
        # This avoids shell injection through filenames.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        # Check whether FFmpeg completed successfully.
        if result.returncode != 0:
            raise AudioExtractionError(
                f"FFmpeg failed: {result.stderr[-2000:]}"
            )

        # Confirm that the output file was actually created.
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise AudioExtractionError(
                "Audio extraction produced an empty or missing file."
            )

        return str(output_file)

    except FileNotFoundError as exc:
        # This also catches a missing FFmpeg executable.
        if exc.filename == "ffmpeg":
            raise AudioExtractionError(
                "FFmpeg is not installed or is not available in PATH."
            ) from exc
        raise
