

import tempfile
from pathlib import Path

from media_worker.media.audio import extract_audio
from media_worker.media.downloader import download_video


VIDEO_URL = "https://youtu.be/CPk8pffKV64"


def main():
    # Temporary storage is automatically cleaned after the test.
    with tempfile.TemporaryDirectory() as temp_dir:

        video_dir = Path(temp_dir) / "video"
        audio_dir = Path(temp_dir) / "audio"

        print("Downloading video...")

        video_path = download_video(
            VIDEO_URL,
            str(video_dir)
        )

        print("Extracting audio...")

        audio_path = extract_audio(
            video_path,
            str(audio_dir)
        )

        audio_file = Path(audio_path)

        assert audio_file.exists()
        assert audio_file.stat().st_size > 0

        print("Audio extraction successful!")
        print("File:", audio_file.name)
        print(
            "Size:",
            round(audio_file.stat().st_size / (1024 * 1024), 2),
            "MB"
        )


if __name__ == "__main__":
    main()
