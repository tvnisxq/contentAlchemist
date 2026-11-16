import tempfile

from pathlib import Path
from media_worker.media.downloader import download_video


VIDEO_URL = "https://youtu.be/CPk8pffKV64"


def main():
    # Create a temporary directory for the downloaded video.
    with tempfile.TemporaryDirectory() as temp_dir:

        print("Starting YouTube download...")

        # Download the video.
        video_path = download_video(VIDEO_URL, temp_dir)

        # Verify the downloaded file.
        path = Path(video_path)

        assert path.exists(), "Downloaded video does not exist"
        assert path.stat().st_size > 0, "Downloaded video is empty"

        print("Download successful!")
        print("File:", path.name)
        print("Size:", round(path.stat().st_size / (1024 * 1024), 2), "MB")


if __name__ == "__main__":
    main()
