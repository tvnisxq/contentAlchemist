"""
YouTube media downloader.

Responsibilities:
1. Validate the supplied YouTube URL.
2. Download a video with a maximum preferred resolution of 480p.
3. Save the downloaded file in a caller-provided directory.
4. Return the downloaded file's path.
"""

from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


class MediaDownloadError(Exception):
    """Raised when a video cannot be downloaded."""
    pass


def validate_youtube_url(url: str) -> bool:
    """
    Check whether the supplied URL belongs to YouTube.

    We explicitly allow known YouTube domains instead of accepting
    arbitrary URLs. This prevents our worker from becoming a
    general-purpose URL downloader.
    """

    try:
        parsed_url = urlparse(url)

        allowed_domains = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
        }

        return (
            parsed_url.scheme == "https"
            and parsed_url.hostname in allowed_domains
        )

    except (ValueError, AttributeError):
        return False


def download_video(url: str, output_dir: str) -> str:
    """
    Download a YouTube video and return its local file path.

    Args:
        url: YouTube video URL.
        output_dir: Directory where the downloaded file is saved.

    Returns:
        Absolute path of the downloaded video.

    Raises:
        ValueError: If the URL is invalid.
        MediaDownloadError: If downloading fails.
    """

    # Step 1: Validate the URL before contacting YouTube.
    if not validate_youtube_url(url):
        raise ValueError("Please provide a valid HTTPS YouTube URL.")

    # Step 2: Create the output directory if it doesn't exist.
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 3: Configure yt-dlp.
    options = {
        # Download the best video stream up to 480p and the best audio stream.
        # If separate streams aren't available, fall back to a combined format.
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]",

        # Use the YouTube video ID as the filename.
        "outtmpl": str(output_path / "%(id)s.%(ext)s"),

        # Download only one video, not an entire playlist.
        "noplaylist": True,

        # Avoid unnecessary terminal output.
        "quiet": True,
        "no_warnings": True,

        # Network reliability settings.
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,

        # Restrict filenames to portable characters.
        "restrictfilenames": True,
    }

    try:
        # Step 4: Initialize yt-dlp and download the video.
        with yt_dlp.YoutubeDL(options) as downloader:

            info = downloader.extract_info(
                url,
                download=True
            )

            # Step 5: Determine the actual downloaded file path.
            downloaded_file = Path(
                info.get("filepath") or downloader.prepare_filename(info)
            ).resolve()

        # Step 6: Verify that the file actually exists.
        if not downloaded_file.is_file():
            raise MediaDownloadError(
                "Download completed, but the video file was not found."
            )

        return str(downloaded_file)

    except yt_dlp.utils.DownloadError as exc:
        raise MediaDownloadError(
            f"YouTube download failed: {exc}"
        ) from exc

    except MediaDownloadError:
        raise

    except Exception as exc:
        raise MediaDownloadError(
            f"Unexpected downloader error: {exc}"
        ) from exc
