import yt_dlp
import cv2
import os
import glob
from datetime import timedelta
from typing import List, Dict, Tuple

class MediaIngestionTool:
    """
    A specific tool for the Ingestion Agent to handle raw media processing.
    """
    def __init__(self):
        # FIX: Force absolute path to Project Root/output/temp_media
        # This prevents files from getting lost in src/output/temp_media
        # We go up 3 levels from this file: src/ingest/media_processor.py -> src/ingest -> src -> Root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(base_dir, 'output', 'temp_media')
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🔧 Ingestion Tool Initialized. Outputting to: {self.output_dir}")

    def process_video(self, url: str, youtube_cookie: str = None) -> Tuple[str, str, List[Dict]]:
        """
        Main entry point: Downloads video, extracts metadata, and gets keyframes.
        Returns: (video_path, video_title, list_of_frames)
        """
        print(f"🔧 Tool: Processing {url}...")
        
        # 1. Download
        video_path, title = self._download_video(url, youtube_cookie)
        
        # 2. Extract Keyframes (Dynamically get 4 frames regardless of duration)
        frames = self._extract_keyframes(video_path, num_frames=4)
        
        return video_path, title, frames

    def _download_video(self, url: str, youtube_cookie: str = None) -> Tuple[str, str]:
        """
        Downloads video and audio using yt-dlp.
        Returns path to the file.
        """
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': f'{self.output_dir}/%(id)s.%(ext)s',
            'quiet': True,
            'overwrites': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            # Common bypasses for bot detection
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web'],
                    'player_skip': ['webpage', 'configs', 'js']
                }
            }
        }

        cookie_file_path = None
        if youtube_cookie:
            import tempfile
            # Create a temporary file to store the cookie
            fd, cookie_file_path = tempfile.mkstemp(suffix=".txt")
            with os.fdopen(fd, 'w') as f:
                f.write(youtube_cookie)
            ydl_opts['cookiefile'] = cookie_file_path
            print(f"   🍪 Loaded user-provided YouTube cookies")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename, info.get('title', 'Unknown Title')
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            raise
        finally:
            if cookie_file_path and os.path.exists(cookie_file_path):
                try: os.remove(cookie_file_path)
                except: pass

    def _extract_keyframes(self, video_path: str, num_frames: int = 4) -> List[Dict]:
        """
        Extracts a fixed number of keyframes distributed evenly across the video.
        """
        print(f"🖼️ Ingestion Agent: Extracting {num_frames} frames from {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"❌ Error: Could not open video {video_path}")
             return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Safety check for invalid video files
        if total_frames <= 0 or fps <= 0:
            print("❌ Error: Invalid video metadata")
            cap.release()
            return []

        frame_paths = []
        # Calculate step to get exactly num_frames distributed evenly
        step = total_frames // num_frames
        
        for i in range(num_frames):
            frame_id = i * step
            
            # Safety: Don't go past end of video
            if frame_id >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            
            if not ret:
                print(f"   ⚠️ Could not read frame at index {frame_id}")
                continue
            
            # Calculate timestamp
            current_sec = frame_id / fps
            timestamp = str(timedelta(seconds=int(current_sec)))
            
            # Unique filename: frame_0.jpg, frame_1.jpg, etc.
            # Using index ensures we don't overwrite if timestamps are similar
            filename = f"frame_{i}_{int(current_sec)}s.jpg"
            out_path = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(out_path, frame)
            
            frame_paths.append({
                "path": out_path,
                "timestamp": timestamp,
                "mime_type": "image/jpeg"
            })
            
        cap.release()
        print(f"   ✅ Extracted {len(frame_paths)} frames.")
        return frame_paths