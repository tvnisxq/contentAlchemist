import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest.media_processor import MediaIngestionTool

def test():
    tool = MediaIngestionTool()
    url = "https://youtu.be/xa17zHJhNhA"
    print(f"Testing download for: {url}")
    try:
        path, title, frames = tool.process_video(url)
        print(f"Success! Downloaded: {title} to {path}")
        print(f"Extracted {len(frames)} frames.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test()
