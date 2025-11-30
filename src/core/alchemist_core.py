# Import asyncio to handle asynchronous tasks like AI generation
import asyncio
# Import os for file path manipulation and directory creation
import os
# Import time for delays and timestamp generation
import time
# Import json to parse the AI's output
import json
# Import standard utilities for URL handling and image processing
import urllib.request
import cv2 
import re
import warnings
import numpy as np
from typing import List, Dict

# Import Google's Gen AI SDK
import google.generativeai as genai
# Import dotenv to securely load API keys
from dotenv import load_dotenv

# Checkpoint: Confirm standard libraries are loaded
print("Imported Successfully!✅")

# Suppress unnecessary warnings from libraries to keep the console clean
warnings.filterwarnings('ignore')

# Attempt to import the custom MediaIngestionTool, handling potential path issues
try:
    from src.ingest.media_processor import MediaIngestionTool
except ImportError:
    # If direct import fails, look in the directory above to find the module
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.ingest.media_processor import MediaIngestionTool

# Checkpoint: Confirm the custom tool is loaded
print("✓ [Core] MediaIngestionTool imported.")

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate that the API key exists before proceeding
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")
else:
    print("API key configured succesfully!✅")

# Configure the Gemini API client with the loaded key
genai.configure(api_key=API_KEY)


class ContentAlchemist:
    """
    Main Orchestrator class. Manages the pipeline: Download -> Analyze -> Generate -> Save.
    """
    def __init__(self):
        # HARDCODED: Directly use Gemini 2.0 Flash Experimental for speed and simplicity
        self.model_name = 'gemini-2.0-flash'
        # Checkpoint: Confirm model selection
        print(f"🤖 Selected AI Model: {self.model_name}")
        
        # Initialize the Google Gemini client
        self.model = genai.GenerativeModel(self.model_name)
        
        # Initialize the local tool for downloading videos and processing frames
        self.ingest_tool = MediaIngestionTool()
        
        # Placeholder for Imagen (unused in this version as we use screenshots)
        self.imagen_model = None

    def _clean_and_parse_json(self, text):
        """Robustly parses JSON from LLM output, stripping Markdown formatting if present"""
        try:
            # Attempt 1: Direct JSON parsing
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        try:
            # Attempt 2: Strip markdown code blocks (```json ... ```)
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            # Attempt 3: Find the first { and last } to extract JSON substring
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except Exception:
            pass
            
        # Fallback if parsing fails entirely to prevent app crash
        print("⚠️ JSON Parsing failed. Using fallback brief.")
        return {
            "summary": "Summary generation failed. The video content was analyzed but the structure was invalid.",
            "visual_description": "Standard professional video style.",
            "hooks": ["Check this out", "Must watch", "Link in bio"]
        }

    def _upload_to_gemini(self, path: str, mime_type: str):
        """Uploads a local file to Gemini's File API for context understanding"""
        print(f"☁️ Uploading {os.path.basename(path)} to Gemini...")
        file_obj = genai.upload_file(path, mime_type=mime_type)
        
        # Wait for the file to be processed by Google servers
        start_time = time.time()
        while file_obj.state.name == "PROCESSING":
            # Timeout after 60 seconds to prevent infinite loops
            if time.time() - start_time > 60:
                raise TimeoutError("File upload timed out")
            time.sleep(1)
            file_obj = genai.get_file(file_obj.name)
            
        if file_obj.state.name == "FAILED":
            raise ValueError(f"❌ File upload failed")
        
        # Checkpoint: File is ready for analysis
        print(f"✓ [Core] Uploaded: {os.path.basename(path)}")
        return file_obj

    async def _generate_with_retry(self, inputs, config=None, retries=3):
        """Wrapper for API calls that handles Rate Limit (429) errors with backoff"""
        for attempt in range(retries):
            try:
                # Make the API call with optional config
                if config:
                    return await self.model.generate_content_async(inputs, generation_config=config)
                else:
                    return await self.model.generate_content_async(inputs)
            except Exception as e:
                error_str = str(e)
                # If error is quota related, wait and retry
                if "429" in error_str or "Quota exceeded" in error_str:
                    wait_time = (2 ** attempt) * 5 # 5s, 10s, 20s
                    print(f"   ⚠️ Quota exceeded (429). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Crash immediately for non-quota errors
                    raise e
        raise Exception("❌ Max retries exceeded for Gemini API")

    async def ingest_and_analyze(self, video_url: str):
        """Stage 1: Download video, extract keyframes, and generate the Master Brief"""
        # Call the ingestion tool to download and slice the video
        video_path, title, frames = self.ingest_tool.process_video(video_url)
        print("✓ [Core] Local ingestion complete.")

        # Upload the video file to the cloud
        video_file = self._upload_to_gemini(video_path, mime_type="video/mp4")
        
        # Upload the first 3 frames as additional visual context
        uploaded_frames = []
        for frame in frames[:3]: 
            f = self._upload_to_gemini(frame['path'], mime_type=frame['mime_type'])
            uploaded_frames.append(f)

        # Define the prompt for the Master Brief
        prompt = f"""
        Analyze this video titled "{title}".
        Task: Create a JSON 'Master Content Brief'.
        Format:
        {{
            "summary": "...",
            "visual_description": "Describe the visual style/aesthetic of the video in 1 sentence.",
            "hooks": ["hook1", "hook2", "hook3"]
        }}
        Return ONLY JSON. Do not include markdown formatting.
        """
        
        # Generate the brief using the retry wrapper
        response = await self._generate_with_retry(
            [video_file, prompt, *uploaded_frames],
            config={"response_mime_type": "application/json"}
        )
        
        # Parse the result
        brief_data = self._clean_and_parse_json(response.text)
        print("✓ [Core] Master Brief generated.")
        
        return {
            "title": title,
            "brief": brief_data,
            "video_ref": video_file,
            "local_frames": frames
        }

    async def generate_blog_image(self, brief_data, fallback_frames=None):
        """Selects the best available screenshot to use as the blog header"""
        print("🎨 Selecting Visual Asset (Screenshot)...")
        
        # Define the output directory for assets
        base_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/assets'))
        os.makedirs(base_output, exist_ok=True)
        filename = f"generated_header_{int(time.time())}.png"
        filepath = os.path.join(base_output, filename)
        
        # Get event loop for thread execution
        loop = asyncio.get_running_loop()

        # --- SMART SELECTION ---
        # If we have extracted frames, use the middle one
        if fallback_frames and len(fallback_frames) > 0:
            print(f"   👉 Using Extracted Frame as Fallback...")
            try:
                frame_idx = len(fallback_frames) // 2
                best_frame = fallback_frames[frame_idx]
                # Read frame with OpenCV
                img = cv2.imread(best_frame['path'])
                if img is not None:
                    # Save frame to the assets folder
                    cv2.imwrite(filepath, img)
                    print(f"   ✅ Saved Video Screenshot: {filepath}")
                    return {"type": "Image", "path": filepath, "url": f"/assets/{filename}"}
            except Exception as e:
                print(f"   ⚠️ Frame Copy Failed: {e}")

        # --- FINAL FALLBACK ---
        # If no frames exist, create a placeholder image
        print(f"   ❌ Network/Frames failed. Creating synthetic placeholder.")
        img = np.zeros((600, 1200, 3), dtype=np.uint8)
        img[:] = (40, 20, 60) # Dark background
        cv2.putText(img, "AI IMAGE GEN FAILED", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.imwrite(filepath, img)
        return {"type": "Image", "path": filepath, "url": f"/assets/{filename}"}

    async def _run_text_worker(self, agent_name, task, brief_str):
        """Runs a specific text generation agent (e.g., Blog, Social)"""
        print(f"🦾🛠️ Starting {agent_name} Agent...")
        full_prompt = f"Role: {agent_name} Agent.\nContext: {brief_str}\nTask: {task}"
        # Generate text
        response = await self._generate_with_retry(full_prompt)
        print(f"✓ [Core] {agent_name} Agent finished.")
        return {"type": agent_name, "content": response.text}

    def _save_outputs(self, results):
        """Saves generated text content to markdown files on disk"""
        print("💾 Saving outputs to disk...")
        base_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        
        for res in results:
            if res.get('type') == 'Image': continue

            agent_name = res.get('type', 'Unknown')
            content = res.get('content', '')
            
            # Map agent names to folder names
            folder_map = {"Blog": "blogs", "Social": "social", "Newsletter": "newsletter"}
            folder_name = folder_map.get(agent_name, agent_name.lower())
            
            # Create folder if needed
            target_dir = os.path.join(base_output, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Write the markdown file
            filename = f"{agent_name.lower()}_output_{int(time.time())}.md"
            filepath = os.path.join(target_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   -> Saved {agent_name} to: {filepath}")

    async def orchestrate(self, video_url: str):
        """Main Pipeline: Ingest -> Brief -> Sequential Agents -> Save"""
        
        # Step 1: Ingest and create brief
        data = await self.ingest_and_analyze(video_url)
        brief_obj = data['brief']
        brief_str = json.dumps(brief_obj)
        print(f"✅ Brief Ready: {brief_obj['summary'][:50]}...")

        # Prepare frames for the UI to display
        ui_frames = []
        if 'local_frames' in data:
            for f in data['local_frames'][:4]:
                ui_frames.append({
                    "url": f"/temp_media/{os.path.basename(f['path'])}",
                    "timestamp": f['timestamp']
                })

        # --- SEQUENTIAL EXECUTION ---
        # Run agents one by one to avoid rate limits
        blog_result = await self._run_text_worker("Blog", "Write a structured blog post (Markdown).", brief_str)
        social_result = await self._run_text_worker("Social", "Write a LinkedIn post and Twitter thread.", brief_str)
        newsletter_result = await self._run_text_worker("Newsletter", "Write a concise newsletter summary with a CTA.", brief_str)
        image_result = await self.generate_blog_image(brief_obj, fallback_frames=data.get('local_frames'))
        
        # Collect all results
        results = [blog_result, social_result, newsletter_result, image_result]
        
        # Save results to local folders
        self._save_outputs(results)

        # Build final JSON response
        final_output = {
            "meta": {
                "title": data['title'], 
                "brief": brief_obj,
                "frames": ui_frames
            },
            "results": results
        }
        
        print("✨ [Core] All tasks completed successfully!")
        return final_output