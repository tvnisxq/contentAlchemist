# Import standard operating system interface for file paths and env vars
import os
# Import time module for sleep delays and timing operations
import time
# Import json for parsing and stringifying data structures
import json
# Import urllib for handling URL requests (used in image downloads if needed)
import urllib.request
# Import OpenCV (cv2) for image manipulation and frame processing
import cv2 
# Import regex module for pattern matching and text cleaning
import re
# Import warnings to control warning message display
import warnings
# Import numpy for numerical operations (used by OpenCV)
import numpy as np
# Import typing for type hinting (List, Dict)
from typing import List, Dict

# Import Google's Generative AI SDK to interact with Gemini models
import google.generativeai as genai
# Import dotenv to load environment variables from .env files
from dotenv import load_dotenv

# Checkpoint: Print message to confirm standard libraries are loaded
print("Imported Successfully!✅")

# Suppress unnecessary warnings to keep the terminal output clean
warnings.filterwarnings('ignore')

# Attempt to import the custom MediaIngestionTool from local source
try:
    # Try importing directly if the script is run from the correct context
    from src.ingest.media_processor import MediaIngestionTool
except ImportError:
    # If import fails, add the project root to system path to find the module
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.ingest.media_processor import MediaIngestionTool

# Checkpoint: Print message to confirm the custom ingestion tool is loaded
print("✓ [Core] MediaIngestionTool imported.")

# Load environment variables from the .env file into the system environment
load_dotenv()
# Retrieve the Google API key from environment variables (Optional fallback)
FALLBACK_API_KEY = os.getenv("GOOGLE_API_KEY")

class ContentAlchemist:
    """
    Main Orchestrator class. 
    RUNNING IN SYNCHRONOUS MODE to ensure compatibility with Flask on Windows.
    This class manages the lifecycle of downloading, analyzing, and generating content.
    """
    def __init__(self, api_key=None):
        # Use provided key or fallback to env variable
        self.api_key = api_key or FALLBACK_API_KEY
        
        # Validate that the API Key exists
        if not self.api_key:
            # Raise an error if the key is missing to stop execution immediately
            raise ValueError("❌ GOOGLE_API_KEY not provided")
        else:
            # Checkpoint: Confirm API key is present
            print("API key configured succesfully!✅")

        # Configure the Gemini SDK with the provided API key
        genai.configure(api_key=self.api_key)
        # HARDCODED: Use Gemini 2.0 Flash as the model
        self.model_name = 'gemini-2.0-flash'
        # Checkpoint: Print which model version is being used
        print(f"🤖 Selected AI Model: {self.model_name}")
        # Initialize the Generative Model instance with the selected model name
        self.model = genai.GenerativeModel(self.model_name)
        # Initialize the ingestion tool instance for handling video files
        self.ingest_tool = MediaIngestionTool()

    def _clean_and_parse_json(self, text):
        """
        Robustly parses JSON from LLM output, cleaning up Markdown formatting.
        This handles cases where the AI wraps JSON in code blocks.
        """
        try:
            # Attempt to parse the text directly as JSON
            return json.loads(text)
        except json.JSONDecodeError: pass # Continue if direct parsing fails
        try:
            # Remove Markdown code block start tag (```json or ```)
            text = re.sub(r'```json\s*', '', text)
            # Remove Markdown code block end tag (```)
            text = re.sub(r'```\s*', '', text)
            # Remove leading/trailing whitespace
            text = text.strip()
            # Attempt to parse the cleaned text
            return json.loads(text)
        except json.JSONDecodeError: pass # Continue if stripped parsing fails
        try:
            # Locate the start of the JSON object
            start = text.find('{')
            # Locate the end of the JSON object
            end = text.rfind('}') + 1
            # If both start and end are found, extract and parse the substring
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except Exception: pass
        
        # Log a warning if all parsing attempts fail
        print("⚠️ JSON Parsing failed. Using fallback brief.")
        # Return a default/fallback object to prevent application crash
        return {
            "summary": "Summary generation failed.",
            "visual_description": "Standard professional video style.",
            "hooks": ["Check this out", "Must watch", "Link in bio"]
        }

    def _clean_text_output(self, text):
        """
        Removes markdown code fences from text output.
        Useful when the AI wraps plain text in ```blocks``` unnecessary.
        """
        # Remove the opening code fence (e.g., ```markdown)
        text = re.sub(r'^```[a-zA-Z]*\n', '', text.strip())
        # Remove the closing code fence
        text = re.sub(r'\n```$', '', text.strip())
        # Return the cleaned, whitespace-stripped text
        return text.strip()

    def _upload_to_gemini(self, path: str, mime_type: str):
        """
        Uploads a local file to Gemini's File API for processing.
        Handles the waiting period while the file is processed remotely.
        """
        # Checkpoint: Print message indicating upload start
        print(f"☁️ Uploading {os.path.basename(path)} to Gemini...")
        # Upload the file using the SDK
        file_obj = genai.upload_file(path, mime_type=mime_type)
        
        # Record start time to track timeout
        start_time = time.time()
        # Loop while the file is still processing on Google's side
        while file_obj.state.name == "PROCESSING":
            # Check if 60 seconds have passed, if so raise timeout
            if time.time() - start_time > 60: raise TimeoutError("File upload timed out")
            # Wait for 1 second before checking status again
            time.sleep(1)
            # Refresh the file object status
            file_obj = genai.get_file(file_obj.name)
            
        # If the file failed to process, raise an error
        if file_obj.state.name == "FAILED": raise ValueError(f"❌ File upload failed")
        
        # Checkpoint: Print message confirming successful upload
        print(f"✓ [Core] Uploaded: {os.path.basename(path)}")
        # Return the file object for use in generation calls
        return file_obj

    def _generate_with_retry(self, inputs, config=None, retries=5):
        """
        Synchronous wrapper for API calls with Exponential Backoff.
        Retries failed requests specifically for Rate Limit (429) errors.
        """
        for attempt in range(retries):
            try:
                # Make the generation call (Synchronously)
                if config:
                    return self.model.generate_content(inputs, generation_config=config)
                else:
                    return self.model.generate_content(inputs)
            except Exception as e:
                # Convert error to string for checking
                error_str = str(e)
                # Check if the error is a Quota Exceeded (429) error
                if "429" in error_str or "Quota exceeded" in error_str:
                    # Calculate wait time: 10s, 20s, 40s...
                    wait_time = (2 ** attempt) * 10 
                    # Checkpoint: Log the retry attempt and wait time
                    print(f"   ⚠️ Quota exceeded (429). Retrying in {wait_time}s...")
                    # Pause execution to let the API quota cool down
                    time.sleep(wait_time) 
                else:
                    # If it's a different error, crash immediately
                    raise e
        # If all retries are exhausted, raise a final exception
        raise Exception("❌ Max retries exceeded for Gemini API")

    def ingest_and_analyze(self, video_url: str, youtube_cookie: str = None):
        """
        Stage 1: Download video, extract keyframes, and generate Master Brief.
        This establishes the context for all downstream tasks.
        """
        # Use the ingestion tool to download video and get 4 frames
        video_path, title, frames = self.ingest_tool.process_video(video_url, youtube_cookie)
        # Checkpoint: Confirm local processing is done
        print("✓ [Core] Local ingestion complete.")

        # Upload the main video file to Gemini
        video_file = self._upload_to_gemini(video_path, mime_type="video/mp4")
        
        # Prepare list for uploaded frame objects
        uploaded_frames = []
        # Loop through the first 3 extracted frames
        for frame in frames[:3]: 
            # Upload each frame to Gemini
            f = self._upload_to_gemini(frame['path'], mime_type=frame['mime_type'])
            uploaded_frames.append(f)

        # Construct the prompt for the initial analysis
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
        
        # Call the API with retry logic to get the brief
        response = self._generate_with_retry(
            [video_file, prompt, *uploaded_frames],
            config={"response_mime_type": "application/json"}
        )
        
        # Parse the JSON response
        brief_data = self._clean_and_parse_json(response.text)
        # Checkpoint: Confirm Brief generation
        print("✓ [Core] Master Brief generated.")
        
        # Return all context data needed for the next steps
        return {
            "title": title,
            "brief": brief_data,
            "video_ref": video_file,
            "local_frames": frames
        }

    def generate_blog_image(self, brief_data, fallback_frames=None):
        """
        Selects the best available screenshot to use as the blog header.
        Uses fallback logic if AI generation logic was present (now just selects frames).
        """
        # Checkpoint: Start visual asset selection
        print("🎨 Selecting Visual Asset (Screenshot)...")
        # Define the absolute path for asset output
        base_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/assets'))
        # Create directory if it doesn't exist
        os.makedirs(base_output, exist_ok=True)
        # Define the filename for the header image
        filename = f"generated_header_{int(time.time())}.png"
        # Define full path
        filepath = os.path.join(base_output, filename)
        
        # Check if we have extracted frames available
        if fallback_frames and len(fallback_frames) > 0:
            print(f"   👉 Using Extracted Frame as Fallback...")
            try:
                # Select the middle frame as it's usually the most relevant
                frame_idx = len(fallback_frames) // 2
                best_frame = fallback_frames[frame_idx]
                # Read the frame from disk using OpenCV
                img = cv2.imread(best_frame['path'])
                # If image read is successful
                if img is not None:
                    # Write the image to the assets folder
                    cv2.imwrite(filepath, img)
                    # Checkpoint: Confirm image save
                    print(f"   ✅ Saved Video Screenshot: {filepath}")
                    return {"type": "Image", "path": filepath, "url": f"/assets/{filename}"}
            except Exception as e:
                print(f"   ⚠️ Frame Copy Failed: {e}")

        # Final Fallback: Create a synthetic image if everything else fails
        print(f"   ❌ Network/Frames failed. Creating synthetic placeholder.")
        # Create a blank dark purple image
        img = np.zeros((600, 1200, 3), dtype=np.uint8)
        img[:] = (40, 20, 60) 
        # Add text to the image
        cv2.putText(img, "AI IMAGE GEN FAILED", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        # Save the placeholder
        cv2.imwrite(filepath, img)
        return {"type": "Image", "path": filepath, "url": f"/assets/{filename}"}

    def _run_text_worker(self, agent_name, task, brief_str, style_dna):
        """
        Runs a specific text generation agent (e.g., Blog, Social) with strict instructions.
        Uses the 'style_dna' to adapt tone and voice.
        """
        # Checkpoint: Log start of specific agent
        print(f"🦾🛠️ Starting {agent_name} Agent with Style: {style_dna[:20]}...")
        
        # Construct the prompt with context, task, style, and negative constraints
        full_prompt = f"""
        Role: {agent_name} Agent.
        
        CONTEXT:
        {brief_str}
        
        TASK:
        {task}
        
        ---
        STYLE & TONE INSTRUCTIONS (STYLE DNA):
        You must adopt the following persona/style guidelines exactly: "{style_dna}"
        
        NEGATIVE CONSTRAINTS (CRITICAL):
        1. Do NOT write meta-commentary like "Here is the blog post" or "The blog would include...".
        2. Do NOT use placeholders. Write the ACTUAL complete content.
        3. Do NOT describe images. Just write the text.
        4. Do NOT say "The blog post would describe...". Just DESCRIBE it.
        5. Output ONLY the final content in Markdown.
        ---
        """
        # Generate text using retry logic
        response = self._generate_with_retry(full_prompt)
        # Clean the output (remove markdown fences)
        cleaned_text = self._clean_text_output(response.text)
        # Checkpoint: Log completion
        print(f"✓ [Core] {agent_name} Agent finished.")
        return {"type": agent_name, "content": cleaned_text}

    def _save_outputs(self, results):
        """
        Iterates through results and saves generated text content to markdown files on disk.
        """
        # Checkpoint: Start saving process
        print("💾 Saving outputs to disk...")
        # Define output base directory
        base_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        
        for res in results:
            # Skip image results as they are saved separately
            if res.get('type') == 'Image': continue
            
            # Get agent name and content
            agent_name = res.get('type', 'Unknown')
            content = res.get('content', '')
            
            # Map agent names to specific folder names
            folder_map = {"Blog": "blogs", "Social": "social", "Newsletter": "newsletter"}
            folder_name = folder_map.get(agent_name, agent_name.lower())
            
            # Construct target directory path
            target_dir = os.path.join(base_output, folder_name)
            # Create directory if needed
            os.makedirs(target_dir, exist_ok=True)
            
            # Create filename with timestamp
            filename = f"{agent_name.lower()}_output_{int(time.time())}.md"
            filepath = os.path.join(target_dir, filename)
            
            # Write content to file with UTF-8 encoding
            with open(filepath, "w", encoding="utf-8") as f: f.write(content)
            print(f"   -> Saved {agent_name} to: {filepath}")

    def orchestrate(self, video_url: str, style_dna: str = "Neutral", youtube_cookie: str = None):
        """
        Main Pipeline function. 
        Orchestrates: Ingest -> Brief -> Sequential Agents -> Save.
        """
        # Step 1: Ingest Video and Generate Master Brief (Synchronous)
        data = self.ingest_and_analyze(video_url, youtube_cookie)
        brief_obj = data['brief']
        # Convert brief object to string for prompts
        brief_str = json.dumps(brief_obj)
        # Checkpoint: Brief is ready
        print(f"✅ Brief Ready: {brief_obj['summary'][:50]}...")

        # Prepare frames data structure for the UI
        ui_frames = []
        if 'local_frames' in data:
            for f in data['local_frames'][:4]:
                ui_frames.append({
                    "url": f"/temp_media/{os.path.basename(f['path'])}",
                    "timestamp": f['timestamp']
                })

        # --- SEQUENTIAL EXECUTION (SYNC) ---
        # Run Blog Agent
        blog_result = self._run_text_worker("Blog", "Write a structured blog post (Markdown).", brief_str, style_dna)
        
        # Run Social Agent with specific prompt for length
        social_prompt = """
        1. Write a DETAILED, LONG-FORM LinkedIn post (300+ words). Use storytelling, short paragraphs, and professional insights. 
        2. Write a DETAILED Twitter Thread (10+ tweets). Break down the core concepts step-by-step.
        Separate the two with a horizontal rule (---).
        """
        social_result = self._run_text_worker("Social", social_prompt, brief_str, style_dna)
        
        # Run Newsletter Agent
        newsletter_result = self._run_text_worker("Newsletter", "Write a concise newsletter summary with a CTA.", brief_str, style_dna)
        
        # Run Image Selection Agent
        image_result = self.generate_blog_image(brief_obj, fallback_frames=data.get('local_frames'))
        
        # Collect all results into a list
        results = [blog_result, social_result, newsletter_result, image_result]
        
        # Save text results to disk
        self._save_outputs(results)

        # Structure final output for the API response
        final_output = {
            "meta": {"title": data['title'], "brief": brief_obj, "frames": ui_frames},
            "results": results
        }
        
        # Checkpoint: Final success message
        print("✨ [Core] All tasks completed successfully!")
        return final_output