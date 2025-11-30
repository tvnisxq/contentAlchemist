# Import necessary Flask modules for creating the web server and handling requests
from flask import Flask, request, jsonify, send_from_directory
# Import CORS to allow cross-origin requests from the frontend
from flask_cors import CORS
# Import standard Python libraries for OS interactions and system path handling
import os
import sys
import logging

# --- HELPER FOR CLEAN LOGS ---
def log_checkpoint(msg):
    """
    Only print logs if we are in the Flask Reloader process (WERKZEUG_RUN_MAIN == 'true').
    This prevents double-printing when debug=True.
    """
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print(msg)

# Checkpoint: Confirm that all standard dependencies have been imported successfully
log_checkpoint("Successfully imported dependencies! ✅")

# Suppress Flask/Werkzeug request logging (e.g. "GET / 200") to keep terminal clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# FIX: Calculate the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

# Ensure the project root is in the system path so Python can find 'src' modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Checkpoint: Confirm that the project root path has been configured
log_checkpoint(f"Project root added to path: {project_root} ✅")

# Import the main core logic class from the src.core package
from src.core.alchemist_core import ContentAlchemist

# Checkpoint: Confirm that the core logic module has been imported
log_checkpoint("Successfully imported ContentAlchemist core logic! ✅")

# CONFIGURATION
# Define absolute paths for the UI templates and Output directories relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ContentAlchemist/

# Path to the UI folder containing index.html
TEMPLATE_DIR = os.path.join(BASE_DIR, 'ui')
# Path to the assets folder for AI-generated images
ASSETS_DIR = os.path.join(BASE_DIR, 'output', 'assets')
# Path to the temp_media folder for extracted screenshots
TEMP_MEDIA_DIR = os.path.join(BASE_DIR, 'output', 'temp_media')

# Checkpoint: Confirm that all directory paths are set up
log_checkpoint("Directory paths configured successfully! ✅")

# Initialize the Flask app, specifying the folder for HTML templates
app = Flask(__name__, static_folder=TEMPLATE_DIR)
# Enable CORS for the app to allow browser requests
CORS(app) 

# --- ROUTES ---

# Route: Serve the main UI (index.html) at the root URL
@app.route('/')
def home():
    # Returns the main dashboard HTML file
    return send_from_directory(TEMPLATE_DIR, 'index.html')

# Route: Serve generated assets (images) from the output/assets folder
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

# Route: Serve temporary media (screenshots) from the output/temp_media folder
@app.route('/temp_media/<path:filename>')
def serve_temp_media(filename):
    return send_from_directory(TEMP_MEDIA_DIR, filename)

# Route: API endpoint to analyze a video
@app.route('/analyze', methods=['POST'])
def analyze_video():
    # Parse the JSON data from the request body
    data = request.json
    # Extract the YouTube URL from the data
    video_url = data.get('url')
    # NEW: Extract the Style DNA preference (defaulting to Neutral if not provided)
    style_dna = data.get('style_dna', 'Neutral and professional')
    
    # Validation: Check if a URL was provided
    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    # Checkpoint: Log that a new API request has been received
    print(f"🚀 API Request received for: {video_url} | Style: {style_dna[:30]}...")
    
    try:
        # Initialize the ContentAlchemist orchestrator
        alchemist = ContentAlchemist()
        
        # Checkpoint: Log that the alchemist instance is ready
        print("Alchemist instance initialized. Starting orchestration... ⏳")

        # FIX: RUN SYNCHRONOUSLY - REMOVED asyncio.run()
        # Since alchemist_core.py is now synchronous, we call it directly like a normal function.
        result = alchemist.orchestrate(video_url, style_dna)
        
        # Checkpoint: Log that the analysis process completed successfully
        print("Orchestration completed successfully! ✅")
        
        # Return the result as a JSON response
        return jsonify(result)
    except Exception as e:
        # Log any errors that occurred during processing
        print(f"❌ API Error: {e}")
        # Return the error message with a 500 status code
        return jsonify({"error": str(e)}), 500

# Main entry point for the script
if __name__ == '__main__':
    # We only print the banner if we are in the Reloader process (true server start)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("\n---------------------------------------------------------")
        print("👾  ALCHEMIST SERVER RUNNING")
        print("🔗 OPEN THIS URL IN BRAVE: http://localhost:5000")
        print("---------------------------------------------------------\n")
    
    # Start the Flask development server on port 5000
    app.run(debug=True, port=5000)