import os

from flask import Flask, jsonify


# Create an independent Flask application for the media worker.
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """
    Health-check endpoint.

    Render uses this endpoint to verify that the service
    has started and is responding to HTTP requests.
    """
    return jsonify({
        "status": "ok",
        "service": "content-alchemist-media-worker"
    }), 200


@app.route("/", methods=["GET"])
def index():
    """Basic endpoint to confirm that the worker is running."""
    return jsonify({
        "message": "Content Alchemist Media Worker is running"
    }), 200


if __name__ == "__main__":
    # Render provides the PORT environment variable.
    # The fallback allows local development.
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
