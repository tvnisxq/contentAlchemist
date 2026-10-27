"""
Content Alchemist - Vercel-ready backend.

What changed vs. the old version:
  * No yt-dlp / FFmpeg / OpenCV: Gemini reads the public YouTube URL directly
    (audio + visuals), so nothing is downloaded and YouTube's cloud-IP blocking
    no longer matters.
  * Stateless: nothing is written to disk (Vercel's filesystem is read-only
    apart from /tmp).
  * The three writer agents run in parallel, so the whole request finishes
    well inside Vercel's function time limit.
  * Uses the current `google-genai` SDK. The model is configurable through the
    GEMINI_MODEL environment variable, so a model retirement is a settings
    change, not a code change.
  * The /analyze request and response shape is unchanged, so the existing UI
    keeps working.
  * Removed the /debug and /debug_yt routes and the unrelated PDF/RAG demo
    code that used to run at import time.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

try:  # local development convenience only
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

# Check Google's current model list and change this in the Vercel dashboard
# (Settings -> Environment Variables) if a model is retired.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK_API_KEY = os.environ.get("GOOGLE_API_KEY")

app = Flask(__name__)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
YT_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#]*&)?v=|embed/|shorts/|live/))"
    r"([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str):
    match = YT_ID_RE.search(url or "")
    return match.group(1) if match else None


def fetch_title(watch_url: str):
    """Video title via YouTube's public oEmbed endpoint (no API key needed)."""
    try:
        query = urllib.parse.urlencode({"url": watch_url, "format": "json"})
        with urllib.request.urlopen(
            f"https://www.youtube.com/oembed?{query}", timeout=8
        ) as resp:
            return json.loads(resp.read().decode("utf-8")).get("title")
    except Exception:
        return None


def best_thumbnail(video_id: str) -> str:
    """Highest-resolution thumbnail that actually exists for this video."""
    for name in ("maxresdefault", "hqdefault"):
        url = f"https://img.youtube.com/vi/{video_id}/{name}.jpg"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return url
        except Exception:
            continue
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def clean_and_parse_json(text: str):
    """Parse JSON from model output, tolerating markdown fences."""
    candidates = [text]
    stripped = re.sub(r"```(?:json)?\s*", "", text or "").strip()
    candidates.append(stripped)
    start, end = stripped.find("{"), stripped.rfind("}") + 1
    if start != -1 and end > start:
        candidates.append(stripped[start:end])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return {
        "title": None,
        "summary": "Summary generation failed.",
        "visual_description": "Standard professional video style.",
        "hooks": ["Check this out", "Must watch", "Link in bio"],
    }


def clean_text_output(text: str) -> str:
    """Strip a wrapping ``` fence if the model added one."""
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def generate(client, contents, json_mode=False, retries=4):
    """generate_content with short backoff on rate-limit / overload errors."""
    config = (
        types.GenerateContentConfig(response_mime_type="application/json")
        if json_mode
        else None
    )
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
            return response.text or ""
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in (429, 503) and attempt < retries - 1:
                time.sleep(min(5 * (2 ** attempt), 20))  # 5s, 10s, 20s
                continue
            raise


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def build_brief(client, watch_url: str):
    prompt = """
    Analyze this video.
    Task: Create a JSON 'Master Content Brief'.
    Format:
    {
        "title": "the video's title",
        "summary": "...",
        "visual_description": "Describe the visual style/aesthetic of the video in 1 sentence.",
        "hooks": ["hook1", "hook2", "hook3"]
    }
    Return ONLY JSON. Do not include markdown formatting.
    """
    contents = types.Content(
        parts=[
            types.Part(file_data=types.FileData(file_uri=watch_url)),
            types.Part(text=prompt),
        ]
    )
    return clean_and_parse_json(generate(client, contents, json_mode=True))


def run_text_worker(client, agent_name, task, brief_str, style_dna):
    prompt = f"""
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
    try:
        return {
            "type": agent_name,
            "content": clean_text_output(generate(client, prompt)),
        }
    except Exception as exc:  # one failed agent shouldn't discard the others
        return {
            "type": agent_name,
            "content": f"⚠️ {agent_name} generation failed: {exc}",
        }


SOCIAL_TASK = """
1. Write a DETAILED, LONG-FORM LinkedIn post (300+ words). Use storytelling, short paragraphs, and professional insights.
2. Write a DETAILED Twitter Thread (10+ tweets). Break down the core concepts step-by-step.
Separate the two with a horizontal rule (---).
"""

TEXT_TASKS = [
    ("Blog", "Write a structured blog post (Markdown)."),
    ("Social", SOCIAL_TASK),
    ("Newsletter", "Write a concise newsletter summary with a CTA."),
]


def orchestrate(api_key: str, video_url: str, style_dna: str):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("That doesn't look like a valid YouTube URL.")
    watch_url = f"https://www.youtube.com/watch?v={video_id}"

    client = genai.Client(api_key=api_key)

    brief = build_brief(client, watch_url)
    title = fetch_title(watch_url) or brief.get("title") or "Untitled video"
    brief_str = json.dumps(brief)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(run_text_worker, client, name, task, brief_str, style_dna)
            for name, task in TEXT_TASKS
        ]
        text_results = [f.result() for f in futures]

    thumbnail = best_thumbnail(video_id)
    image_result = {"type": "Image", "path": thumbnail, "url": thumbnail}

    frames = [
        {
            "url": f"https://img.youtube.com/vi/{video_id}/hq{i}.jpg",
            "timestamp": f"Preview {i}",
        }
        for i in (1, 2, 3)
    ]

    return {
        "meta": {"title": title, "brief": brief, "frames": frames},
        "results": text_results + [image_result],
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    # On Vercel, public/index.html is served by the CDN before Flask is hit;
    # this route is for running locally with `python app.py`.
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL})


@app.route("/analyze", methods=["POST"])
def analyze_video():
    data = request.get_json(silent=True) or {}
    video_url = (data.get("url") or "").strip()
    style_dna = data.get("style_dna") or "Neutral and professional"
    api_key = data.get("api_key") or FALLBACK_API_KEY
    # `cookie` is accepted for backwards compatibility with the old UI but is
    # no longer needed or used.

    if not video_url:
        return jsonify({"error": "No URL provided"}), 400
    if not api_key:
        return jsonify({"error": "No API Key provided"}), 400

    try:
        return jsonify(orchestrate(api_key, video_url, style_dna))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        code = getattr(exc, "code", None)
        message = str(exc)
        if code in (400, 401, 403) and "API key" in message:
            return jsonify({"error": "Gemini rejected the API key. Check it and try again."}), 401
        if code == 429:
            return jsonify({"error": "Gemini rate limit or quota reached. Wait a minute and retry."}), 429
        print(f"API error: {message}")
        return jsonify({"error": message}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
