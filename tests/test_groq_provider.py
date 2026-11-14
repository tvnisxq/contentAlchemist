# os provides access to environment variables.
import os

# Load environment variables from a local file.
from dotenv import load_dotenv

# Import our concrete Groq provider.
from providers.groq_provider import GroqProvider


# Load variables from .env.local.
#
# This makes GROQ_API_KEY and GROQ_MODEL available
# through os.getenv().
load_dotenv(".env.local")


# Retrieve the Groq API key.
#
# The API key should never be hardcoded in this file.
api_key = os.getenv("GROQ_API_KEY")

# Retrieve the model identifier.
#
# If GROQ_MODEL is missing, use GPT-OSS 120B as the default.
model = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# Check whether the API key exists.
#
# This prevents us from initializing the client
# without valid configuration.
if not api_key:
    raise ValueError("GROQ_API_KEY is missing")


# Create an instance of GroqProvider.
#
# The constructor receives the API key and model.
provider = GroqProvider(
    api_key=api_key,
    model=model
)


# Send a test prompt to the model.
#
# generate_text() internally sends this prompt
# to Groq and returns the generated response.
response = provider.generate_text(
    "Explain what a YouTube video transcript is in three sentences."
)


# Display the generated response.
print("\nGenerated response:\n")
print(response)
