# os allows us to access environment variables.
import os

# Import our common provider interface.
from providers.base import LLMProvider

# Import the actual Groq implementation.
from providers.groq_provider import GroqProvider


def get_provider() -> LLMProvider:
    """
    Creates and returns an LLM provider based on configuration.

    The application does not need to know which provider
    is being initialized. It simply calls get_provider().
    """

    # Read the selected provider from environment variables.
    # If it is not configured, default to Groq.
    provider_name = os.getenv("LLM_PROVIDER", "groq").lower()

    # Check whether Groq has been selected.
    if provider_name == "groq":

        # Retrieve Groq credentials.
        api_key = os.getenv("GROQ_API_KEY")

        # Retrieve the selected model.
        # Use GPT-OSS 120B if no model is configured.
        model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        # Fail early if the API key is missing.
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        # Create and return the Groq provider instance.
        return GroqProvider(
            api_key=api_key,
            model=model
        )

    # Reject unsupported provider names.
    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )
