# Load local environment variables.
from dotenv import load_dotenv

# Import the factory function.
from providers.factory import get_provider

load_dotenv(".env.local")


# Ask the factory to create the configured provider.
provider = get_provider()

# Display the provider class name.
print("Provider:", type(provider).__name__)

# Test the provider using a simple prompt.
response = provider.generate_text(
    "Explain artificial intelligence in three sentences."
)

# Display the generated response.
print("\nGenerated response:\n")
print(response)
