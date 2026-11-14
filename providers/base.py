# Protocol allows us to define a common structure (contract)
# that different LLM providers must follow.
from typing import Protocol


# LLMProvider defines what functionality an LLM provider
# must expose to the rest of our application.
#
# For example:
# - GroqProvider
# - TogetherProvider
# - LocalLLMProvider
#
# All these providers should implement generate_text().
class LLMProvider(Protocol):

    # self represents the current provider object.
    #
    # prompt: The instruction or question we want the LLM to answer.
    #
    # json_mode: Determines whether we want the model to return
    # structured JSON instead of ordinary text.
    #
    # -> str means this method should return a Python string.
    def generate_text(
        self,
        prompt: str,
        json_mode: bool = False
    ) -> str:

        # Ellipsis (...) means this is only a declaration.
        # The actual implementation belongs to individual providers.
        ...


# A custom exception shared by all our LLM providers.
#
# Instead of exposing different SDK errors from Groq, Together AI,
# or another provider, we can convert them into this common exception.
class LLMProviderError(Exception):

    # pass means this class does not need additional functionality
    # yet. It inherits normal exception behaviour from Exception.
    pass
