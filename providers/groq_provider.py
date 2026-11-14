# Import Groq's official Python SDK.
#
# This SDK handles HTTP communication with Groq's API,
# authentication, request formatting and response parsing.
from groq import Groq


# Import our custom exception.
#
# This allows us to expose a consistent error type
# instead of making the rest of our application
# dependent on Groq-specific exceptions.
from providers.base import LLMProviderError


# Define our concrete provider.
#
# This class is responsible for communicating with Groq.
class GroqProvider:

    # Constructor: executed automatically when we create an object.
    #
    # api_key: Secret key used to authenticate with Groq.
    #
    # model: Model identifier, such as openai/gpt-oss-120b.
    def __init__(self, api_key: str, model: str):

        # Create a Groq client.
        #
        # The client manages communication with the Groq API.
        # We store it in self.client so other methods
        # in this class can reuse it.
        self.client = Groq(api_key=api_key)

        # Store the selected model.
        #
        # Using self.model means we don't need to hardcode
        # the model name inside every API request.
        self.model = model

    # This method generates text using the selected Groq model.
    #
    # prompt: The actual instruction sent to the model.
    #
    # json_mode: If True, request JSON-formatted output.
    def generate_text(
        self,
        prompt: str,
        json_mode: bool = False
    ) -> str:

        # Validate the prompt before making an API request.
        #
        # strip() removes whitespace from both ends.
        # If nothing remains, the prompt is empty.
        #
        # There is no reason to spend an API request
        # on an empty prompt.
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # API communication can fail for several reasons:
        # - Invalid API key
        # - Rate limits
        # - Model unavailable
        # - Network problems
        #
        # try/except lets us handle these failures.
        try:

            # Prepare the parameters that will be sent to Groq.
            #
            # A dictionary makes it easy to conditionally add
            # parameters before making the API request.
            request_params = {

                # Tell Groq which model should process the prompt.
                "model": self.model,

                # Messages represent the conversation sent to the model.
                "messages": [
                    {
                        # user means this message comes from the user.
                        "role": "user",

                        # Pass the actual prompt received by this method.
                        "content": prompt
                    }
                ]
            }

            # Check whether the caller requested JSON output.
            if json_mode:

                # Ask Groq for JSON-formatted output.
                #
                # This is useful when our agents need structured
                # information that can be parsed using json.loads().
                request_params["response_format"] = {
                    "type": "json_object"
                }

            # Send the request to Groq.
            #
            # **request_params unpacks the dictionary.
            #
            # For example:
            # create(**{"model": "abc", "messages": [...]})
            #
            # is equivalent to:
            # create(model="abc", messages=[...])
            response = self.client.chat.completions.create(
                **request_params
            )

            # Extract the generated text from the response.
            #
            # choices[0] selects the first generated response.
            # message accesses the response message.
            # content contains the actual generated text.
            content = response.choices[0].message.content

            # Check whether Groq returned an empty response.
            #
            # An empty response should be treated as a failure,
            # rather than pretending generation succeeded.
            if not content:
                raise LLMProviderError("Empty response from Groq")

            # Return the generated content to whoever called
            # generate_text().
            return content

        # If our own custom exception occurs, preserve it.
        except LLMProviderError:
            raise

        # Catch other errors from the Groq SDK or request processing.
        except Exception as exc:

            # Convert the original error into our common
            # application-level exception.
            #
            # 'from exc' preserves the original exception
            # as the underlying cause for debugging.
            raise LLMProviderError(
                f"Groq generation failed: {exc}"
            ) from exc
