from app.tools.logging_template import setup_logging
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import dspy
import os
from openinference.instrumentation.dspy import DSPyInstrumentor

logger = setup_logging("configs")

has_anything_loaded = load_dotenv()
if not has_anything_loaded:
    logger.error("env file could not be found")
    raise ValueError("No .env file found")

ollama_base_url = os.environ["OLLAMA_DOCKER_SERVICE"]

#todo: passing the model from ui input to the config
class Config:
    def __init__(self, **kwargs):
        """
        kwargs can include 'lm_model' to pass to dspy. The default is gpt-oss:20b.
        """
        # set up the langfuse
        self.langfuse = get_client()
        DSPyInstrumentor().instrument() # langfuse config to trace dspy activities
        self.langfuse_callback_handler = CallbackHandler()

        # set up the dspy
        lm_model = kwargs.get("lm_model", "ollama_chat/gpt-oss:20b")
        lm = dspy.LM(lm_model, api_base="http://{}:11434".format(ollama_base_url), api_key="")
        dspy.configure(lm=lm)

        # verify langfuse setup
        self.check_langfuse_auth()

    def check_langfuse_auth(self):
        # Verify connection
        if self.langfuse.auth_check():
            logger.info("Langfuse client is authenticated and ready!")
        else:
            logger.error("Authentication failed. Please check your credentials and host.")


config = Config()
