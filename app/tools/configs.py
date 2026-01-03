from app.tools.logging_template import setup_logging
from langfuse import get_client
from dotenv import load_dotenv
import dspy
import os
from openinference.instrumentation.dspy import DSPyInstrumentor

logger = setup_logging("configs")

has_anything_loaded = load_dotenv()
if not has_anything_loaded:
    logger.error("env file could not be found")
    raise ValueError("No .env file found")


langfuse = get_client()
# langfuse config to trace dspy activities
DSPyInstrumentor().instrument()


ollama_base_url = os.environ["OLLAMA_DOCKER_SERVICE"]
lm = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://{}:11434".format(ollama_base_url), api_key="")
dspy.configure(lm=lm)


# Verify connection
if langfuse.auth_check():
    logger.info("Langfuse client is authenticated and ready!")
else:
    logger.error("Authentication failed. Please check your credentials and host.")

class Config:
    def __init__(self):
        pass

config = Config()
