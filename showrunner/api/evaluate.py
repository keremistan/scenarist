from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import ToolMessage
from langchain.chat_models import init_chat_model
import regex as re
from showrunner.logging_template import setup_logging
from showrunner.api.models import ScoreCard

logger = setup_logging("eval")


def evaluate_scene(generated_scene: str, user_intent: str, reference_scenes: str, is_openai: bool = False) -> Optional[ScoreCard]:
    
    if is_openai:
        the_judge = init_chat_model('gpt-5.2', model_provider='openai')
    else:
        the_judge = ChatOllama(model='gpt-oss:20b', reasoning='high'
                           ).with_structured_output(ScoreCard)
    
    system_prompt = {
        "role": "system",
        "content": """You are a Film Critic. Compare the GENERATED SCENE to the REFERENCE SCENES.

            SCORING RULES:
            - The scale is 1 to 5.
            - 5 = MASTERPIECE (Matches style perfectly).
            - 1 = GARBAGE (Completely ignores style).
            - Do NOT use the German grading system (where 1 is good). Use the Standard 5-Star system.

            TASK:
            1. Write a critique.
            2. Assign scores based on the 1 (Low) - 5 (High) scale.
        """
    }
    
    input_prompt = {
        "role": "user",
        "content": "USER INTENT: {}\nREFERENCE SCENES: {}\nGENERATED SCENE: {}\n".format(user_intent, reference_scenes, generated_scene)
    }
    
    response = the_judge.invoke([
        system_prompt,
        input_prompt
    ])
    
    logger.info("eval result:\n{}\n\n\n".format(response))
    
    return response # type: ignore #TODO: this might be broken
    
