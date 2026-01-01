from typing import Optional
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from app.tools.logging_template import setup_logging
from app.api.models import ScoreCard

logger = setup_logging("eval")


def evaluate_scene(generated_scene: str, user_intent: str, reference_scenes: str, is_openai: bool = False) -> Optional[ScoreCard]:
    
    if is_openai:
        the_judge = init_chat_model('gpt-5.2', model_provider='openai')
    else:
        the_judge = ChatOllama(model='gpt-oss:20b', reasoning='high'
                           ).with_structured_output(ScoreCard)
    
    system_prompt = {
        "role": "system",
        "content": """You are a Film Critic. Compare the GENERATED SCENE to the REFERENCE SCENES. This comparison must be based on the storytelling quality and not surface level content.

            TASK:
            1. Write a critique.
            2. Depending on how successful the GENERATED SCENE is assign coherence_score and style_adherence_score where 5 mean High/Succeded and 1 means Low/Failed.
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
    
