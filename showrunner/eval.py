from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import ToolMessage
from langchain.chat_models import init_chat_model
import regex as re
from logging_template import setup_logging

logger = setup_logging("eval")

class ScoreCard(BaseModel):
    coherence: int = Field(description="does the text make sense. 1 is bad and 5 is great.", ge=1, le=5)
    style_adherence: int = Field(description="does the text adhere the reference scene. The style adherence refers to the content and not the formatting of reference text. 1 means no adherence and 5 means a great mimicry", ge=1, le=5)
    critique: str = Field(description="The final evaluation text. Explain WHY you gave these scores based on the comparison.")

def extract_tool_and_latest_message_from_model_response(model_response: Any) ->  Optional[tuple[str, str]]:
    """
    Docstring for extract_tool_and_latest_message_from_model_response
    
    :param model_response: whatever the writer model responded to its invocation
    :type model_response: Any
    :return: generated_scene and reference_scenes as a tuple
    :rtype: tuple[str, str] | None
    """
    try:
        generated_scene = model_response['messages'][-1].content
        
        # get the tool message where 'reference scene' is embedded
        style_ref_message = [message for message in model_response['messages'] 
                            if isinstance(message, ToolMessage) 
                            and 
                            len(re.findall(
                                '--- Reference Scene ', message.content.__str__()
                                )) > 0 ].pop()

        # extract the content of that style message
        style_ref_content = style_ref_message.content
        # convert it to string explicitly only if it's not a string already
        reference_scenes = style_ref_content if isinstance(style_ref_content, str) else style_ref_content.__str__()
        
        return generated_scene, reference_scenes

    except IndexError as ie:
        logger.error("no tool is called. Therefore, no style is retrieved or used.")        
    except Exception as e:
        logger.error("a problem happened when searching for a tool message with '--- Reference Scene'.\n{}".format(e))

        
def evaluate(model_response: Any, user_intent: str, is_openai: bool = False) -> Optional[ScoreCard]:
    scenes_extraction = extract_tool_and_latest_message_from_model_response(model_response)
    if scenes_extraction:
        generated_scene, reference_scenes = scenes_extraction
    else:
        logger.info("extraction of scenes failed")
        return
    
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
    
