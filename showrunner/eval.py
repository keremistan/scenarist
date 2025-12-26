from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import ToolMessage
import regex as re

class ScoreCard(BaseModel):
    coherence: int = Field(description="does the text make sense", ge=1, le=5)
    style_adherence: int = Field(description="does the text adhere the reference scene. The style adherence refers to the content and not the formatting of reference text", ge=1, le=5)
    reasoning: str = Field(description="explanation of why you graded as you did")

def extract_tool_and_latest_message_from_model_response(model_response: Any) ->  Optional[tuple[str, str]]:
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
        print("no tool is called. Therefore, no style is retrieved or used.")        
    except Exception as e:
        print("a problem happened when searching for a tool message with '--- Reference Scene'.\n{}".format(e))
        


def evaluate(model_response: Any, user_intent: str) -> Optional[ScoreCard]:
    scenes_extraction = extract_tool_and_latest_message_from_model_response(model_response)
    if scenes_extraction:
        generated_scene, reference_scenes = scenes_extraction
    else:
        print("extraction of scenes failed")
        return
    
    the_judge = ChatOllama(model='gpt-oss:20b', reasoning='high'
                           ).with_structured_output(ScoreCard)
    
    system_prompt = {
        "role": "system",
        "content": "You are a film critic. Compare this GENERATED SCENE against these REFERENCE SCENES and the USER INTENT. Grade it."
    }
    
    input_prompt = {
        "role": "user",
        "content": "USER INTENT: {}\nREFERENCE SCENES: {}\nGENERATED SCENE: {}\n".format(user_intent, reference_scenes, generated_scene)
    }
    
    response = the_judge.invoke([
        system_prompt,
        input_prompt
    ])
    
    print("eval result:\n{}\n\n\n".format(response))
    
    return response # type: ignore #TODO: this might be broken
    
