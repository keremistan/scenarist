from typing import Any, Union
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from models import SceneRequest, SceneResponse
from showrunner.logging_template import setup_logging
from showrunner.retrieve import SceneRetriever
from showrunner.story_guidelines import story_guideline

logger = setup_logging("generation")

has_anything_loaded = load_dotenv()

if not has_anything_loaded:
    raise ValueError("No .env file found")


# a catcher/handler from the API call
# it might use model response for further processing --> it would extract the style plan, logical plan, reference scenes used and generated scene
# or not.

def catch_api_request():
    pass

def process_response_of_writer_model(scene_request: SceneRequest) -> SceneResponse:
    # it might use model response for further processing --> it would extract 
    # the style plan, 
    # logical plan, 
    # reference scenes used,
    # generated scene,
    # evaluate the response
    
    writing_response = write_scene(scene_request.user_prompt, scene_request.writer_model)
    
    generated_scene = extract_generated_scene(writing_response)

    return SceneResponse(
        generated_scene=generated_scene,
        style_plan="todo",
        logical_plan="todo",
        referenced_scenes=[],
        critique_score=0.0,
        critique_text="todo"
    )

def extract_generated_scene(writing_response: Any) -> str:
    logger.info("starting the extraction of generated scene")
    
    for message in writing_response.get('messages', []):
        try:
            logger.info(message.content) if message.content != "" else logger.info(message.additional_kwargs["reasoning_content"])
        except Exception as e:
            logger.error("problem happened when logging. This:\n {}\n".format(e))
            
    most_recent_message = writing_response['messages'][-1].content
    
    logger.info("most recent message: \n{}".format(most_recent_message))
    
    return most_recent_message
    

def extract_style_plan_from_generated_scene():
    pass

def extract_logical_plan_from_generated_scene():
    pass

def extract_reference_scenes_from_generated_scene():
    pass

def evaluate_the_generated_scene():
    pass


def write_scene(
    user_prompt: str,
    writer_model: str = 'gpt-oss',
    temperature_of_writer: float = 0.7,
    # ) -> Union[dict[str, Any], Any]:
    ) -> str:
    
    logger.info("starting with writing the scene")
    
    if writer_model == 'gpt-5.2':
        chat_model = init_chat_model('gpt-5.2', model_provider='openai')
    else:
        chat_model = ChatOllama(model='gpt-oss:20b', reasoning='medium')

    logger.info("chat model initialized.")
    
    # get keywords that represent what emotion user wants to deliver in his story
    scene_retrieval_response = chat_model.invoke([
        {'role': 'system', 'content': """
         Convert the topic into DRAMATIC KEYWORDS.
    
            Example:
            - User: "A sad breakup" -> Query: "melancholy slow pacing silence heartbreak"
            - User: "Funny argument" -> Query: "sitcom banter snappy fast-paced comedy"
         """},
        {'role': 'user', 'content': user_prompt}
    ])
    
    # if already string, use it. Else, convert it to string.
    scene_retrieval_query = scene_retrieval_response.content if isinstance(scene_retrieval_response.content, str) else scene_retrieval_response.content.__str__()
    logger.info("scene_retrieval_query:\n {}".format(scene_retrieval_query))
    
    # get the scenes
    reference_scenes = get_reference_scenes(scene_retrieval_query)
    logger.info("reference_scenes:\n {}".format(reference_scenes))
    
    
    logical_plan_response = chat_model.invoke([
        {'role': 'system', 'content': """
         You are an expert in storytelling and screenwriting.
         Analyze the REFERENCE SCENES and output a "LOGICAL PLAN" that make the story coherent and logical while adhering the STORY GUIDELINES
         You will use this PLAN later to write the story.
         Do NOT use specifics from REFERENCE SCENES such as character names, locations, objects, etc.

         FORMAT:
         --- LOGICAL PLAN ---
         1. Story arc: what the actual story is
         2. Characters: who the characters are? what are their relations to each other? How are they moving the story forward?
         3. Location: where the story takes place? why is it actually this place? how is this place relevant for the story?
    
    
         --- STORY GUIDELINES ---
         {}
         --- REFERENCE SCENES ---
         {}
         """.format(story_guideline, reference_scenes)},
        {'role': 'user', 'content': user_prompt}
    ])

    # if already string, use it. Else, convert it to string.
    logical_plan = logical_plan_response.content if isinstance(logical_plan_response.content, str) else logical_plan_response.content.__str__()
    logger.info("logical_plan:\n {}".format(logical_plan))
    


    style_plan_response = chat_model.invoke([
        {'role': 'system', 'content': """
         You are an expert in storytelling and screenwriting.
         Analyze the REFERENCE SCENES and output a "STYLE PLAN" that make the story similar to the REFERENCE SCENES while adhering the STORY GUIDELINES and LOGICAL PLAN
         You will use this PLAN later to write the story.

         FORMAT:
         --- STYLE PLAN ---
         1. Pacing Analysis: (e.g. "Fast, short sentences" or "Slow, monologues")
         2. Subtext Strategy: (How the characters hide their true feelings)
         3. Vocabulary Rules: (Specific words or grammar to use/avoid)
    
    
         --- STORY GUIDELINES ---
         {}
         --- REFERENCE SCENES ---
         {}
         --- LOGICAL PLAN ---
         {}         
         """.format(story_guideline, reference_scenes, logical_plan)},
        {'role': 'user', 'content': user_prompt}
    ])

    # if already string, use it. Else, convert it to string.
    style_plan = style_plan_response.content if isinstance(style_plan_response.content, str) else style_plan_response.content.__str__()
    logger.info("style_plan:\n {}".format(style_plan))

    the_new_scene_response = chat_model.invoke([
        {'role': 'system', 'content': """
         You are an expert in storytelling and screenwriting.
         WRITE the scene while following the LOGICAL PLAN and STYLE PLAN and STORY GUIDELINES


         STORY GUIDELINES:
         {}
         LOGICAL PLAN:    
         {}
         STYLE PLAN:    
         {}
         """.format(story_guideline, logical_plan, style_plan)},
        {'role': 'user', 'content': user_prompt}
    ])

    # if already string, use it. Else, convert it to string.
    the_new_scene = the_new_scene_response.content if isinstance(the_new_scene_response.content, str) else the_new_scene_response.content.__str__()
    logger.info("the_new_scene:\n {}".format(the_new_scene))
    

    
    # model_output = extract_tool_and_latest_message_from_model_response(response)
    # if model_output:
    #     draft_text, retrieved_references = model_output

    #     editor_prompt = f"""
    #     You are a ruthless Script Editor.
        
    #     ORIGINAL REFERENCES:
    #     {retrieved_references}
        
    #     DRAFT SCENE:
    #     {draft_text}
        
    #     TASK:
    #     Compare the DRAFT to the REFERENCES.
    #     1. Did the draft actually use the story telling elements to deliver the targeted feelings?
    #     2. Did it use the subtext approach?
        
    #     If the draft is perfect, output: "PERFECT".
    #     If not, output a REVISED VERSION of the scene that fixes the style issues.
    #     """
    #     logger.info("calling the editor")
    #     editor = ChatOllama(model='gpt-oss:20b', reasoning='medium')
    #     editor_response = editor.invoke([{'role': 'system', 'content': editor_prompt}])
        
    #     logger.info("editor_response:\n {}\n".format(editor_response))
        
    #     if editor_response.content != "PERFECT":
    #         logger.info("it is not perfect")
    #         if isinstance(editor_response.content, str):
    #             response_message = editor_response.content
    #         else:
    #             response_message = editor_response.content.__str__()
                
    #         logger.info("revised draft:\n {}\n".format(response_message))
            
    #         # while the generated scene is from the editor, the response is still from the previous writer
    #         return (response_message, response)
    
        
    return the_new_scene


def get_reference_scenes(scene_retrieval_query: str) -> str:
    """
    Call this tool to find screenplay examples.
    
    CRITICAL USAGE INSTRUCTION:
    Do not just pass the user's topic. You must convert the topic into DRAMATIC KEYWORDS.
    
    Example:
    - User: "A sad breakup" -> Query: "melancholy slow pacing silence heartbreak"
    - User: "Funny argument" -> Query: "sitcom banter snappy fast-paced comedy"
    
    Args:
        scene_retrieval_query: A string of dramatic keywords (mood, pacing, subtext).
    """    

    retriever = SceneRetriever()
    
    retrieved_scenes = retriever.query(scene_retrieval_query)

    scenes_as_single_text = ""
    for i, retrieved_scene in enumerate(retrieved_scenes):
        scenes_as_single_text += """
        --- Reference Scene {} --
        {}
        
        """.format(i+1, retrieved_scene.page_content)

    # logger.info("This concatenated scenes string is returned:\n {}".format(scenes_as_single_text))

    return scenes_as_single_text

if __name__ == '__main__':
    logger.info("added logical-conclusion field to the story guideline. inserted logical plan to the style creation.")
    write_scene("An unexpected turn of events happen and raises the stakes")