from typing import Any, Union
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from showrunner.api.models import SceneRequest, SceneResponse
from logging_template import setup_logging
from showrunner.retrieve import SceneRetriever
from series_reference import story_guideline

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
            logger.error("problem happened when logging. This: {}\n".format(e))
            
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
    ) -> Union[dict[str, Any], Any]:
    
    logger.info("starting with writing the scene")
    
    if writer_model == 'gpt-5.2':
        chat_model = init_chat_model('gpt-5.2', model_provider='openai')
    else:
        chat_model = ChatOllama(model='gpt-oss:20b', reasoning='medium')

    logger.info("chat model initialized.")

    writer = create_agent(
        chat_model,
        # checkpointer=InMemorySaver(), # I'M REMOVING THIS. THIS CHANGE HASN'T BEEN BENCHMARKED
        tools=[get_reference_scenes],
        system_prompt="""
            You are a Ghostwriter. You must MIMIC the style of the reference scenes that you will fetch.

            CRITICAL PROCESS:
            Fetch the reference scenes
            You are FORBIDDEN from writing the scene immediately.
            You must first output a "LOGICAL PLAN" and then a "STYLE PLAN" where you analyze the reference scenes.
            You have to write the scene while following the both plans.

            FORMAT:
            --- LOGICAL PLAN ---
            1. Story arc: what the actual story is
            2. Characters: who the characters are? what are their relations to each other? How are they moving the story forward?
            3. Location: where the story takes place? why is it actually this place? how is this place relevant for the story?
            --- STYLE PLAN ---
            1. Pacing Analysis: (e.g. "Fast, short sentences" or "Slow, monologues")
            2. Subtext Strategy: (How the characters hide their true feelings)
            3. Vocabulary Rules: (Specific words or grammar to use/avoid)
            ------------------
            --- SCENE START ---
            [Write the scene here, strictly following the plans above]


            STORY GUIDELINE:
            {}
        """.format(story_guideline)
        )
    
    response = writer.invoke({
        'messages': {'role': 'user', 'content': user_prompt}})  # type: ignore
    # Removed the configuration/thread-id; not benchmarked yet.
    
    logger.info("writer's response:\n{}\n\n".format(response))


    
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
        
    #     logger.info("editor_response: {}\n".format(editor_response))
        
    #     if editor_response.content != "PERFECT":
    #         logger.info("it is not perfect")
    #         if isinstance(editor_response.content, str):
    #             response_message = editor_response.content
    #         else:
    #             response_message = editor_response.content.__str__()
                
    #         logger.info("revised draft: {}\n".format(response_message))
            
    #         # while the generated scene is from the editor, the response is still from the previous writer
    #         return (response_message, response)
    
        
    return response


@tool
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

    # logger.info("This concatenated scenes string is returned: {}".format(scenes_as_single_text))

    return scenes_as_single_text
