from typing import Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from statistics import mean
from api.models import SceneRequest, SceneResponse
from tools.logging_template import setup_logging
from tools.retrieve import SceneRetriever
from tools.story_guidelines import story_guideline
from tools.evaluate import evaluate_scene

logger = setup_logging("generation")

has_anything_loaded = load_dotenv()

if not has_anything_loaded:
    raise ValueError("No .env file found")

def write_scene(scene_request: SceneRequest, test_response: bool = False) -> SceneResponse:    
    """
    This function accepts a scene_request and does the followings:
    1. choose the writer model
    2. generate keywords that describe the emotional aspect of user's prompt
    3. retrieve the most relevant scenes from vector store
    4. create a logical plan using story guidelines and reference scenes
    5. on top, create a style plan
    6. write the scene
    7. return it
    """
    user_prompt = scene_request.user_prompt
    writer_model = scene_request.writer_model
    temperature_of_writer = scene_request.temperature_of_writer
    
    if test_response:
        return SceneResponse(
            generated_scene="the_new_scene",
            style_plan="style_plan",
            logical_plan="logical_plan",
            referenced_scenes=["reference_scenes", "reference_scenes", "reference_scenes"],
            critique_score= mean([4, 5]),
            critique_text="evaluation.critique"
        )
    
    logger.info("starting with writing the scene")
    
    if writer_model == 'gpt-5.2':
        chat_model = init_chat_model('gpt-5.2', model_provider='openai', temperature=temperature_of_writer)
    else:
        chat_model = ChatOllama(model='gpt-oss:20b', reasoning='high', temperature=temperature_of_writer)

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
    
    # combine the scenes inside a single str
    reference_scenes_as_single_text = ""
    for i, retrieved_scene in enumerate(reference_scenes):
        reference_scenes_as_single_text += """
        --- Reference Scene {} --
        {}
        
        """.format(i+1, retrieved_scene)
    
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
         """.format(story_guideline, reference_scenes_as_single_text)},
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
         """.format(story_guideline, reference_scenes_as_single_text, logical_plan)},
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
    
    evaluation = evaluate_scene(the_new_scene, user_prompt, reference_scenes_as_single_text)
    
    if evaluation is None:
        logger.error("scene evaluation failed")
        raise RuntimeError("scene evaluation result could not be obtained")
    
    return SceneResponse(
        generated_scene=the_new_scene,
        style_plan=style_plan,
        logical_plan=logical_plan,
        referenced_scenes=reference_scenes,
        critique_score= mean([evaluation.coherence_score, evaluation.style_adherence_score]),
        critique_text=evaluation.critique
    )


def get_reference_scenes(scene_retrieval_query: str) -> list[str]:
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

    return [scene.page_content for scene in retrieved_scenes]


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
    

if __name__ == '__main__':
    logger.info("added evaluation to the api. adjusted it so that it works with a chain output.")
    write_scene(SceneRequest(user_prompt="An unexpected turn of events happen and raises the stakes"))