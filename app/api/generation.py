from statistics import mean
from app.api.models import SceneRequest, SceneResponse, SceneWriter
from app.tools.evaluate import evaluate_scene
from app.tools.logging_template import setup_logging

logger = setup_logging("generation")

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
    
    # if writer_model == 'gpt-5.2':
    #     chat_model = init_chat_model('gpt-5.2', model_provider='openai', temperature=temperature_of_writer)
    # else:
    #     chat_model = ChatOllama(model='gpt-oss:20b', reasoning='high', temperature=temperature_of_writer)

    chat_model = SceneWriter()

    import os
    logger.info("current working dir: {}\n".format(os.getcwd()))

    chat_model.load("app/data/optimized_scene_generator.json")
    logger.info("chat model initialized.")

    generated_scene = chat_model(scene_gist=user_prompt).generated_scene
    logger.info("generated the scene")

    evaluation = evaluate_scene(chat_model.generated_scene, user_prompt, chat_model.reference_scenes)
    logger.info("evaluation results: {}".format(evaluation))

    # todo: the api is filled with placeholders. Either change the api or fill the values
    return SceneResponse(
        generated_scene=generated_scene,
        style_plan="style_plan",
        logical_plan="logical_plan",
        referenced_scenes=["reference_scenes"],
        critique_score= 1,
        critique_text=str(evaluation)
    )


# def get_reference_scenes(scene_retrieval_query: str) -> list[str]:
#     """
#     Call this tool to find screenplay examples.
#
#     CRITICAL USAGE INSTRUCTION:
#     Do not just pass the user's topic. You must convert the topic into DRAMATIC KEYWORDS.
#
#     Example:
#     - User: "A sad breakup" -> Query: "melancholy slow pacing silence heartbreak"
#     - User: "Funny argument" -> Query: "sitcom banter snappy fast-paced comedy"
#
#     Args:
#         scene_retrieval_query: A string of dramatic keywords (mood, pacing, subtext).
#     """
#
#     retriever = SceneRetriever()
#
#     retrieved_scenes = retriever.query(scene_retrieval_query)
#
#     return [scene.page_content for scene in retrieved_scenes]
#

# def extract_generated_scene(writing_response: Any) -> str:
#     logger.info("starting the extraction of generated scene")
#
#     for message in writing_response.get('messages', []):
#         try:
#             logger.info(message.content) if message.content != "" else logger.info(message.additional_kwargs["reasoning_content"])
#         except Exception as e:
#             logger.error("problem happened when logging. This:\n {}\n".format(e))
#
#     most_recent_message = writing_response['messages'][-1].content
#
#     logger.info("most recent message: \n{}".format(most_recent_message))
#
#     return most_recent_message
    

if __name__ == '__main__':
    logger.info("added evaluation to the api. adjusted it so that it works with a chain output.")
    write_scene(SceneRequest(user_prompt="The hero has to be vulnerable in front of a person who is the most valuable to him"))