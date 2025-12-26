from typing import Any, Union
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_ollama import ChatOllama
from retrieve import SceneRetriever
from series_reference import my_series_reference
import os
from dotenv import load_dotenv
import regex as re

from eval import evaluate

has_anything_loaded = load_dotenv()

if not has_anything_loaded:
    raise ValueError("No .env file found")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

retriever = SceneRetriever()

@tool
def get_reference_scenes(scene_retrieval_query: str) -> str:
    """
    This function searches for relevant scenes in the database and retrieves them.
    
    Args:
        scene_retrieval_query: specifies what kind of scenes should be retrieved
        
    Return:
        All the relevant scenes that are concatenated as a single text
    """
    
    retrieved_scenes = retriever.query(scene_retrieval_query)

    scenes_as_single_text = ""
    for i, retrieved_scene in enumerate(retrieved_scenes):
        scenes_as_single_text += """
        --- Reference Scene {} --
        {}
        
        """.format(i+1, retrieved_scene.page_content)

    # print("This concatenated scenes string is returned: {}".format(scenes_as_single_text))

    return scenes_as_single_text

def write_scene(command: str, is_openai: bool = False, do_evaluate: bool = False, return_model_response: bool = False) -> Union[str, tuple[str, Any]]:
    if is_openai:
        chat_model = init_chat_model('gpt-5.2', model_provider='openai')
    else:
        chat_model = ChatOllama(model='gpt-oss:20b', reasoning='medium')

    print("chat model initialized.")

    writer = create_agent(
        chat_model,
        checkpointer=InMemorySaver(),
        tools=[get_reference_scenes],
        system_prompt="""
        You are an expert screenwriter.
        Use reference scenes. Their storytelling elements are important; not the specific actions, locations or characters, but how they deliver the emotion. 
        
        Scenes: {}
        """.format(my_series_reference)
        # BEFORE CHANGING THE PROMPT, i want to see how the increase in reference scenes affects the model output
        # You are an expert screenwriter.
        # You can use the reference scenes.
        # Stick to this guideline when writing: {}
        # """.format(my_series_reference)
        )
    
    response = writer.invoke({
        'messages': {'role': 'user', 'content': command}},  # type: ignore
                               {'configurable': {'thread_id': 7}})
    print(response, "\n\n")

    for message in response.get('messages', []):
        try:
            print(message.content) if message.content != "" else message.additional_kwargs["reasoning_content"]
        except Exception as e:
            print("problem happened when printing. This: {}\n".format(e))
            
    most_recent_message = response['messages'][-1].content
    
    print("most recent message: \n{}".format(most_recent_message))
    
    if return_model_response:
        return (most_recent_message, response)
    else:
        return most_recent_message
    
    
# scene = write_scene("write an introductory scene", do_evaluate=True)
# print("\n\nResult:\n {}".format(scene))
