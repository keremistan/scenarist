from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import Optional
from pprint import pprint

# when searching for a scene, what would i need it to reference by?
# - type? whether it is a comedic, tragic, action, --introductory,--, 
# - audience's emotion? what the watcher feels. happy, sad, tension, suspense
# - location? whether it occurs outside, inside, restaurant, internet cafe, etc.
# - how many characters are interacting?
# - style of narrating the story? mundane talk when doing something serious, 
# - what happens in story? e.g. character is at rock bottom, flow reverses and good guy starts winning, character's arc is affected
# - elements of how to introduce an idea to the story 
# - creating the world of story with locations, props, chracters, etc.
# -- what props are used? what locations are used?
# - how the story is moved forward. 
# - what happens at the surface level vs what happens really (the subtext)
# -- what can happen at surface: eating, going, entering, talking, arguing, stealing, etc.
# -- what can happen as subtext? following a dream, defending ego, proving someone wrong, etc.

class SceneAnalysis(BaseModel):
    happening: str = Field(description="what concrete actions are happening")
    subtext_level_happening: str = Field(description="what is actually told regarding story within this scene")
    reader_reaction: str = Field(description="how the reader should feel when reading this scene")
    
# model initialization is here because it should not be re-initialized at every function call
analyzer_model = ChatOllama(
    model="gpt-oss:20b", 
    temperature=0,
    ).with_structured_output(SceneAnalysis)

def analyze_scene(scene: str) -> Optional[SceneAnalysis]:

    system_prompt = {
        "role": "system",
        "content": "Analyze the given scene and extract the features from it. Especially take care of getting the subtext correct."
    }

    analysis_response = analyzer_model.invoke(
        [
            system_prompt,
            {
                "role": "user",
                "content": "analyze this scene: {}".format(scene)
            }
        ]
    )

    pprint("analysis_response: \n{}".format(analysis_response))
    
    try:
        scene_analysis = SceneAnalysis.model_validate(analysis_response)
        return scene_analysis
    except:
        print("an error occured when validating the model's output")
