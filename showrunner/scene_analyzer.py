from pydantic import BaseModel, Field


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
    surface_level_happening: str = Field(description="what is visible on the surface within this scene")
    subtext_level_happening: str = Field(description="what is actually told regarding story within this scene")
    reader_reaction: str = Field(description="how the reader should feel when reading this scene")
    
    
    
    

    