from pydantic import BaseModel, Field

class StoryGuidelines(BaseModel):
    theme: str = Field(description="this is the theme or message that holds everything together in the series")
    genre: str = Field(description="The genre of the series")
    characters: list[str] = Field(description="This list contains the characters of the series")
    tone_guidelines: list[str] = Field(description="this tells how the tone of the series should be")
    logic_guidelines: list[str] = Field(description="this tells what logical relations should exist")



story_guideline = StoryGuidelines(
    theme="kids must be taught that they deserve respect so that they can be mature individuals later on in life",
    genre="sitcom",
    characters=["Ahmet faces challenges when communicating with others, including his friends, colleagues and lovers. He is not taught be an adult and cannot handle the adult life."],
    tone_guidelines=["light, but meaningful and yet funny"],
    logic_guidelines=["events should happen for logical reasons.", "the story line must follow a consistent path. For example, when two characters are cooking in the kitchen, an alien should not appear in the hallway."]
)
