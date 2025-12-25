from pydantic import BaseModel, Field

class SeriesReference(BaseModel):
    theme: str = Field(description="this is the theme or message that holds everything together in the series")
    genre: str = Field(description="The genre of the series")
    characters: list[str] = Field(description="This list contains the characters of the series")
    tone_guidelines: list[str] = Field(description="this tells how the tone of the series should be")



my_series_reference = SeriesReference(
    theme="kids must be taught that they deserve respect so that they can be mature individuals later on in life",
    genre="sitcom",
    characters=["Ahmet faces challenges when communicating with others, including his friends, colleagues and lovers. He is not taught be an adult and cannot handle the adult life."],
    tone_guidelines=["light, but meaningful and yet funny"]
)