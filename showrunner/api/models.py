from typing import Literal
from pydantic import BaseModel, Field


class SceneRequest(BaseModel):
    user_prompt: str = Field(description="The user's prompt to generate a new scene")
    writer_model: Literal['gpt-oss', 'gpt-5.2'] = Field(description="The model used to write the requested scene", default='gpt-oss')
    temperature_of_writer: float = Field(description="The temparature for how creative the writer should behave", ge=0, le=1.0, default=0.7)
    #TODO: parametrise the series_reference as well
    
class SceneResponse(BaseModel):
    generated_scene: str = Field(description="The scene that is newly generated based on the user's prompt")
    style_plan: str = Field(description="The generated plan for mimicking the style of reference scenes")
    logical_plan: str = Field(description="The generated logical plan for writing a coherent scene")
    referenced_scenes: list[str] = Field(description="The used scenes as reference")
    critique_score: float = Field(description="The evaluation score for the generated scene", ge=1, le=5)
    critique_text: str = Field(description="The reasoning behind why the score is determined as it is")
