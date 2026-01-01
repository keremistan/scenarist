from typing import Literal
from pydantic import BaseModel, Field
import dspy
from app.tools.logging_template import setup_logging
from app.tools.retrieve import SceneRetriever


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


class ScoreCard(BaseModel):
    coherence_score: int = Field(description="does the text make sense. 5 means that the scene makes perfect sense and 1 means that the scene makes no sense.", ge=1, le=5)
    style_adherence_score: int = Field(description="does the text adhere the reference scene. The style adherence refers to the content and not the formatting of reference text. 5 means a great adherence and 1 means no adherence", ge=1, le=5)
    critique: str = Field(description="The final evaluation text. Explain shortly WHY you gave these scores based on the comparison.")


class SceneWriter(dspy.Module):

    def __init__(self):
        self.keyword_extractor = dspy.Predict(dspy.make_signature("scene_gist -> dramatic_keywords: list[str]",
                                                                  instructions="dramatic_keywords describe the scene_gist and these are to be used for document retrieval."))
        self.document_retriever = SceneRetriever()
        self.scene_writer = dspy.ChainOfThought(dspy.make_signature("scene_gist, reference_scenes: list[str] -> generated_scene"))

        self.logger = setup_logging("SceneWriter")


    def forward(self, scene_gist):
        # keyword extraction
        dramatic_keywords: list[str] = self.keyword_extractor(scene_gist=scene_gist).dramatic_keywords
        self.logger.info("dramatic_keywords: {}".format(dramatic_keywords))

        # reference scene retrieval
        reference_scenes = [doc.page_content for doc in self.document_retriever.query(query_text=", ".join(dramatic_keywords))]
        self.logger.info("reference_scenes: {}".format(reference_scenes))

        # writing the scene
        scene_writer_response = self.scene_writer(scene_gist=scene_gist, reference_scenes=reference_scenes)
        self.logger.info("scene_writer_response: {}".format(scene_writer_response))

        # log the written scene
        generated_scene = scene_writer_response.generated_scene
        self.logger.info("generated_scene: {}".format(generated_scene[:25]))

        return scene_writer_response
