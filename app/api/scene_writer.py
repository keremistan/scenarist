from typing import Optional

import dspy

from app.tools.logging_template import setup_logging
from app.tools.retrieve import SceneRetriever


class SceneWriter(dspy.Module):

    def __init__(self):
        super().__init__()
        self.keyword_extractor = dspy.Predict(dspy.make_signature("scene_gist -> dramatic_keywords: list[str]",
                                                                  instructions="dramatic_keywords describe the scene_gist and these are to be used for document retrieval."))
        self.document_retriever = SceneRetriever()
        self.scene_writer = dspy.ChainOfThought(dspy.make_signature("scene_gist, reference_scenes: list[str] -> generated_scene"))

        self.logger = setup_logging("SceneWriter")

        self.dramatic_keywords = []
        self.reference_scenes = []
        self.generated_scene: Optional[str] = None

    def forward(self, scene_gist):
        # keyword extraction
        dramatic_keywords: list[str] = self.keyword_extractor(scene_gist=scene_gist).dramatic_keywords
        self.logger.info("dramatic_keywords: {}".format(dramatic_keywords))
        self.dramatic_keywords = dramatic_keywords

        # reference scene retrieval
        reference_scenes = [doc.page_content for doc in self.document_retriever.query(query_text=", ".join(dramatic_keywords))]
        self.logger.info("reference_scenes: {}".format(reference_scenes))
        self.reference_scenes = reference_scenes

        # writing the scene
        scene_writer_response = self.scene_writer(scene_gist=scene_gist, reference_scenes=reference_scenes)
        self.logger.info("scene_writer_response: {}".format(scene_writer_response))

        # log the written scene
        generated_scene = scene_writer_response.generated_scene
        self.logger.info("generated_scene: {}".format(generated_scene[:25]))
        self.generated_scene = generated_scene

        return scene_writer_response
