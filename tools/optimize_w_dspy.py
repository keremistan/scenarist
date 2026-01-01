import dspy
import mlflow
from tools.logging_template import setup_logging
from tools.retrieve import SceneRetriever

mlflow.dspy.autolog()

logger = setup_logging("dspy_optimizer")

lm = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434", api_key="")

dspy.configure(lm=lm)

class SceneWriter(dspy.Module):

    def __init__(self):
        self.keyword_extractor = dspy.Predict(dspy.make_signature("scene_gist -> dramatic_keywords: list[str]",
                                                                  instructions="dramatic_keywords describe the scene_gist and these are to be used for document retrieval."))
        self.document_retriever = SceneRetriever()
        self.scene_writer = dspy.ChainOfThought(dspy.make_signature("scene_gist, reference_scenes: list[str] -> generated_scene"))


    def forward(self, scene_gist):
        # keyword extraction
        dramatic_keywords: list[str] = self.keyword_extractor(scene_gist=scene_gist).dramatic_keywords
        logger.info("dramatic_keywords: {}".format(dramatic_keywords))

        # reference scene retrieval
        reference_scenes = [doc.page_content for doc in self.document_retriever.query(query_text=", ".join(dramatic_keywords))]
        logger.info("reference_scenes: {}".format(reference_scenes))

        # writing the scene
        scene_writer_response = self.scene_writer(scene_gist=scene_gist, reference_scenes=reference_scenes)
        logger.info("scene_writer_response: {}".format(scene_writer_response))

        # log the written scene
        generated_scene = scene_writer_response.generated_scene
        logger.info("generated_scene: {}".format(generated_scene[:25]))

        return scene_writer_response

scene_generator = SceneWriter()

# response = scene_generator(scene_gist="a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before")

# logger.info(response.get("generated_scene", "response contains no keyword 'generated_scene'"))

training_set = [
    # dspy.Example({"scene_gist": "a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before"}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "An unexpected turn of events happen and raises the stakes"}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "A romantic confession that feels awkward and painful."}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "A comedic misunderstanding in a shared flat. High subtext."}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "An unexpected turn of events happen and raises the stakes"}).with_inputs("scene_gist"),
]

def scene_metric(example, pred, trace=None):
    logger.info("the start of metric function - args of scene-metric func: \nexample: {}\npred: {}\ntrace: {}\n".format(example, pred, trace))

    # use the llm-as-judge pattern
    judge = dspy.ChainOfThought(dspy.make_signature("generated_scene -> score: float", instructions="score should be between 0.0 and 1.0. 0.0 means that the quality of generated scene is terrible. While 1.0 means that the quality of generated scene is perfect."))
    judgement = judge(generated_scene=pred.get("generated_scene", "n/a"))
    # logger.info("judge's response: \n{}\n".format(judgement))
    score = judgement.get("score")
    logger.info("judge's score: {}\n".format(score))

    return score

# optimizer = dspy.BootstrapFewShot(scene_metric)
optimizer = dspy.MIPROv2(scene_metric, max_bootstrapped_demos=3, max_labeled_demos=2, num_candidates=1, auto=None)

# before optimizing, let's just evaluate it
# from dspy.evaluate import evaluate
# evaluator = evaluate.Evaluate(devset=training_set, metric=scene_metric)
# evaluation_result = evaluator(scene_generator)
# logger.info("evaluation_result: {}".format(evaluation_result))
# logger.info("evaluation score: {}".format(evaluation_result.get("score", "unknown")))
# logger.info("getting the results from the evaluation")
# for current_evaluation_res in evaluation_result.get("results", []):
#     logger.info("current_evaluation_res: {}".format(current_evaluation_res))


logger.info("starting the optimization")
optimized_scene_generator = optimizer.compile(scene_generator, trainset=training_set, num_trials=1, minibatch=False)
logger.info("optimized model")

optimized_response = optimized_scene_generator(scene_gist="a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before")
logger.info(optimized_response.get("generated_scene", "optimized response contains no keyword 'generated_scene'"))


logger.info("saving the optimized model")
optimized_scene_generator.save("./optimized_scene_generator.json")

logger.info("completing...")
