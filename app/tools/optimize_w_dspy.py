import dspy
import mlflow
import datetime

from app.api.scene_writer import SceneWriter
from app.tools.logging_template import setup_logging

mlflow.dspy.autolog()

logger = setup_logging("dspy_optimizer")

lm = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434", api_key="")

dspy.configure(lm=lm)

scene_generator = SceneWriter()

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

optimizer = dspy.MIPROv2(scene_metric, max_bootstrapped_demos=3, max_labeled_demos=2, num_candidates=1, auto=None)

logger.info("starting the optimization")
optimized_scene_generator = optimizer.compile(scene_generator, trainset=training_set, num_trials=1, minibatch=False)
logger.info("optimized model")

# generate a sample scene to see what it can do
optimized_response = optimized_scene_generator(scene_gist="a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before")
logger.info(optimized_response.get("generated_scene", "optimized response contains no keyword 'generated_scene'"))

logger.info("saving the optimized model")
optimized_scene_generator.save("./data/{}-optimized_scene_generator.json".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

logger.info("completing...")