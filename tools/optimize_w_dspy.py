import dspy
import mlflow
from tools.logging_template import setup_logging

logger = setup_logging("dspy_optimizer")

mlflow.dspy.autolog()

lm = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434", api_key="")

dspy.configure(lm=lm)

scene_signature = dspy.make_signature("scene_gist -> full_scene") 
# as of now (31.12.25), there are three steps in generation.write_scene, but this may find a single step prompt that reduces the three steps to one. maybe. let's see.

scene_generator = dspy.ChainOfThought(scene_signature)

# response = scene_generator(scene_gist="a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before")

# logger.info(response.get("full_scene", "response contains no keyword 'full_scene'"))

training_set = [
    # dspy.Example({"scene_gist": "a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before"}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "An unexpected turn of events happen and raises the stakes"}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "A romantic confession that feels awkward and painful."}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "A comedic misunderstanding in a shared flat. High subtext."}).with_inputs("scene_gist"),
    dspy.Example({"scene_gist": "An unexpected turn of events happen and raises the stakes"}).with_inputs("scene_gist"),
]

def scene_metric(example, pred, trace=None):
    #todo
    logger.info("example: {}\npred: {}\ntrace: {}\n".format(example, pred, trace))
    return 1

optimizer = dspy.BootstrapFewShot(scene_metric)

# before optimizing, let's just evaluate it
from dspy.evaluate import evaluate
evaluator = evaluate.Evaluate(devset=training_set, metric=scene_metric)
evaluation_result = evaluator(scene_generator)
logger.info("evaluation_result: {}".format(evaluation_result))

# optimized_scene_generator = optimizer.compile(scene_generator, trainset=training_set)
#
# optimized_response = optimized_scene_generator(scene_gist="a man whose life changes for the better but the new circumstances bring their own challenges that he hasn't faced before")
# logger.info(optimized_response.get("full_scene", "optimized response contains no keyword 'full_scene'"))
#
# optimized_scene_generator.save("./optimized_scene_generator")
#
