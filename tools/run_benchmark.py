import csv
import time
from statistics import mean
from tools.logging_template import setup_logging
from api.models import SceneRequest
from api.generation import write_scene

logger = setup_logging("run_benchmark")

# 1. Define the parameters for THIS run (Change these manually before running)
EXPERIMENT_NAME = "changed the eval's prompt to make it clearer that 1 is bad and 5 is good. Adjusted ScoreCard field names and descriptions as well to make it clear that high scores mean a better result"
PARAMS = {
    "model_name": "gpt-oss:20b",
    "reasoning": "medium",
    "retriever_k": 5,
    "database_size": "491 entries from multiple genre",  # Just a label for your own reference
    "prompt_version": "You are a Ghostwriter. You must MIMIC the style of the reference scenes that you will fetch.\\n\\nCRITICAL PROCESS:\\nFetch the reference scenes\\nYou are FORBIDDEN from writing the scene immediately.\\nYou must first output a \"LOGICAL PLAN\" and then a \"STYLE PLAN\" where you analyze the reference scenes.\\nYou have to write the scene while following the both plans.\\n\\nFORMAT:            --- LOGICAL PLAN ---\\n1. Story arc: what the actual story is\\n2. Characters: who the characters are? what are their relations to each other? How are they moving the story forward?\\n3. Location: where the story takes place? why is it actually this place? how is this place relevant for the story?\\n--- STYLE PLAN ---\\n1. Pacing Analysis: (e.g. \"Fast, short sentences\" or \"Slow, monologues\")\\n2. Subtext Strategy: (How the characters hide their true feelings)\\n3. Vocabulary Rules: (Specific words or grammar to use/avoid)\\n------------------\\n--- SCENE START ---\\n[Write the scene here, strictly following the plans above]\\n\\nSTORY GUIDELINE:..." # this should be a single line
}

GOLDEN_PROMPTS = {
    "An unexpected turn of events happen and raises the stakes",
    "A romantic confession that feels awkward and painful.",
    "A comedic misunderstanding in a shared flat. High subtext."
    # "A quiet scene of betrayal in a diner. High subtext.",
    # "A tense interrogation where the cop is corrupt.",
}

def run_suite():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    scores = []
    
    logger.info(f"--- STARTING EXPERIMENT: {EXPERIMENT_NAME} ---")
    
    for prompt in GOLDEN_PROMPTS:
        # Generate & Judge
        scene_response = write_scene(scene_request=SceneRequest(user_prompt=prompt))
        scores.extend([scene_response.critique_score])

    avg_score = mean(scores)
    logger.info("avg_score: {}\n".format(avg_score))
    
    # 2. LOGGING: Save to CSV
    log_entry = [timestamp, EXPERIMENT_NAME, avg_score] + list(PARAMS.values())
    logger.info("log_entry: {}\n".format(log_entry))
    
    # Write header if file doesn't exist
    file_exists = False
    try:
        with open('experiment_history.csv', 'r') as f: file_exists = True
    except FileNotFoundError: pass

    with open('experiment_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header: Time, Name, Score, [Param Names...]
            writer.writerow(["Timestamp", "Exp Name", "Avg Score"] + list(PARAMS.keys()))
        
        writer.writerow(log_entry)

    logger.info(f"\n✅ Experiment Saved! Score: {avg_score:.2f}")

if __name__ == "__main__":
    run_suite()
