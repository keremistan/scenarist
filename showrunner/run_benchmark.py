import csv
import time
from writer import write_scene
from eval import evaluate
from statistics import mean

# 1. Define the parameters for THIS run (Change these manually before running)
EXPERIMENT_NAME = "Initial Baseline"
PARAMS = {
    "model_name": "gpt-oss:20b",
    "reasoning": "medium",
    "retriever_k": 5,
    "database_size": "metropolis ve silicon valley 0402",  # Just a label for your own reference
    "prompt_version": "You are an expert screenwriter.\nUse reference scenes. Their storytelling elements are important; not the specific actions, locations or characters, but how they deliver the emotion. \n\nScenes: ..." # this should be a single line
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
    
    print(f"--- STARTING EXPERIMENT: {EXPERIMENT_NAME} ---")
    
    for prompt in GOLDEN_PROMPTS:
        # Generate & Judge
        draft, model_response = write_scene(prompt, return_model_response=True) # Ensure write_scene uses your global PARAMS if possible
        score_card = evaluate(model_response, prompt)
        if score_card:
            scores.append(score_card.style_adherence)
        else:
            print("an error happened with getting the score card. It is null.")

    avg_score = mean(scores)
    
    # 2. LOGGING: Save to CSV
    log_entry = [timestamp, EXPERIMENT_NAME, avg_score] + list(PARAMS.values())
    
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

    print(f"\n✅ Experiment Saved! Score: {avg_score:.2f}")

if __name__ == "__main__":
    run_suite()
