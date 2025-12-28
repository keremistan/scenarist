# 🎬 Scenarist: Eval-Driven Narrative Engine
*a.k.a. Ghostwriter*
### An autonomous agentic workflow for consistent, high-fidelity narrative generation using Local LLMs.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-red)
![Architecture](https://img.shields.io/badge/Architecture-Chain--of--Thought-orange)

---

## 🚀 The Problem
Standard LLMs suffer from **Context Drift** and **Hallucination** when generating long-form content. They often forget established facts, ignore stylistic guidelines, or produce generic, "lazy" output. "Prompt and pray" strategies are insufficient for production-grade creative applications.

## 🛠 The Solution: Deterministic Chains > Agents
Scenarist replaces the black-box generation approach with a **Structured Logic Chain**. Instead of generating a scene in one shot, the system decomposes the creative process into discrete, verifiable steps.

**The Pipeline:**
1.  **Keyword Extraction:** Converts user intent into dramatic search queries.
2.  **RAG (Retrieval):** Fetches semantic references from a curated vector store of screenplay datasets.
3.  **Logical Planning:** Generates a structural blueprint (Beats, Characters, Location) *before* writing dialogue.
4.  **Style Planning:** Analyzes reference scenes to enforce specific pacing and vocabulary rules.
5.  **Drafting:** Generates the scene adhering strictly to the Logical and Style plans.
6.  **Auto-Critique (The Guardrail):** A separate evaluator model scores the draft (1.0 - 5.0) on coherence and style adherence.

## ✨ Key Features

* **Structured Outputs:** Uses **Pydantic** to enforce strict JSON schemas, preventing the "markdown bleed" common in LLM outputs.
* **Local LLM Optimization:** Tuned to run on `gpt-oss:20b` (Ollama), reducing dependency on expensive proprietary models like GPT-4 while maintaining structural integrity.
* **Automated Evaluation:** Every generation is scored against a rubric. This allows for **Eval-Driven Development**—we measure improvements via benchmarks, not "vibes."
* **Decoupled Architecture:**
* **Frontend:** Streamlit (UI/Visualization)
* **Backend:** FastAPI (Stateless logic)
* **Core:** Python/LangChain (Orchestration)


## ⚡️ Quick Start

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) running locally (for local inference)

### Installation

```bash
# 1. Clone the repo
git clone [https://github.com/keremistan/scenarist.git](https://github.com/keremistan/scenarist.git)
cd scenarist

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Running the System

**1. Start the API Server:**

```bash
uvicorn showrunner.api.main:app --reload
```

**2. Start the UI:**

```bash
streamlit run ui/app.py
```

## 🔮 Roadmap

* [x] Deterministic Generation Chain
* [x] Automated Critique Loop
* [ ] **DSPy Implementation:** Replacing manual prompt templates with automated prompt optimization.
* [ ] **Knowledge Graph:** Integrating Neo4j to track character relationships across multiple scenes.
