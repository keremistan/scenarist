# 🎬 Scenarist: Eval-Driven Narrative Engine
a.k.a. Ghostwriter
### An autonomous agentic platform for consistent, high-fidelity narrative generation.
**Built with DSPy, FastAPI, and Docker.**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![DSPy](https://img.shields.io/badge/AI-DSPy_Optimized-purple)
![Docker](https://img.shields.io/badge/Deploy-Docker_Compose-blue)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-orange)

---

## 🚀 The Problem
Standard LLMs suffer from **Context Drift** and **Hallucination** when generating long-form content. They often forget established facts, ignore stylistic guidelines, or produce generic, "lazy" output.
Manual prompt engineering is brittle; changing one word in a prompt can break the entire pipeline.

## 🛠 The Solution: Compiled Logic > Prompt Engineering
Scenarist replaces fragile string manipulation with **DSPy (Declarative Self-Improving Python)**.
Instead of manually tweaking prompts, we define **Signatures** (Inputs/Outputs) and **Modules** (Logic). We then use an **Optimizer** (MIPROv2/BootstrapFewShot) to "compile" the best possible prompts by learning from a feedback metric.

**The Pipeline:**
1.  **Keyword Extraction:** Converts user intent into dramatic search queries.
2.  **RAG (Retrieval):** Fetches semantic references from a curated vector store.
3.  **Chain-of-Thought:** The agent plans the scene structure (Beats, Characters, Subtext) *before* generation.
4.  **Auto-Critique:** A separate "Judge" model evaluates the draft against a rubric.
5.  **Optimization Loop:** The system learns from its best outputs, automatically injecting successful few-shot examples into future prompts.

## 🏗 Architecture

```mermaid
graph TD
    subgraph Docker Cluster
        UI[Frontend Service<br/>Streamlit] --> API[Backend Service<br/>FastAPI]
        API --> Engine[DSPy Engine]
        
        subgraph "AI Core"
            Engine --> Optimizer[Compiler<br/>MIPROv2]
            Optimizer -->|Trains| CompiledProgram[Optimized JSON]
            CompiledProgram -->|Inference| LLM
        end
        
        Engine --> DB[(Vector Store)]
    end
    
    User --> UI
