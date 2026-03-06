import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""\
# Iterative Post Generator for X (Twitter) using LangGraph
This notebook implements an iterative workflow that takes a topic name as the only user input 
and produces a high-quality X (Twitter) post through an automated loop of **generation**, 
**evaluation**, and **optimization**.

## Architecture Overview
- **Generator LLM**: Creates the initial X post based on the given topic.
- **Evaluator LLM**: Scores and critiques the generated post against 7 quality criteria (each scored 1-10), returning a numeric average and detailed per-criterion feedback.
- **Optimizer LLM**: Takes a failed post and evaluator feedback to produce an improved version.

**Pass threshold**: Average score ≥ 9/10 (programmatically enforced — the LLM does NOT decide pass/fail).  
**Max iterations**: 4.
"""))

    # Setup cell
    nb.cells.append(nbf.v4.new_markdown_cell("## Setup & Imports\nEnsure you have `.env` file with `GOOGLE_API_KEY` set."))
    nb.cells.append(nbf.v4.new_code_cell("""\
import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# ── Configurable Constants ──────────────────────────────────
PASS_THRESHOLD = 9        # Average score must be >= this to PASS
MAX_ITERATIONS = 4        # Maximum optimization loops
# ────────────────────────────────────────────────────────────

# Generator LLM: Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Evaluator LLM: Groq - llama-3.3-70b-versatile (strong reasoning for strict evaluation)
evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Optimizer LLM: Groq - llama-3.1-8b-instant (fast, efficient text rewriting)
optimizer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

from IPython.display import Image, display
"""))

    # State Definition
    nb.cells.append(nbf.v4.new_markdown_cell("## Graph State Definition\nWe define the state that is passed between nodes in our LangGraph workflow."))
    nb.cells.append(nbf.v4.new_code_cell("""\
class PostState(TypedDict):
    topic: str
    current_post: str
    score: int               # Computed average — NOT decided by the LLM
    feedback: str
    verdict: str             # "PASS" or "FAIL" — computed programmatically
    iteration_count: int
    status_message: str
    iteration_log: list[str]
"""))

    # Generator Node
    nb.cells.append(nbf.v4.new_markdown_cell("""\
## 1. Generator Node
Creates the **initial draft** of the X post.  

> **FIX NOTE:** The generator prompt is kept intentionally *simple* so it produces a 
> decent-but-not-perfect first draft. This gives the optimizer room to work. 
> Previously the generator prompt was so detailed that the first draft already 
> satisfied every criterion.
"""))
    nb.cells.append(nbf.v4.new_code_cell("""\
def generator_node(state: PostState) -> PostState:
    topic = state["topic"]
    
    # ── FIX: Simplified generator prompt ──────────────────────
    # The old prompt gave too many instructions (hashtags, engagement,
    # no spam, etc.), which meant the first draft was already near-perfect
    # and the evaluator had nothing to critique.
    # Now we give a SHORT prompt so the draft is a rough starting point.
    # ──────────────────────────────────────────────────────────
    sys_msg = SystemMessage(
        content="You are a social media copywriter. Write short X (Twitter) posts."
    )
    prompt = f"Write a quick X (Twitter) post about: '{topic}'. Keep it under 280 characters."
    response = llm.invoke([sys_msg, HumanMessage(content=prompt)])
    
    return {
        "current_post": response.content.strip(),
        "iteration_count": 1,
        "iteration_log": []
    }
"""))

    # Evaluator Node
    nb.cells.append(nbf.v4.new_markdown_cell("""\
## 2. Evaluator Node

### What was wrong before (and how it's fixed)

| Problem | Fix |
|---|---|
| The 7 old criteria were **trivially easy** (e.g. "under 280 chars" — every LLM does this). The evaluator always said PASS. | Replaced with **7 high-bar criteria** that demand virality, emotional resonance, originality, etc. |
| The LLM decided both the **score AND the verdict**. It could give 7/10 and still say "PASS". | The LLM now returns **per-criterion scores (1-10)**. PASS/FAIL is **computed programmatically** from the average. |
| No minimum threshold — "PASS" was entirely the LLM's subjective call. | Added a **configurable `PASS_THRESHOLD`** (default: 9). Average must be ≥ 9 to pass. |
| System prompt said "strict" but nothing enforced it. | Added explicit instructions: *"A score of 10 means absolutely perfect. 7 means decent but flawed. Be harsh."* |
"""))
    nb.cells.append(nbf.v4.new_code_cell("""\
# ── FIX: Per-criterion structured output ──────────────────
# Instead of a single score + verdict, we force the LLM to score
# each criterion individually (1-10). This prevents the LLM from
# hand-waving a high overall score.
# ──────────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    criterion: str = Field(description="Name of the criterion being scored")
    score: int = Field(description="Score from 1 to 10 for this criterion")
    reason: str = Field(description="Brief justification for this score and specific improvement suggestion")

class EvaluatorOutput(BaseModel):
    criteria_scores: list[CriterionScore] = Field(
        description="A list of exactly 7 scored criteria"
    )
    overall_feedback: str = Field(
        description="A 2-3 sentence summary of the most critical improvements needed"
    )

# ── FIX: Much stricter system message ─────────────────────
evaluator_sys_msg = \"\"\"You are an elite-level X (Twitter) content strategist and critic.
You are EXTREMELY hard to impress. Your standards are those of posts that go genuinely viral (10K+ engagements).

Scoring guidelines:
- 10 = Absolutely flawless, would go viral as-is. Reserve this for truly exceptional work.
- 8-9 = Very strong but has minor room for improvement.
- 6-7 = Decent but clearly missing something. Most first drafts land here.
- 4-5 = Mediocre, needs significant rework.
- 1-3 = Poor, fails the criterion.

You MUST find specific, actionable flaws. Do NOT give a perfect score unless the post is genuinely exceptional.
A generic, safe, corporate-sounding post should NEVER score above 7 on Engagement or Originality.\"\"\"

# ── FIX: Higher-bar criteria ──────────────────────────────
# Old criteria like "under 280 chars" and "has 1-3 hashtags" were
# trivially satisfied. New criteria demand actual QUALITY.
# ──────────────────────────────────────────────────────────
evaluator_prompt = \"\"\"Evaluate this X (Twitter) post about the topic "{topic}" on these 7 criteria.
Score each criterion from 1-10 and provide specific feedback.

CRITERIA:
1. **Hook Power**: Does the first line immediately grab attention? Would someone stop scrolling? (A bland opening = max 6)
2. **Emotional Resonance**: Does it evoke curiosity, excitement, surprise, humor, or urgency? (Neutral/flat tone = max 6)
3. **Originality**: Is the take fresh and unique, or is it a generic/cliché statement anyone could write? (Generic = max 5)
4. **Engagement Potential**: Does it invite replies, retweets, or debate? Is there a question, hot take, or CTA? (No interaction driver = max 6)
5. **Clarity & Conciseness**: Is every word earning its place? No filler, no jargon, crystal clear message?
6. **Platform Fit**: Proper length (≤280 chars), appropriate hashtag usage (1-3, relevant), reads well on X's format?
7. **Topic Authority**: Does the post demonstrate genuine insight about "{topic}", not just surface-level observation?

POST TO EVALUATE:
\\\"{post}\\\"

Score each criterion individually. Be specific about what's weak and how to fix it.\"\"\"

def evaluator_node(state: PostState) -> PostState:
    post = state["current_post"]
    topic = state["topic"]
    iteration = state["iteration_count"]
    
    structured_evaluator = evaluator_llm.with_structured_output(EvaluatorOutput)
    prompt = evaluator_prompt.format(topic=topic, post=post)
    
    sys_msg = SystemMessage(content=evaluator_sys_msg)
    result: EvaluatorOutput = structured_evaluator.invoke(
        [sys_msg, HumanMessage(content=prompt)]
    )
    
    # ── FIX: Programmatic score & verdict ─────────────────
    # The LLM no longer decides PASS/FAIL. We compute the average
    # of all per-criterion scores and compare to PASS_THRESHOLD.
    # This removes the LLM's tendency to be overly generous.
    # ──────────────────────────────────────────────────────
    scores = [c.score for c in result.criteria_scores]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    
    # Programmatic verdict — NOT from the LLM
    verdict = "PASS" if avg_score >= PASS_THRESHOLD else "FAIL"
    
    # Build detailed feedback string from per-criterion results
    detailed_feedback_lines = []
    for c in result.criteria_scores:
        marker = "✅" if c.score >= PASS_THRESHOLD else "❌"
        detailed_feedback_lines.append(
            f"  {marker} {c.criterion}: {c.score}/10 — {c.reason}"
        )
    detailed_feedback = "\\n".join(detailed_feedback_lines)
    
    full_feedback = f"Per-criterion breakdown:\\n{detailed_feedback}\\n\\nOverall: {result.overall_feedback}"
    
    # Log the iteration
    log_entry = (
        f"── Iteration {iteration} ──\\n"
        f"Post: [{post}]\\n"
        f"Scores: {', '.join(f'{c.criterion}={c.score}' for c in result.criteria_scores)}\\n"
        f"Average: {avg_score}/10 (threshold: {PASS_THRESHOLD}) | Verdict: {verdict}\\n"
        f"Feedback: {result.overall_feedback}"
    )
    new_log = state.get("iteration_log", []) + [log_entry]
    
    return {
        "verdict": verdict,
        "score": avg_score,
        "feedback": full_feedback,
        "iteration_log": new_log
    }
"""))

    # Optimizer Node
    nb.cells.append(nbf.v4.new_markdown_cell("""\
## 3. Optimizer Node

### What was wrong before (and how it's fixed)

| Problem | Fix |
|---|---|
| Optimizer received vague single-string feedback | Now receives **per-criterion scores and specific suggestions** |
| No instruction to focus on the *weakest* areas | Explicit instruction: "Focus on the criteria that scored lowest" |
"""))
    nb.cells.append(nbf.v4.new_code_cell("""\
optimizer_sys_msg = \"\"\"You are an elite X (Twitter) copywriter. You rewrite posts to be more 
viral, engaging, and impactful. You take evaluator feedback seriously and make bold, creative changes — 
not just minor word swaps.\"\"\"

# ── FIX: Optimizer now sees per-criterion scores ──────────
optimizer_prompt = \"\"\"Rewrite this X (Twitter) post to address ALL evaluator feedback.

Topic: {topic}

Current Post (scored {score}/10 — needs ≥ {threshold} to pass):
\\\"{post}\\\"

Evaluator Feedback:
{feedback}

INSTRUCTIONS:
- Focus your rewrite on the criteria that scored LOWEST.
- Make BOLD changes — don't just rearrange words. Rethink the angle, hook, or structure if needed.
- The post MUST be ≤ 280 characters.
- Include 1-3 relevant hashtags.
- Make it feel human, punchy, and scroll-stopping.

Output ONLY the rewritten post text, nothing else.\"\"\"

def optimizer_node(state: PostState) -> PostState:
    topic = state["topic"]
    post = state["current_post"]
    feedback = state["feedback"]
    score = state["score"]
    iteration = state["iteration_count"]
    
    prompt = optimizer_prompt.format(
        topic=topic,
        post=post,
        feedback=feedback,
        score=score,
        threshold=PASS_THRESHOLD
    )
    sys_msg = SystemMessage(content=optimizer_sys_msg)
    response = optimizer_llm.invoke([sys_msg, HumanMessage(content=prompt)])
    
    return {
        "current_post": response.content.strip(),
        "iteration_count": iteration + 1
    }
"""))

    # Workflow Definition
    nb.cells.append(nbf.v4.new_markdown_cell("""\
## 4. LangGraph Workflow
Define the graph, add nodes, and set the conditional edges.

### What was wrong before (and how it's fixed)

| Problem | Fix |
|---|---|
| Router function tried to mutate `state` directly (which doesn't work in LangGraph — nodes return new state, they don't mutate) | Router is now a pure function that only returns a routing string |
| `status_message` was set inside the router but never actually persisted | `status_message` is computed at display time from final state |
"""))
    nb.cells.append(nbf.v4.new_code_cell("""\
def router(state: PostState) -> str:
    verdict = state.get("verdict", "FAIL")
    iteration = state.get("iteration_count", 1)
    
    # ── FIX: Router is now a pure routing function ────────
    # It does NOT try to mutate state (which didn't work before).
    # It only returns the next node name or END.
    # ──────────────────────────────────────────────────────
    if verdict == "PASS":
        return END
    elif iteration >= MAX_ITERATIONS:
        return END
    else:
        return "optimizer"

# Build Graph
builder = StateGraph(PostState)

builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("optimizer", optimizer_node)

builder.set_entry_point("generator")

builder.add_edge("generator", "evaluator")
builder.add_edge("optimizer", "evaluator")

# Conditional edge from Evaluator
builder.add_conditional_edges("evaluator", router)

graph = builder.compile()
"""))

    # Graph Visualization Cell
    nb.cells.append(nbf.v4.new_markdown_cell("## Graph Visualization\nVisualize the execution layout of the LangGraph workflow."))
    nb.cells.append(nbf.v4.new_code_cell("""\
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    # This might fail if the required dependencies are missing or if graphviz is unavaialable
    print("Could not display graph:", e)
"""))

    # Execution Cell
    nb.cells.append(nbf.v4.new_markdown_cell("## Execution & Output Formatting\nProvide a topic and run the complete loop."))
    nb.cells.append(nbf.v4.new_code_cell("""\
def generate_x_post(topic: str):
    
    print(f"🚀 Starting workflow for topic: '{topic}'")
    print(f"   Pass threshold: {PASS_THRESHOLD}/10 | Max iterations: {MAX_ITERATIONS}\\n")
    
    initial_state = {"topic": topic}
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    # Determine status
    iters = final_state['iteration_count']
    verdict = final_state['verdict']
    score = final_state['score']
    
    if verdict == "PASS":
        status = f"✅ PASSED at iteration {iters} (score: {score}/10)"
    else:
        status = f"⚠️ MAX ITERATIONS REACHED — best effort (score: {score}/10)"
    
    # Print results
    print("=" * 50)
    print(f"📌 Topic: {final_state['topic']}")
    print(f"📝 Final Post: {final_state['current_post']}")
    print(f"📊 Status: {status}")
    print(f"🔄 Total Iterations: {iters} / {MAX_ITERATIONS}")
    print(f"⭐ Final Score: {score} / 10")
    print("\\n--- Iteration Log ---")
    
    for log in final_state['iteration_log']:
        print(log)
        print("-" * 40)
        
    return final_state

# Example Run
final_output = generate_x_post("The future of Agentic AI in software development")
"""))

    # Save to file
    notebook_content = nbf.writes(nb)
    with open("x_post_generator.ipynb", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("Notebook 'x_post_generator.ipynb' created successfully!")

if __name__ == "__main__":
    create_notebook()