# Advanced Sequential LangGraph Workflow 🔄

This project demonstrates a sophisticated **Sequential Workflow** using LangGraph, featuring **nested conditional feedback loops** for self-correction. It emulates a professional, multi-stage editorial team.

## Overview

Unlike a simple linear sequence (A -> B -> C), this workflow introduces intermediate evaluation nodes that can send the process _backwards_ if quality standards aren't met.

### The Pipeline

1. **Outline Generation (`generate_outline`)**: A creative LLM drafts an outline based on a topic.
2. **Outline Review (`review_outline`)**: A strict editorial LLM evaluates the outline.
   - If **Rejected**, it routes back to `generate_outline` with specific feedback.
   - If **Approved**, it proceeds to drafting.
3. **Blog Generation (`generate_blog`)**: The creative LLM writes the full article following the approved outline.
4. **Final Editorial Review (`evaluate_blog`)**: The Editor-in-Chief evaluates the complete draft.
   - If **Needs Revision**, it routes back to `generate_blog` with feedback.
   - If **Ready**, it publishes the final article.

## Features

- **Multi-Agent Simulation**: Uses two different LLM configurations (a high-temperature creative writer and a low-temperature strict editor).
- **Conditional Edges**: Dynamic routing based on the outcome of previous nodes.
- **Cyclic Graphs**: The graph can loop back on itself to iterate on a single piece of content.
- **State Limits**: Enforces a maximum number of revisions (e.g., max 3 loops) to prevent infinite loops in edge cases.

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Msudais78/LangGraph-Journey.git
   cd LangGraph-Journey/Advanced_Sequential_Workflow
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API Key:**
   Create a `.env` file in the directory (or set it in your environment variables):

   ```env
   GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
   ```

4. **Run the Notebook:**
   Open `advanced_sequential_workflow.ipynb` in Jupyter Notebook, VS Code, or Google Colab and execute the cells. The final output will be saved automatically to a text file in your directory.
