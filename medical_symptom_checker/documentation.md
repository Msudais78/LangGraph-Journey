# Medical Symptom Checker - Complete Technical Documentation

> **⚠️ EDUCATIONAL PURPOSE ONLY** — Not medical advice. Always consult a licensed healthcare professional. Call 911 in emergencies.

---

## 1. Project Overview

The **Medical Symptom Checker** is an intelligent, multi-turn, stateful application designed to assess reported medical symptoms, classify their severity (triage), and provide educational guidance.

It is built on a robust architecture using **LangGraph** (for state machine control flow) and **Gemini 2.5 Flash** (via LangChain's `ChatGoogleGenerativeAI`) for reasoning, natural language understanding, and structured data extraction. By employing a graph-based workflow, the application ensures deterministic routing, fault tolerance, and comprehensive logic handling (e.g., loops for missing info, fast-paths for red flags).

---

## 2. Directory Structure & Architecture

```
e:\AgenticAI\medical_symptom_checker\
├── core/               # Core Application Infrastructure
│   ├── config.py       # Configuration management (Pydantic Settings)
│   ├── logger.py       # Centralized, file + console logging
│   └── exceptions.py   # Custom exception hierarchy
├── models/             # Data Validation & Schemas (Pydantic v2)
│   ├── patient.py      # PatientProfile and MedicalHistory models
│   ├── symptom.py      # Symptom and VitalSigns models
│   ├── assessment.py   # SeverityAssessment and Recommendation models
│   ├── mental_health.py# MentalHealthScreen model
│   └── state.py        # LangGraph State definitions (MedicalCheckerState)
├── nodes/              # Graph Nodes (The actual logic steps)
│   ├── utils.py        # Shared LLM retry logic, data loaders
│   ├── greeting.py     # Static disclaimer
│   ├── patient_profile.py # LLM Demographics extraction
│   ├── symptom_intake.py  # LLM Multi-symptom parser
│   ├── follow_up.py    # LLM clarifying questions
│   ├── red_flag.py     # Rule-based + LLM critical flag detector
│   ├── medical_history.py # Background health data collection
│   ├── severity.py     # Hybrid ESI / LLM classification
│   ├── emergency.py    # Emergency advice + Tavily hospital search
│   ├── moderate.py     # Doctor visit planner + Specialist mapper
│   ├── mild.py         # Safe home remedies + lifestyle advice
│   └── shared.py       # Drug interactions, Mental health screening, Report Gen.
├── routing/            # Graph Edges (Conditional flow logic)
│   └── edges.py        # Functions dictating next steps based on State
├── data/               # Static Data Source Files
│   ├── red_flags.json  # Keywords for emergency detection
│   ├── specialist_map.json # Symptom -> Doctor Type mappings
│   └── disclaimer.txt  # Hardcoded legal disclaimer
├── tests/              # Pytest Unit Test Suite (No API keys needed)
│   ├── test_models.py
│   ├── test_graph.py
│   └── test_routing.py
├── graph.py            # LangGraph Assembly File (StateGraph definition)
├── main.py             # CLI Entry Point (Interactive & Demo)
└── README.md
```

### Core Architecture principles

1. **Separation of Concerns:** Nodes (action logic) and Edges (routing logic) are entirely separated. Data structures (Models) are separated from processing logic.
2. **Graceful Fallbacks:** Every node that calls an external LLM or API has a hardcoded static fallback. The system will _never_ crash due to a network timeout or context window failure.
3. **Hybrid AI + Rule-based:** Critical decisions (e.g., Red Flags, calculating Triage Scores) never rely solely on LLMs. They combine deterministic rules (e.g., keyword searches, Emergency Severity Index logic) with LLM natural language reasoning.

---

## 3. Data Models (`models/`)

We heavily rely on **Pydantic v2** to strongly type and validate data throughout the application.

- `patient.py`:
  - **`PatientProfile`**: Normalizes demographic data (e.g. "M", "male", "MALE" all become "male") and calculates boolean risk tags (`is_elderly`, `is_pediatric`, `is_high_risk_age`).
  - **`MedicalHistory`**: Normalizes lists of conditions/medications. Includes computed properties using substring matching (e.g. matching "warfarin 5mg" to `is_on_blood_thinners`) and checking against high-risk comorbidities.
- `symptom.py`:
  - **`Symptom`**: Extracts `name`, `intensity` (1-10 slider bounds checking), `duration`, `onset`, and `body_location`. Has computed risks like `is_severe` (intensity >= 8) and `is_acute`.
  - **`VitalSigns`**: Extracts numbers safely, computes rule-based flags (`has_fever` > 100.4F, `has_low_oxygen` < 94%).
- `assessment.py`:
  - **`SeverityAssessment`**: The absolute core output of the engine. Tracks `level` ("emergency", "severe", "moderate", "mild"), `confidence_score`, rule-based `triage_score`, and extracted clinical reasoning.
  - **`Recommendation`**: Tracks actionable next steps.
- `mental_health.py`:
  - **`MentalHealthScreen`**: Pure boolean flags for anxiety/depression indicators.
- `state.py`:
  - **`MedicalCheckerState`**: The master LangGraph state. Inherits from Pythons `TypedDict`. Allows `total=False` so nodes can return only partial state updates (e.g., a node returning just `{"red_flag_triggered": True}`).

---

## 4. Graph Flow & Routing (`routing/edges.py` & `graph.py`)

LangGraph works by updating a shared state object through a series of "Nodes", and then determining the next Node using "Conditional Edges".

The graph follows this precise logic flow:

1.  **START**
2.  **`greeting`**: Immediate legal disclaimer printout.
3.  **`patient_profile`**: Extracts Name, Age, Gender, Location from the first chat input.
4.  **`symptom_intake`**: The AI attempts to extract explicit Symptom objects.
5.  _Conditional_: `should_continue_followup()` checks if the AI needs more info.
    - If **YES**: routes to **`follow_up`** mode, asks the user a question, and loops back to `symptom_intake`. Hardcapped at 5 loops to prevent infinite chats.
    - If **NO**: routes to **`red_flag_check`**.
6.  **`red_flag_check`**: Checks the parsed data against `red_flags.json` and a secondary LLM rule-check.
7.  _Conditional_: `route_after_red_flag()`
    - If **RED_FLAG**: routes to **`emergency_guidance`** (Fast-Pathing the user to safety).
    - If **SAFE**: routes to **`medical_history`** to collect background info safely.
8.  **`severity_classifier`**: Looks at patient age, history, and symptoms to apply an ESI-like score (1-5) and categorizes as Emergency / Severe / Moderate / Mild.
9.  _Conditional_: `route_by_severity()` branches the path:
    - `emergency` / `severe` -> **`emergency_guidance`** -> `route_after_emergency()` (checks for location data) -> **`hospital_finder`** (Tavily search).
    - `moderate` -> **`doctor_recommendation`** -> **`specialist_mapper`**
    - `mild` -> **`home_remedy`** -> **`lifestyle_prevention`**
10. All non-emergency branches eventually converge.
11. **Convergence Layer** (`shared.py`):
    - **`drug_interaction_check`**: Looks at history vs current recommendations to find contraindications (e.g., NSAIDs vs blood thinners).
    - **`mental_health_screen`**: Looks at the raw chat history for key words (suicide, self-harm). If present, immediately provides 988 lifeline. Otherwise does a gentle wellness check.
    - **`summary_report`**: Pure string formatting. Builds the final UI/ASCII output block.
    - **`feedback_followup`**: Schedules the follow-up times (e.g., "See a doctor in 1-3 days if no improvement").
12. **END**

---

## 5. detailed Node Breakdown (`nodes/`)

### Node 1: `greeting_disclaimer_node`

Simply formats the disclaimer text located in `data/disclaimer.txt`. Static node. No LLM used.

### Node 2: `patient_profile_node`

Uses Gemini's `with_structured_output` capability directly bound to the `PatientProfile` Pydantic class. Prompts the LLM to extract demographic data from natural language. Fallback: Returns an empty generic profile.

### Node 3: `symptom_intake_node`

A highly complex "dual-pass" node. First, it extracts a list of `Symptom` and `VitalSign` objects. Second, it uses conversational logic to determine if the `needs_more_info` flag should be set to True (e.g., if a user said "My head hurts" but didn't provide intensity, duration, or onset).

### Node 4: `follow_up_question_node`

Only hit if `needs_more_info=True`. Uses the LLM to craft a targeted, single follow-up question. Increments `loop_count`. Fallback asks a generic "Can you provide more details?"

### Node 5: `red_flag_detector_node`

Dual-layer detection:

1.  **Rule-Based:** Checks user text explicitly against `data/red_flags.json` (categories like "chest_pain", "stroke_signs").
2.  **LLM-Based:** Prompts the LLM to look for subtle emergencies the rule-set missed.
    If either triggers, `red_flag_triggered` is set to `True` in the LangGraph State.

### Node 6: `medical_history_node`

Uses structured extraction bound to `MedicalHistory` to parse out comorbidities, medications, and allergies.

### Node 7: `severity_classifier_node` ⭐ Core Engine

Calculates the **Emergency Severity Index (ESI) Score**.

- Level 1: Immediate life-saving intervention needed
- Level 2: High risk / red flags detected
- Level 3: Multiple resources needed (moderate)
- Level 4: One resource needed (mild)
- Level 5: No resources needed
  Uses the LLM to provide clinical reasoning, but uses programmatic rule-checking (e.g. `patient.is_high_risk_age`, `symptom.is_severe`) to override the LLM if it estimates too low. Fails-safe to "moderate/severe" if the LLM crashes.

### Nodes 8, 9, 10

The distinct pathway nodes (Emergency, Moderate, Mild). They each construct specific `Recommendation` objects.

- The `emergency` path uses the **Tavily API** (`nearest_hospital_node`) to search the live web for "Nearest emergency room near [Patient Location]".
- The `moderate` path maps the symptom to a doctor using `data/specialist_map.json`.
- The `mild` path strictly enforces safety checks by reading allergies and medications before suggesting any OTC home remedies.

### Nodes 11, 12, 13, 14 (`shared.py`)

See Graph Flow section above (Steps 11). `summary_report` constructs a beautiful, formatted console output report.

---

## 6. Testing Strategy (`tests/`)

The application embraces **Pure Logic Testing**.
Running `python -m pytest` executes 55 unit tests entirely offline without consuming any LLM API credits.

- **Models:** Validates bounds checking (intensity 1-10), list de-duplication, gender normalization, computed property triggers (e.g. `is_on_blood_thinners`).
- **Routing:** Parametrized tests that fake State objects and verify `route_by_severity` and `should_continue_followup` properly return the correct string names for the next nodes.
- **Graph:** Compiles the LangGraph to ensure no hanging nodes or misnamed edges exist.

---

## 7. How to Extend / Modify

**1. Adding new questions/intake flows:**
Modify the loop condition in `routing/edges.py -> should_continue_followup()`. Adjust `max_loops`.

**2. Adding a new LLM task in the graph:**

- Create a node function in `nodes/` that accepts `(state: MedicalCheckerState) -> dict`.
- Update `graph.py` to `workflow.add_node("your_node", your_func)`
- Update `graph.py` to wire `workflow.add_edge("previous_node", "your_node")`.

**3. Adjusting Severity Tolerance:**
Edit `nodes/severity.py`. To make the bot _more_ conservative, adjust the hardcoded severity bumps (e.g. uncommenting or adding extra flags to bump ESI scores from 3 to 2).

**4. Changing LLM Models:**
Edit `.env` and `core/config.py`. By default it uses `gemini-2.5-flash`. You can change the environment variable to `gemini-pro` easily. LangChain's wrapper handles the rest.
