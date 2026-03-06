# 🏥 Medical Symptom Checker

> **⚠️ EDUCATIONAL PURPOSE ONLY** — Not medical advice. Always consult a licensed healthcare professional. Call 911 in emergencies.

A **production-quality**, modular **Medical Symptom Checker** built with **LangGraph** + **Gemini 2.5 Flash**. Implements 17 graph nodes with conditional pathways for emergency, moderate, and mild symptom presentations.

---

## ✨ Features

| Feature                | Details                                                |
| ---------------------- | ------------------------------------------------------ |
| **LLM**                | Gemini 2.5 Flash (via `langchain-google-genai`)        |
| **Graph**              | LangGraph StateGraph — 17 nodes, 5 conditional routers |
| **Severity Triage**    | Hybrid: ESI rule-based + LLM classification            |
| **Red Flag Detection** | Dual-layer: keyword scan + LLM analysis                |
| **Safety**             | Allergy-aware, medication-interaction checking         |
| **Mental Health**      | Built-in screening + crisis resource referral          |
| **Hospital Search**    | Tavily-powered real-time facility lookup               |
| **Error Handling**     | Every node has graceful fallback — zero crashes        |
| **CLI**                | Interactive + demo mode with 5 pre-built scenarios     |

---

## 📁 Project Structure

```
medical_symptom_checker/
├── core/
│   ├── config.py       # Settings, LLM init (Gemini), env loading
│   ├── logger.py       # Centralized logging (console + file)
│   └── exceptions.py   # Custom exception hierarchy
├── models/
│   ├── patient.py      # PatientProfile, MedicalHistory
│   ├── symptom.py      # Symptom, VitalSigns
│   ├── assessment.py   # SeverityAssessment, Recommendation
│   ├── mental_health.py
│   └── state.py        # MedicalCheckerState TypedDict + factory
├── nodes/
│   ├── utils.py        # Shared utilities, LLM retry, data loaders
│   ├── greeting.py     # Node 1: Disclaimer display
│   ├── patient_profile.py  # Node 2: Demographics extraction
│   ├── symptom_intake.py   # Node 3: Multi-pass symptom parsing
│   ├── follow_up.py    # Node 4: Clarifying questions
│   ├── red_flag.py     # Node 5: Dual-layer emergency detection
│   ├── medical_history.py  # Node 6: Background collection
│   ├── severity.py     # Node 7: Core ESI + LLM classification
│   ├── emergency.py    # Nodes 8a & 8b: Emergency guidance + hospital finder
│   ├── moderate.py     # Nodes 9a & 9b: Doctor rec + specialist mapping
│   ├── mild.py         # Nodes 10a & 10b: Home remedies + lifestyle
│   └── shared.py       # Nodes 11-14: Drug check, mental health, report, feedback
├── routing/
│   └── edges.py        # All 5 conditional routing functions
├── data/
│   ├── red_flags.json     # Red flag keyword categories
│   ├── specialist_map.json # Symptom → specialist mapping
│   └── disclaimer.txt     # Medical disclaimer text
├── tests/
│   ├── test_models.py     # 40+ pure unit tests (no API)
│   ├── test_graph.py      # Graph compilation tests
│   └── test_routing.py    # Routing logic tests
├── logs/                  # Runtime logs (auto-created)
├── graph.py               # ⭐ LangGraph assembly
├── main.py                # ⭐ CLI entry point
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## 🚀 Setup

### 1. Prerequisites

```powershell
# Activate the virtual environment
.\myvirtualenvironment\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
cd medical_symptom_checker
pip install -r requirements.txt
```

### 3. Configure API Keys

```powershell
# Create .env (one directory up from this package — in e:\AgenticAI)
copy medical_symptom_checker\.env.example .env
# Then edit .env and add your keys:
```

```env
GOOGLE_API_KEY=AIza...your-gemini-api-key...
TAVILY_API_KEY=tvly-...your-tavily-key...  # Optional
```

Get your Gemini API key: [aistudio.google.com](https://aistudio.google.com)

### 4. Run

```powershell
# Interactive mode (from e:\AgenticAI)
python -m medical_symptom_checker.main

# Demo mode (all scenarios menu)
python -m medical_symptom_checker.main --demo

# Specific demo scenario
python -m medical_symptom_checker.main --demo 2
```

---

## 🎭 Demo Scenarios

| #   | Scenario                                | Severity     |
| --- | --------------------------------------- | ------------ |
| 1   | Mild cold                               | 🟢 Mild      |
| 2   | Chest pain + arm radiating              | 🚨 Emergency |
| 3   | Persistent fever with diabetes          | 🟡 Moderate  |
| 4   | Stress + daily headaches                | 🟡 Moderate  |
| 5   | Elderly multi-condition (heart failure) | 🔴 Severe    |

---

## 🧪 Running Tests

```powershell
# All tests (no API calls required)
pytest tests/ -v

# Coverage report
pytest tests/ --cov=medical_symptom_checker --cov-report=html

# Model tests only (fastest)
pytest tests/test_models.py -v
```

---

## 🔄 Graph Flow

```
START → greeting → patient_profile → symptom_intake
  ├── [needs more info] → follow_up → symptom_intake (loop up to 5x)
  └── [info sufficient] → red_flag_check
        ├── [RED FLAG] → emergency_guidance → [has location?] → hospital_finder
        └── [safe] → medical_history → severity_classifier
              ├── [emergency/severe] → emergency_guidance → hospital_finder
              ├── [moderate] → doctor_recommendation → specialist_mapper → home_remedy
              └── [mild] → home_remedy → lifestyle_prevention
                          (all paths) → drug_interaction_check → mental_health_screen
                                       → summary_report → feedback_followup → END
```

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It does **not** provide medical advice, diagnosis, or treatment recommendations. Always consult a licensed, qualified healthcare professional for any medical concerns.

**In an emergency: Call 911 (US) or your local emergency number immediately.**

Suicide & Crisis Lifeline: Call or Text **988** | Crisis Text Line: Text HOME to **741741**
