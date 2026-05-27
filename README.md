# World Cup Prediction Project ⚽🏆

![WIP](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)
![CI](https://github.com/Yccne/Worlcup_Prediction/actions/workflows/ci.yml/badge.svg)

A full-stack Machine Learning application to predict and simulate FIFA World Cup match outcomes. 
Built with a FastAPI backend serving an XGBoost model, and a React frontend for bracket simulations.

## 🔗 Live Demo
*Placeholder: [Link to live demo](https://example.com)*

## 🚀 Project Status

- [x] **Phase 1 — Data collection & feature engineering** (pandas, Elo ratings, lag features, no leakage)
- [ ] **Phase 2 — XGBoost model** (multi-class, TimeSeriesSplit, calibrated probabilities)
- [ ] **Phase 3 — FastAPI backend** (POST /predict, POST /simulate, CORS, Dockerfile)
- [ ] **Phase 4 — React UI** (bracket view, match predictor, simulation runner, team comparison)

## 🛠️ Tech Stack

**Data Science & Machine Learning:**
* Python 3.11, pandas, numpy
* XGBoost, scikit-learn

**Backend API:**
* FastAPI, Uvicorn, Pydantic
* Docker

**Frontend:**
* React, Vite, TypeScript
* TailwindCSS

## 🏗️ Architecture

*(Placeholder for architectural diagram)*

## 🧠 How the ML Works (Plain English)

Instead of relying on basic FIFA rankings, this model calculates and learns from **Elo Ratings**, which mathematically adjust based on the strength of opponents a team beats. 
The model also looks at **Recent Form** (rolling 10-game win/loss records) and **Goal Differences** to understand momentum. 

**Preventing Data Leakage:**
A critical piece of our feature engineering pipeline ensures that when predicting a specific match, the model *only* uses data that was available 1 day *before* that match. Rolling statistics are computed using chronological state tracking (`df.shift(1)` logic) to prevent the model from "cheating" by seeing future outcomes.

## 📊 Model Accuracy

*(Placeholder for test set metrics — expected after Phase 2)*
* Test Accuracy (2022 World Cup): TBD
* Log Loss: TBD

## 🏃‍♂️ How to Run Locally

### Prerequisites
* Python 3.11+
* Node.js 18+ (for frontend)
* Docker (optional)

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 💡 What I'd Improve Next
*(Placeholder for post-project reflections)*
