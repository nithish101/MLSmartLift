# SmartLift

SmartLift is a workout tracking app that uses machine learning to help you figure out what weight to lift next. I built it because I got tired of guessing whether I should go up in weight or stay where I am — so I trained a model to answer that for me.

## What it does

You log your workouts (exercise, weight, reps, how many reps you had left in the tank), and SmartLift uses a Gradient Boosting model to:

- **Predict your next session** — what weight to use, how many reps to expect, and what intensity to target
- **Track your estimated 1RM** over time so you can see if you're actually getting stronger
- **Score your weekly training volume** (0–100) to flag if you're undertraining or overdoing it
- **Compare predictions vs reality** so you can see how well the model is actually working
- **Estimate recovery time** after each session based on intensity and muscle group

Right now it supports bench press, squat, and deadlift. The model was trained on synthetic progression data that follows realistic strength curves.

## The ML side

The model is a `GradientBoostingRegressor` from scikit-learn. It takes in:

| Feature | Importance |
|---------|-----------|
| Weight lifted | 35.5% |
| Bodyweight | 31.3% |
| Progression trend | 13.0% |
| Exercise type | 9.9% |
| Volume score | 6.2% |
| RIR | 1.8% |
| Sets | 1.3% |
| Frequency | 1.0% |

It predicts how many reps you'll get at a given weight. From there, the recommendation engine decides whether to increase weight (if predicted reps > 9), decrease (if < 5), or stay put. Target range is 5–9 reps at RIR 2, which is a solid hypertrophy zone.

Current performance: **MAE of 1.12 reps** on the test set. Not perfect, but good enough to be useful — and it gets better as it sees more of your data.

1RM is estimated with the Epley formula (adjusted for RIR):

```
e1RM = weight × (1 + (reps + RIR) / 30)
```

## Tech stack

- **Backend**: FastAPI + SQLite + scikit-learn
- **Frontend**: React (Vite) with Recharts for visualizations
- **Styling**: Custom CSS, dark theme, mobile-first
- **Deployment**: PWA-ready (can be added to iPhone home screen)

## Running it locally

### Backend

```bash
cd backend
pip install -r requirements.txt

# Generate training data and train the model
python -m ml.generate_data
python -m ml.train_model

# Seed with demo data (optional, gives you something to look at right away)
python seed_demo.py

# Start the API
python main.py
# → http://localhost:8000 (docs at /docs)
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### iPhone access

Start the frontend with `npm run dev -- --host`, then open your local IP in Safari and add to home screen. It'll run as a standalone app without the browser chrome.

## Project structure

```
backend/
  main.py           — FastAPI endpoints
  database.py       — SQLite setup
  models.py         — request/response schemas
  seed_demo.py      — populates DB with demo workouts
  ml/
    generate_data.py — synthetic training data
    train_model.py   — model training
    predictor.py     — inference, 1RM, volume, recovery

frontend/
  src/
    App.jsx          — tab navigation shell
    api.js           — API client
    components/
      WorkoutLogger.jsx   — log exercises and sets
      Dashboard.jsx       — charts and progress tracking
      Recommendation.jsx  — AI predictions and alerts
      ModelExplainer.jsx  — how the model works (for demo)
```

## What I'd do next

- Expand the exercise library beyond the big 3
- Train on real anonymized data instead of synthetic
- Add adaptive retraining so the model improves per user
- Build out a full training program generator
- Injury risk estimation based on volume/intensity patterns
