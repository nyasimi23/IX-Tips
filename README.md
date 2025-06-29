# ⚽ IX-Tips – Football Match Prediction System

**IX-Tips** is an intelligent football match prediction system built with **Django**, **Celery**, and **scikit-learn**. It fetches real-time fixtures, trains ML models using historical data, generates predictions, and displays daily tips with accuracy evaluation.

---

## 📌 Features

- 🔮 Predict match outcomes using machine learning (RandomForest).
- ⏰ Schedule automatic weekly predictions via **Celery**.
- 🧠 Store and compare actual results vs predictions.
- 🟢 Highlight correct, 🔴 incorrect, 🔵 upcoming predictions.
- ⚙️ Admin dashboard with live prediction tools and metadata monitoring.
- 📊 League table integration and tip filtering (e.g., Over 2.5, GG, 1X2).
- 📅 Match calendar filtering with AJAX UI.

---

## 🛠️ Technologies Used

- **Backend:** Django, Celery, Redis, PostgreSQL/SQLite
- **Machine Learning:** scikit-learn (RandomForest, LabelEncoder)
- **Frontend:** Bootstrap, jQuery, AJAX
- **Data Source:** [football-data.org](https://football-data.org)

---

## 🚀 Setup Instructions

### 1. Clone the project


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nyasimi23/IX-Tips
   cd IX-Tips
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Football Data API:
   - Obtain an API key from [Football Data](https://www.football-data.org/).
   - Replace `API_KEY` in the code with your API key.

4. Run database migrations:
   ```bash
   python manage.py migrate
   ```

5. Start the development server:
   ```bash
   python manage.py runserver
   ```

6. Access the application in your browser:
   ```
   http://127.0.0.1:8000/
   ```

---

## Usage

1. **Home Page**:
   - Use the date picker to select a specific date.
   - Click "Get Predictions" to fetch match fixtures, predictions, and results for that date.

2. **Predictions and Results**:
   - View predicted results and goal counts for each match.
   - Check actual scores and results for completed matches.
   - Match status is color-coded:
     - **Green**: Correct prediction.
     - **Red**: Incorrect prediction.
     - **Grey**: Match not yet played.

---

## Supported Competitions
- Premier League (PL)
- La Liga (PD)
- Serie A (SA)
- Bundesliga (BL1)
- Ligue 1 (FL1)
- Eredivisie (DED)
- Primeira Liga (PPL)
- Championship (ELC)
- UEFA Champions League (CL)
- FIFA World Cup (WC)

---

## File Structure

```
matchday-predictions/
├── manage.py
├── predict/
│   ├── models.py
│   ├── views.py
│   ├── templates/
│   │   └── predict/
│   │       └── matchday_predictions_by_date.html
├── static/
├── requirements.txt
└── README.md
```

---

## Key Functions

### `fetch_competition_matches(api_key, competition_code, season)`
Fetches match data for a given competition and season.

### `get_actual_results(competition_id, matchday)`
Retrieves actual results for a specific competition and matchday.

### `train_models(df)`
Trains machine learning models for predicting match results and goal counts.

### `predict_match_outcome(home_team, away_team, classifier_model, regressor_models, label_encoder_X)`
Predicts the match result and goal outcomes for a specific fixture.

### `safe_encode(team_name, label_encoder)`
Encodes team names while handling unseen teams.

---

## API Reference
- **Football Data API**: [Documentation](https://www.football-data.org/documentation/quickstart)

---

## To-Do

1. Improve prediction accuracy by adding more features to the dataset.
2. Add support for additional competitions.
3. Enable user authentication for personalized predictions.
4. Implement a caching mechanism to reduce API calls.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
- [Football Data API](https://www.football-data.org/) for providing match data.
- Open-source libraries and frameworks: Django, scikit-learn, pandas.

 
