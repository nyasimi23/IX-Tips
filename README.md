# IX-Tips

## Overview
The IX-Tips is a Django-based web application that provides predictions for football matches across various competitions. It fetches data from the Football Data API and uses machine learning models to predict match results and goal outcomes. The application also displays actual scores and results for matches that have already been played.

---

## Features

1. **Fetch Match Data**: Retrieves match fixtures for all supported competitions on a selected date.
2. **Predictions**:
   - Predicted match result (Home Win, Away Win, Draw).
   - Predicted goals for both home and away teams.
   - "GG" (Both Teams to Score) and Over/Under goal categories.
3. **Actual Results**:
   - Displays the actual result and score if the match has already been played.
   - Highlights whether the prediction matches the actual result.
4. **Dynamic Date Handling**:
   - Allows users to select a specific date.
   - Defaults to the current date if no date is selected.
5. **Support for Multiple Competitions**:
   - Includes competitions like Premier League, La Liga, Serie A, Bundesliga, UEFA Champions League, and more.

---

## Prerequisites

1. Python 3.7+
2. Django 3.2+
3. Required Python Libraries:
   - `pandas`
   - `scikit-learn`
   - `requests`

---

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

 
