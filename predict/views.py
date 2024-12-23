import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render
from datetime import datetime
from .models import MatchPrediction

API_KEY = '7419be10abd14d7fb752e6fe6491e38f'
BASE_URL = "https://api.football-data.org/v4"


# Fetch competition matches
def fetch_competition_matches(api_key, competition_code, season):
    url = f"{BASE_URL}/competitions/{competition_code}/matches?season={season}"
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        matches = data['matches']
        match_data = []
        for match in matches:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            full_time_score = match['score']['fullTime']
            score = f"{full_time_score['home']} - {full_time_score['away']}"
            match_data.append([home_team, away_team, score])
        return pd.DataFrame(match_data, columns=['Home Team', 'Away Team', 'Score'])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching competition matches: {e}")
        return None


# Fetch matches by date
def fetch_matches_by_date(api_key, competition_code, match_date):
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": api_key}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        matches = response.json().get("matches", [])
        filtered_matches = [
            match for match in matches if match.get("utcDate", "").startswith(match_date)
        ]
        return filtered_matches
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches by date: {e}")
        return []

# Preprocess match data
def preprocess_api_data(api_df):
    api_df = api_df.dropna(subset=["Score"])
    api_df[['HomeGoals', 'AwayGoals']] = api_df['Score'].str.split(' - ', expand=True)
    api_df['HomeGoals'] = pd.to_numeric(api_df['HomeGoals'], errors='coerce').fillna(0)
    api_df['AwayGoals'] = pd.to_numeric(api_df['AwayGoals'], errors='coerce').fillna(0)
    api_df['FTR'] = api_df.apply(
        lambda row: 'Home' if row['HomeGoals'] > row['AwayGoals']
        else ('Away' if row['HomeGoals'] < row['AwayGoals'] else 'Draw'),
        axis=1
    )
    return api_df[['Home Team', 'Away Team', 'HomeGoals', 'AwayGoals', 'FTR']]


# Train models
def train_models(df):
    X = df[['Home Team', 'Away Team']]
    y_regression = df[['HomeGoals', 'AwayGoals']]

    label_encoder_X = LabelEncoder()
    label_encoder_X.fit(pd.concat([X['Home Team'], X['Away Team']]).unique())

    X['Home Team'] = X['Home Team'].apply(lambda team: safe_encode(team, label_encoder_X))
    X['Away Team'] = X['Away Team'].apply(lambda team: safe_encode(team, label_encoder_X))

    X_train, X_test, y_train_regression, y_test_regression = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    regressor_models = {}
    for column in y_regression.columns:
        regressor_model = DecisionTreeRegressor(random_state=42)
        regressor_model.fit(X_train, y_train_regression[column])
        regressor_models[column] = regressor_model

    return regressor_models, label_encoder_X


# Predict match outcome
def predict_match_outcome(home_team, away_team, regressor_models, label_encoder_X):
    home_team_encoded = safe_encode(home_team, label_encoder_X)
    away_team_encoded = safe_encode(away_team, label_encoder_X)

    if home_team_encoded == -1 or away_team_encoded == -1:
        return "Error", 0, 0

    match_data = [[home_team_encoded, away_team_encoded]]

    predicted_home_goals = round(regressor_models['HomeGoals'].predict(match_data)[0])
    predicted_away_goals = round(regressor_models['AwayGoals'].predict(match_data)[0])

    if predicted_home_goals > predicted_away_goals:
        predicted_result = "Home"
    elif predicted_home_goals < predicted_away_goals:
        predicted_result = "Away"
    else:
        predicted_result = "Draw"

    return predicted_result, predicted_home_goals, predicted_away_goals


# Helper for encoding team names
def safe_encode(team_name, label_encoder):
    try:
        return label_encoder.transform([team_name])[0]
    except ValueError:
        return -1

def get_actual_results(api_key, competition_code, match_date):
    """
    Fetches actual results of matches for a specific competition and date.

    Parameters:
    - api_key: API key for the football-data.org API.
    - competition_code: The competition code (e.g., 'PL' for Premier League).
    - match_date: The date of the matches in 'YYYY-MM-DD' format.

    Returns:
    - A list of dictionaries containing match results, including teams, scores, and status.
    """
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": api_key}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        matches = response.json().get("matches", [])
        actual_results = []

        # Filter matches by date and ensure they are finished
        for match in matches:
            if match.get("utcDate", "").startswith(match_date) and match["status"] == "FINISHED":
                home_team = match["homeTeam"]["name"]
                away_team = match["awayTeam"]["name"]
                full_time_score = match["score"]["fullTime"]
                actual_home_goals = full_time_score["home"]
                actual_away_goals = full_time_score["away"]
                actual_result = (
                    "Home" if actual_home_goals > actual_away_goals
                    else "Away" if actual_home_goals < actual_away_goals
                    else "Draw"
                )
                actual_results.append({
                    "home_team": home_team,
                    "away_team": away_team,
                    "actual_home_goals": actual_home_goals,
                    "actual_away_goals": actual_away_goals,
                    "actual_result": actual_result,
                })

        return actual_results

    except requests.exceptions.RequestException as e:
        print(f"Error fetching actual results: {e}")
        return []


# Main view
from django.shortcuts import render
from datetime import datetime
import pandas as pd

def matchday_predictions(request):
    competitions = {
        "PL": "Premier League",
        "PD": "La Liga",
        "SA": "Serie A",
        "BL1": "Bundesliga",
        "FL1": "Ligue 1",
        "DED": "Eredivisie",
        "PPL": "Primeira Liga",
        "ELC": "Championship",
        "CL": "UEFA Champions League",
        "EC": "European Championship",
        "BSA": "Campeonato Brasileiro Série A",
        "CLI": "Copa Libertadores",
        "WC": "FIFA World Cup",
    }

    predictions = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    default_competition = "PL"

    if request.method == "GET":
        competition = request.GET.get("competition")
        match_date = request.GET.get("date", current_date)

        competition_codes = [competition] if competition else list(competitions.keys())

        for comp_code in competition_codes:
            # Fetch all matches for the date
            matches = fetch_matches_by_date(API_KEY, comp_code, match_date)
            if not matches:
                continue

            # Get actual results for the matches
            actual_results = get_actual_results(API_KEY, comp_code, match_date)

            for match in matches:
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                match_status = match['status']

                # Check for existing prediction in the database
                prediction = MatchPrediction.objects.filter(
                    competition=competitions[comp_code],
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                ).first()

                if not prediction:
                    # Generate prediction if not found
                    predicted_result, home_goals, away_goals = predict_match_outcome(
                        home_team, away_team, regressor_models, label_encoder_X
                    )
                    predicted_score = f"{home_goals} - {away_goals}"

                    # Calculate additional fields
                    gg = "Yes" if home_goals >= 1 and away_goals >= 1 else "No"
                    average_goals_category = (
                        "Over 1.5" if home_goals + away_goals >= 2 else "Under 1.5"
                    )

                    # Save the prediction to the database
                    prediction = MatchPrediction.objects.create(
                        competition=competitions[comp_code],
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        predicted_result=predicted_result,
                        predicted_score=predicted_score,
                        status=match_status,
                    )
                else:
                    # Use existing prediction
                    predicted_score = prediction.predicted_score
                    if " - " in predicted_score:
                        try:
                            home_goals, away_goals = map(int, predicted_score.split(" - "))
                        except ValueError:
                            home_goals, away_goals = 0, 0

                        gg = "Yes" if home_goals >= 1 and away_goals >= 1 else "No"
                        average_goals_category = (
                            "Over 1.5" if home_goals + away_goals >= 2 else "Under 1.5"
                        )
                    else:
                        home_goals, away_goals, gg, average_goals_category = 0, 0, "No", "Low"

                # Get actual match results
                actual_match = next(
                    (
                        r for r in actual_results
                        if r["home_team"] == home_team and r["away_team"] == away_team
                    ),
                    None,
                )

                if actual_match:
                    actual_home_goals = actual_match["actual_home_goals"]
                    actual_away_goals = actual_match["actual_away_goals"]
                    actual_result = actual_match["actual_result"]
                    actual_score = f"{actual_home_goals} - {actual_away_goals}"

                    agg = "Yes" if actual_home_goals >= 1 and actual_away_goals >= 1 else "No"
                    ov = (
                        "Over 1.5"
                        if actual_home_goals + actual_away_goals >= 2
                        else "Under 1.5"
                    )
                else:
                    actual_result, actual_score, agg, ov = None, "--", "No", "Under 1.5"

                # Add match prediction to the list
                predictions.append({
                    "competition": competitions[comp_code],
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_result": prediction.predicted_result,
                    "actual_result": actual_result,
                    "predicted_score": predicted_score,
                    "actual_score": actual_score,
                    "gg": gg,
                    "agg": agg,
                    "ov": ov,
                    "average_goals_category": average_goals_category,
                    "status": match_status,
                })

    # Calculate accuracy
    finished_games = [p for p in predictions if p["status"] == "FINISHED"]
    total_finished = len(finished_games)

    correct_results = sum(1 for p in finished_games if p["predicted_result"] == p["actual_result"])
    correct_scores = sum(1 for p in finished_games if p["predicted_score"] == p["actual_score"])
    correct_ov = sum(1 for p in finished_games if p["average_goals_category"] == p["ov"])
    correct_gg = sum(1 for p in finished_games if p["gg"] == p["agg"])

    accuracy = {
        "result_accuracy": round((correct_results / total_finished) * 100, 2)
        if total_finished
        else 0,
        "score_accuracy": round((correct_scores / total_finished) * 100, 2)
        if total_finished
        else 0,
        "ov_accuracy": round((correct_ov / total_finished) * 100, 2)
        if total_finished
        else 0,
        "gg_accuracy": round((correct_gg / total_finished) * 100, 2)
        if total_finished
        else 0,
        "overall_accuracy": round(
            (
                (correct_results + correct_scores + correct_ov + correct_gg)
                / (4 * total_finished)
            )
            * 100,
            2,
        )
        if total_finished
        else 0,
    }

    return render(
        request,
        "predict/matchday_predictions.html",
        {
            "competitions": competitions,
            "predictions": predictions,
            "current_date": current_date,
            "accuracy": accuracy,
        },
    )
