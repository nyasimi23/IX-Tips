import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render
from datetime import datetime

API_KEY = '7419be10abd14d7fb752e6fe6491e38f'
BASE_URL = "https://api.football-data.org/v4"


def fetch_competition_matches(api_key, competition_code, season):
    url = f"{BASE_URL}/competitions/{competition_code}/matches?season={season}"
    headers = {"X-Auth-Token": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
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
    else:
        return None

# Fetch matches by date
def fetch_matches_by_date(api_key, competition_code, match_date):
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        matches = response.json().get("matches", [])
        # Filter matches by the provided date
        filtered_matches = [
            match for match in matches 
            if match.get("utcDate", "").startswith(match_date)
        ]
        return filtered_matches

    except requests.exceptions.RequestException as e:
        # Log the error and return an empty list if the API call fails
        print(f"Error fetching matches: {e}")
        return []

def get_actual_results(competition_id, match_date):
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"match_date": match_date}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        matches = response.json().get("matches", [])
        results = []
        for match in matches:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            if match['status'] == 'FINISHED':
                full_time_score = match['score']['fullTime']
                actual_result = (
                    'Home' if full_time_score['home'] > full_time_score['away'] else
                    'Away' if full_time_score['home'] < full_time_score['away'] else
                    'Draw'
                )
                results.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_result': actual_result,
                    'actual_home_goals': full_time_score.get('home'),
                    'actual_away_goals': full_time_score.get('away'),
                    'status': 'FINISHED'
                })
        return results
    else:
        print(f"Error fetching results: {response.status_code} - {response.text}")
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

# Train models for predictions
def train_models(df):
    X = df[['Home Team', 'Away Team']]
    y_regression = df[['HomeGoals', 'AwayGoals']]

    # Encode team names
    label_encoder_X = LabelEncoder()
    X['Home Team'] = label_encoder_X.fit_transform(X['Home Team'])
    X['Away Team'] = label_encoder_X.transform(X['Away Team'])

    # Train regressors
    X_train, X_test, y_train_regression, y_test_regression = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    regressor_models = {}
    for column in y_regression.columns:
        regressor_model = DecisionTreeRegressor(random_state=42)
        regressor_model.fit(X_train, y_train_regression[column])
        regressor_models[column] = regressor_model

    return regressor_models, label_encoder_X

# Unified prediction for match result and goals
def predict_match_outcome(home_team, away_team, regressor_models, label_encoder_X):
    home_team_encoded = safe_encode(home_team, label_encoder_X)
    away_team_encoded = safe_encode(away_team, label_encoder_X)

    match_data = [[home_team_encoded, away_team_encoded]]

    # Predict goals
    predicted_home_goals = round(regressor_models['HomeGoals'].predict(match_data)[0])
    predicted_away_goals = round(regressor_models['AwayGoals'].predict(match_data)[0])

    # Determine match result based on predicted goals
    if predicted_home_goals > predicted_away_goals:
        predicted_result = "Home"
    elif predicted_home_goals < predicted_away_goals:
        predicted_result = "Away"
    else:
        predicted_result = "Draw"

    return predicted_result, predicted_home_goals, predicted_away_goals

# Helper to handle unseen teams
def safe_encode(team_name, label_encoder):
    try:
        return label_encoder.transform([team_name])[0]
    except ValueError:
        return -1

# Main view
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

    # Default to the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    if request.method == "GET":
        # Retrieve competition and date from request
        competition = request.GET.get("competition")
        match_date = request.GET.get("date", current_date)  # Default to current_date if not provided

        try:
            # Determine competitions to process: all competitions if none is specified
            competition_codes = [competition] if competition else list(competitions.keys())
            
            for comp_code in competition_codes:
                matches = fetch_matches_by_date(API_KEY, comp_code, match_date)

                if matches:
                    all_seasons_data = []
                    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
                    actual_results = get_actual_results(comp_code, match_date)

                    # Process matches for predictions
                    for season in seasons:
                        season_data = fetch_competition_matches(API_KEY, comp_code, season)
                        if season_data is not None:
                            all_seasons_data.append(season_data)

                    if all_seasons_data:
                        all_seasons_df = pd.concat(all_seasons_data, ignore_index=True)
                        processed_data = preprocess_api_data(all_seasons_df)

                        # Train models
                        regressor_models, label_encoder_X = train_models(processed_data)

                    for match in matches:
                        home_team = match['homeTeam']['name']
                        away_team = match['awayTeam']['name']
                        match_status = match['status']

                        actual_match = next(
                            (r for r in actual_results if r['home_team'] == home_team and r['away_team'] == away_team),
                            None
                        )

                        actual_result = actual_match['actual_result'] if actual_match else None
                        actual_home_goals = actual_match['actual_home_goals'] if actual_match else None
                        actual_away_goals = actual_match['actual_away_goals'] if actual_match else None
                        actual_score = f"{actual_home_goals} - {actual_away_goals}" if actual_match else "--"

                        predicted_result, home_goals, away_goals = predict_match_outcome(
                            home_team, away_team, regressor_models, label_encoder_X
                        )
                        predicted_score = f"{home_goals} - {away_goals}"
                        total_goals = home_goals + away_goals
                        actual_total = actual_home_goals + actual_away_goals if actual_match else None

                        agg = "Yes" if actual_home_goals and actual_away_goals and actual_home_goals >= 1 and actual_away_goals >= 1 else "No"
                        gg = "Yes" if home_goals >= 1 and away_goals >= 1 else "No"
                        average_goals_category = "Over 1.5" if total_goals >= 2 else "Under 1.5"
                        ov = "Over 1.5" if actual_total and actual_total >= 2 else "Under 1.5"

                        predictions.append({
                            'competition': competitions[comp_code],  # Include competition name
                            'home_team': home_team,
                            'away_team': away_team,
                            'predicted_result': predicted_result,
                            'actual_result': actual_result,
                            'predicted_score': predicted_score,
                            'actual_score': actual_score,
                            'gg': gg,
                            'agg': agg,
                            'ov': ov,
                            'average_goals_category': average_goals_category,
                            'status': match_status,
                        })
        except Exception as e:
            print(f"Error in matchday_predictions: {e}")
            predictions = []

    return render(
        request,
        'predict/matchday_predictions.html',
        {'competitions': competitions, 'predictions': predictions, 'current_date': current_date}
    )
