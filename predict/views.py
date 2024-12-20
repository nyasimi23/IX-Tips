import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render

API_KEY = '7419be10abd14d7fb752e6fe6491e38f'
BASE_URL = "https://api.football-data.org/v4"

# Fetch matches for a given competition and season
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

def get_matchday_fixtures(competition_id, matchday):
    """
    Fetch fixtures for a specific competition and matchday.
    """
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    headers = {
        "X-Auth-Token": API_KEY
    }
    params = {"matchday": matchday}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data['matches']  # Return the matches for the given matchday
    else:
        print(f"Error fetching matchday fixtures: {response.status_code} - {response.text}")
        return None

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
    # Prepare features and labels
    X = df[['Home Team', 'Away Team']]
    y_classification = df['FTR']
    y_regression = df[['HomeGoals', 'AwayGoals']]

    # Encode team names
    label_encoder_X = LabelEncoder()
    X['Home Team'] = label_encoder_X.fit_transform(X['Home Team'])
    X['Away Team'] = label_encoder_X.transform(X['Away Team'])

    # Encode classification labels
    label_encoder_y_classification = LabelEncoder()
    y_classification_encoded = label_encoder_y_classification.fit_transform(y_classification)

    # Train classifier
    X_train, X_test, y_train_classification, y_test_classification = train_test_split(
        X, y_classification_encoded, test_size=0.2, random_state=42
    )
    classifier_model = DecisionTreeClassifier(random_state=42)
    classifier_model.fit(X_train, y_train_classification)

    # Train regressors
    X_train_reg, X_test_reg, y_train_regression, y_test_regression = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    regressor_models = {}
    for column in y_regression.columns:
        regressor_model = DecisionTreeRegressor(random_state=42)
        regressor_model.fit(X_train_reg, y_train_regression[column])
        regressor_models[column] = regressor_model

    return classifier_model, label_encoder_y_classification, regressor_models, label_encoder_X

# Predict match results and goals
def predict_match_result(home_team, away_team, classifier_model, label_encoder_X, label_encoder_y_classification):
    home_team_encoded = safe_encode(home_team, label_encoder_X)
    away_team_encoded = safe_encode(away_team, label_encoder_X)
    match_data = [[home_team_encoded, away_team_encoded]]
    prediction = classifier_model.predict(match_data)
    return label_encoder_y_classification.inverse_transform(prediction)[0]

def predict_goals(home_team, away_team, regressor_models, label_encoder_X):
    home_team_encoded = safe_encode(home_team, label_encoder_X)
    away_team_encoded = safe_encode(away_team, label_encoder_X)
    match_data = [[home_team_encoded, away_team_encoded]]
    predictions = {column: round(model.predict(match_data)[0]) for column, model in regressor_models.items()}
    return predictions

# Helper to handle unseen teams
def safe_encode(team_name, label_encoder):
    try:
        return label_encoder.transform([team_name])[0]
    except ValueError:
        return -1

# Main view
def matchday_predictions(request):
    matchdays = range(1, 39)  # Generate matchdays from 1 to 38
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
    
    if request.method == "POST":
        competition = request.POST.get("competition")
        matchday = request.POST.get("matchday")
        
        if competition and matchday:
            try:
                matchday = int(matchday)
                matches = get_matchday_fixtures(competition, matchday)

                if matches:
                    all_seasons_data = []
                    seasons = [2019, 2020, 2021, 2022, 2023]
                    
                    for season in seasons:
                        season_data = fetch_competition_matches(API_KEY, competition, season)  # Use selected competition
                        if season_data is not None:
                            all_seasons_data.append(season_data)

                    if all_seasons_data:
                        all_seasons_df = pd.concat(all_seasons_data, ignore_index=True)
                        processed_data = preprocess_api_data(all_seasons_df)

                        # Train models
                        classifier_model, label_encoder_y_classification, regressor_models, label_encoder_X = train_models(processed_data)

                        # Make predictions
                        for match in matches:
                            home_team = match['homeTeam']['name']
                            away_team = match['awayTeam']['name']

                            predicted_result = predict_match_result(
                                home_team, away_team, classifier_model, label_encoder_X, label_encoder_y_classification
                            )
                            predicted_goals = predict_goals(
                                home_team, away_team, regressor_models, label_encoder_X
                            )

                            home_goals = predicted_goals.get('HomeGoals', 0)
                            away_goals = predicted_goals.get('AwayGoals', 0)
                            total_goals = home_goals + away_goals

                            if total_goals < 2:
                                average_goals_category = "Under 1.5"
                            elif total_goals == 2:
                                average_goals_category = "Over 1.5"
                            else:
                                average_goals_category = "Over 2.5"

                            predictions.append({
                                'home_team': home_team,
                                'away_team': away_team,
                                'predicted_result': predicted_result,
                                'predicted_home_goals': home_goals,
                                'predicted_away_goals': away_goals,
                                'average_goals_category': average_goals_category
                            })
            except ValueError:
                predictions = []

    return render(
        request,
        'predict/matchday_predictions.html',
        {'matchdays': matchdays, 'competitions': competitions, 'predictions': predictions}
    )
