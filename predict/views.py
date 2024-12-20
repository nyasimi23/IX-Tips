import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render

API_KEY = '7419be10abd14d7fb752e6fe6491e38f'
BASE_URL = "https://api.football-data.org/v4"

def fetch_pl_matches(api_key, season):
    url = f"{BASE_URL}/competitions/PD/matches?season={season}"
    headers = {
        "X-Auth-Token": api_key
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        matches = data['matches']
        match_data = []
        for match in matches:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            score = f"{match['score']['fullTime']['home']} - {match['score']['fullTime']['away']}"
            match_data.append([home_team, away_team, score])
        return pd.DataFrame(match_data, columns=['Home Team', 'Away Team', 'Score'])
    else:
        return None

def preprocess_api_data(api_df):
    api_df[['HomeGoals', 'AwayGoals']] = api_df['Score'].str.split(' - ', expand=True)
    api_df['HomeGoals'] = pd.to_numeric(api_df['HomeGoals'])
    api_df['AwayGoals'] = pd.to_numeric(api_df['AwayGoals'])
    api_df['FTR'] = api_df.apply(lambda row: 'Home' if row['HomeGoals'] > row['AwayGoals']
                                 else ('Away' if row['HomeGoals'] < row['AwayGoals'] else 'Draw'), axis=1)
    return api_df[['Home Team', 'Away Team', 'HomeGoals', 'AwayGoals', 'FTR']]

def train_models(df):
    X = df[['Home Team', 'Away Team']]
    y_classification = df['FTR']
    y_regression = df[['HomeGoals', 'AwayGoals']]

    label_encoder_X = LabelEncoder()
    X['Home Team'] = label_encoder_X.fit_transform(X['Home Team'])
    X['Away Team'] = label_encoder_X.transform(X['Away Team'])

    label_encoder_y_classification = LabelEncoder()
    y_classification_encoded = label_encoder_y_classification.fit_transform(y_classification)

    X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification_encoded, test_size=0.2, random_state=42)

    classifier_model = DecisionTreeClassifier(random_state=42)
    classifier_model.fit(X_train, y_train_classification)

    X_train, X_test, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=42)

    regressor_models = {}
    for column in y_regression.columns:
        regressor_model = DecisionTreeRegressor(random_state=42)
        regressor_model.fit(X_train, y_train_regression[column])
        regressor_models[column] = regressor_model

    return classifier_model, label_encoder_y_classification, regressor_models, label_encoder_X

def predict_match_result(home_team, away_team, classifier_model, label_encoder_X, label_encoder_y_classification):
    try:
        # Handle unseen teams by assigning a default value or using a placeholder for unknown teams
        home_team_encoded = safe_encode(home_team, label_encoder_X)
        away_team_encoded = safe_encode(away_team, label_encoder_X)
        
        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]
        
        # Make prediction
        prediction = classifier_model.predict(match_data)
        predicted_result = label_encoder_y_classification.inverse_transform(prediction)[0]
        
        return predicted_result
    except ValueError as e:
        print(f"Error: {e}")
        return "Unknown"

# Function to safely encode team names (handling unseen teams)
def safe_encode(team_name, label_encoder):
    try:
        return label_encoder.transform([team_name])[0]
    except ValueError:
        # If the team is unseen, you can either use an unknown value or a placeholder
        return -1  # For example, use -1 as the placeholder for unseen teams

# Function to predict goals
def predict_goals(home_team, away_team, regressor_models, label_encoder_X):
    try:
        # Handle unseen teams by assigning a default value or using a placeholder for unknown teams
        home_team_encoded = safe_encode(home_team, label_encoder_X)
        away_team_encoded = safe_encode(away_team, label_encoder_X)
        
        # Create input data for prediction
        match_data = [[home_team_encoded, away_team_encoded]]
        
        # Make predictions for each goal feature
        predictions = {}
        for column, model in regressor_models.items():
            prediction = model.predict(match_data)
            rounded_prediction = round(prediction[0])  # Round to the nearest integer
            predictions[column] = rounded_prediction
        
        return predictions
    except ValueError as e:
        print(f"Error: {e}")
        return None
def matchday_predictions(request):
    all_seasons_data = []
    seasons = [2019, 2020, 2021, 2022, 2023]

    for season in seasons:
        season_data = fetch_pl_matches(API_KEY, season)
        if season_data is not None:
            all_seasons_data.append(season_data)

    if all_seasons_data:
        all_seasons_df = pd.concat(all_seasons_data, ignore_index=True)
        processed_data = preprocess_api_data(all_seasons_df)

        classifier_model, label_encoder_y_classification, regressor_models, label_encoder_X = train_models(processed_data)

        matchday = 16  # Example: matchday for predictions
        matches = get_matchday_fixtures("PD", matchday)

        predictions = []
        if matches:
            for match in matches:
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
                # Predict result and goals
                predicted_result = predict_match_result(home_team, away_team, classifier_model, label_encoder_X, label_encoder_y_classification)
                predicted_goals = predict_goals(home_team, away_team, regressor_models, label_encoder_X)
                
                home_goals = predicted_goals.get('HomeGoals', 0)
                away_goals = predicted_goals.get('AwayGoals', 0)
                total_goals = home_goals + away_goals
                
                # Determine the average goal category
                if total_goals < 2:
                    average_goals_category = "Under 1.5"
                elif total_goals == 2:
                    average_goals_category = "Over 1.5"
                else:
                    average_goals_category = "Over 2.5"

                # Add prediction details
                predictions.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_result': predicted_result,
                    'predicted_home_goals': home_goals,
                    'predicted_away_goals': away_goals,
                    'average_goals_category': average_goals_category
                })

        return render(request, 'predict/matchday_predictions.html', {'predictions': predictions})
    else:
        return render(request, 'predict/matchday_predictions.html', {'predictions': []})
    
def get_matchday_fixtures(competition_id, matchday):
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    headers = {
        "X-Auth-Token": API_KEY
    }
    params = {"matchday": matchday}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['matches']
    return []

