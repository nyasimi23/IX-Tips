# predict/match_predictor.py
from datetime import datetime, timedelta

from predict.tasks import COMPETITIONS
from predict.views import fetch_training_data_all_seasons
from .utils import (
    fetch_matches_by_date, get_actual_results, fetch_competition_matches,
    preprocess_api_data, train_models, predict_match_outcome
)
from django.core.cache import cache
from .models import MatchPrediction

API_KEY = '7419be10abd14d7fb752e6fe6491e38f'

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


def train_and_cache_models():
    for comp_code in COMPETITIONS:
        df = fetch_training_data_all_seasons(comp_code)

        if df.empty:
            print(f"[WARN] No training data for {comp_code}")
            continue

        features = ['home_team', 'away_team']
        label_home = 'home_goals'
        label_away = 'away_goals'

        team_names = pd.concat([df['home_team'], df['away_team']]).unique()
        le = LabelEncoder()
        le.fit(team_names)

        df['home_team'] = le.transform(df['home_team'])
        df['away_team'] = le.transform(df['away_team'])

        X = df[features]
        y_home = df[label_home]
        y_away = df[label_away]

        model_home = RandomForestClassifier(n_estimators=100, random_state=42)
        model_away = RandomForestClassifier(n_estimators=100, random_state=42)

        model_home.fit(X, y_home)
        model_away.fit(X, y_away)

        cache.set(f"model_home_{comp_code}", model_home)
        cache.set(f"model_away_{comp_code}", model_away)
        cache.set(f"encoder_{comp_code}", le)

        print(f"[INFO] Cached model and encoder for {comp_code}")


def predict_and_store_fixtures_for_today():
    today = datetime.now().strftime("%Y-%m-%d")

    for comp_code, comp_name in competitions.items():
        matches = fetch_matches_by_date(API_KEY, comp_code, today)

        # If no matches today, try next 5 days
        if not matches:
            for i in range(1, 6):
                future_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                matches = fetch_matches_by_date(API_KEY, comp_code, future_date)
                if matches:
                    today = future_date
                    break

        if not matches:
            continue

        # Load and cache models
        seasons = [2019, 2020, 2021, 2022, 2023, 2024]
        all_seasons_data = []
        for season in seasons:
            season_data = fetch_competition_matches(API_KEY, comp_code, season)
            if season_data is not None:
                all_seasons_data.append(season_data)

        if not all_seasons_data:
            continue

        all_seasons_df = pd.concat(all_seasons_data, ignore_index=True)
        processed_data = preprocess_api_data(all_seasons_df)

        cached_models = cache.get(f"{comp_code}_models")
        if cached_models:
            regressor_models, label_encoder_X = cached_models
        else:
            regressor_models, label_encoder_X = train_models(processed_data)
            cache.set(f"{comp_code}_models", (regressor_models, label_encoder_X), timeout=3600)

        for match in matches:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            status = match["status"]

            # Check if prediction exists
            if MatchPrediction.objects.filter(
                competition=comp_name,
                home_team=home_team,
                away_team=away_team,
                match_date=today
            ).exists():
                continue

            predicted_result, home_goals, away_goals = predict_match_outcome(
                home_team, away_team, regressor_models, label_encoder_X
            )

            predicted_score = f"{home_goals} - {away_goals}"

            MatchPrediction.objects.create(
                competition=comp_name,
                home_team=home_team,
                away_team=away_team,
                match_date=today,
                predicted_result=predicted_result,
                predicted_score=predicted_score,
                status=status,
            )
