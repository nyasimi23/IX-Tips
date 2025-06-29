from django.core.management.base import BaseCommand
from django.utils.timezone import now
from datetime import timedelta, datetime
from predict.models import MatchPrediction
from predict.utils import (
    fetch_matches_by_date,
    fetch_competition_matches,
    preprocess_api_data,
    train_models,
    predict_match_outcome,
    get_actual_results,
)
from django.core.cache import cache
import pandas as pd

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

class Command(BaseCommand):
    help = 'Predicts upcoming fixtures and stores them in the database'

    def handle(self, *args, **options):
        date_checked = now().date()
        max_days_ahead = 7
        found_fixtures = False

        for day in range(max_days_ahead):
            match_date = (date_checked + timedelta(days=day)).strftime('%Y-%m-%d')
            for comp_code in competitions:
                matches = fetch_matches_by_date(API_KEY, comp_code, match_date)
                if not matches:
                    continue

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

                # Use cache if available
                cached_models = cache.get(f"{comp_code}_models")
                if cached_models:
                    regressor_models, label_encoder_X = cached_models
                else:
                    regressor_models, label_encoder_X = train_models(processed_data)
                    cache.set(f"{comp_code}_models", (regressor_models, label_encoder_X), timeout=3600)

                for match in matches:
                    home_team = match['homeTeam']['name']
                    away_team = match['awayTeam']['name']
                    match_status = match['status']

                    if MatchPrediction.objects.filter(
                        competition=competitions[comp_code],
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                    ).exists():
                        continue  # Skip if already predicted

                    predicted_result, home_goals, away_goals = predict_match_outcome(
                        home_team, away_team, regressor_models, label_encoder_X
                    )
                    predicted_score = f"{home_goals} - {away_goals}"

                    gg = "Yes" if home_goals >= 1 and away_goals >= 1 else "No"
                    average_goals_category = "Over 1.5" if (home_goals + away_goals) >= 2 else "Under 1.5"

                    MatchPrediction.objects.create(
                        competition=competitions[comp_code],
                        home_team=home_team,
                        away_team=away_team,
                        match_date=match_date,
                        predicted_result=predicted_result,
                        predicted_score=predicted_score,
                        status=match_status,
                        gg=gg,
                        average_goals_category=average_goals_category,
                    )

                found_fixtures = True

            if found_fixtures:
                self.stdout.write(self.style.SUCCESS(f"Predictions stored for {match_date}"))
                break

        if not found_fixtures:
            self.stdout.write(self.style.WARNING("No fixtures found in the next 7 days."))
