# tasks.py
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from celery import shared_task
from django.core.cache import cache
from datetime import date

from predict.views import fetch_actual_results
from .models import MatchPrediction


from .utils import (
    fetch_and_cache_team_metadata,
    fetch_matches_by_date,
    fetch_training_data_all_seasons,
    find_next_match_date,
    generate_predictions_for_date,
    get_league_table,
    get_top_predictions,
    save_predictions,
    store_top_pick_for_date
)
import pandas as pd

API_KEY = "7419be10abd14d7fb752e6fe6491e38f"
 
COMPETITIONS = {
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

CACHE_TIMEOUT = 60 * 60 * 24 * 7  # 1 week

def get_or_cache_training_data(comp_code):
    key = f"training_data_{comp_code}"
    cached_df = cache.get(key)
    if cached_df is not None:
        print(f"[INFO] Loaded cached training data for {comp_code}")
        return cached_df

    print(f"[INFO] Fetching and caching training data for {comp_code}")
    df = fetch_training_data_all_seasons(comp_code)
    if not df.empty:
        cache.set(key, df, CACHE_TIMEOUT)
    return df

@shared_task
def schedule_predictions_staggered(match_date=None):
    delay = 0
    for comp in COMPETITIONS:
        print(f"[INFO] Scheduling prediction for {comp} in {delay} seconds")
        predict_next_fixtures_for_competition.apply_async(
            args=[comp, match_date],
            countdown=delay
        )
        delay += 180  # now using 2 minutes instead of 1

@shared_task
def schedule_predictions(match_date=None):
    """Main scheduler: triggers one task per competition with delay."""
    for index, comp in enumerate(COMPETITIONS):
        countdown_seconds = index * 60  # space by 60 seconds
        predict_for_competition.apply_async(
            args=[comp, match_date],
            countdown=countdown_seconds
        )
@shared_task
def trigger_staggered_scheduling():
    schedule_predictions_staggered.delay()

@shared_task
def predict_for_competition(competition_code, match_date=None):
    """Predicts matches for a single competition (copied logic from predict_next_fixtures)."""
    from .models import MatchPrediction
    from .utils import (
        fetch_matches_by_date,
        save_predictions
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    from datetime import datetime, timedelta

    def find_next_match_date(comp, days=10):
        today = datetime.today()
        for i in range(days):
            date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
            matches = fetch_matches_by_date(API_KEY,comp, date)
            if matches:
                return date
        return None

    if not match_date:
        match_date = find_next_match_date(competition_code)
        if not match_date:
            print(f"[WARN] No match date found for {competition_code}")
            return

    matches = fetch_matches_by_date(API_KEY,competition_code, match_date)
    if not matches:
        print(f"[WARN] No matches found for {competition_code} on {match_date}")
        return
    df = fetch_training_data_all_seasons(competition_code)
    if df.empty:
        print(f"[WARN] No training data for {competition_code}")
        return

    features = ['home_team', 'away_team']
    label_home = 'home_goals'
    label_away = 'away_goals'

    team_names = pd.concat([df['home_team'], df['away_team']]).unique()
    le = LabelEncoder()
    le.fit(team_names)

    for col in features:
        df[col] = le.transform(df[col])

    X = df[features]
    y_home = df[label_home]
    y_away = df[label_away]

    model_home = RandomForestClassifier(n_estimators=100, random_state=42)
    model_home.fit(X, y_home)

    model_away = RandomForestClassifier(n_estimators=100, random_state=42)
    model_away.fit(X, y_away)

    predictions = save_predictions(
        matches, model_home, model_away, le,
        match_date=match_date,
        competition_code=competition_code
    )

    print(f"[INFO] Saved {len(predictions)} predictions for {competition_code} on {match_date}")

@shared_task
def predict_next_fixtures_for_competition(competition_code, match_date=None):
     
    print(f"[INFO] Running prediction for {competition_code} on {match_date if match_date else 'auto'}")

    if not match_date:
        match_date_to_use = find_next_match_date(fetch_matches_by_date, None, [competition_code])
        if not match_date_to_use:
            return
    else:
        match_date_to_use = match_date

    print(f"[INFO] Processing competition: {competition_code} for {match_date_to_use}")
    matches = fetch_matches_by_date(API_KEY,competition_code, match_date_to_use)
    if not matches:
        print(f"[WARN] No matches found for {competition_code} on {match_date_to_use}")
        return

    df = cache.get(f"training_data_{competition_code}")
    if df is None:
        df = fetch_training_data_all_seasons(competition_code)
        cache.set(f"training_data_{competition_code}", df, timeout=60 * 60 * 24 * 7)  # cache for 7 days

    if df.empty:
        print(f"[WARN] No training data for {competition_code}")
        return

    features = ['home_team', 'away_team']
    label_home = 'home_goals'
    label_away = 'away_goals'

    team_names = pd.concat([df['home_team'], df['away_team']]).unique()
    le = LabelEncoder()
    le.fit(team_names)

    for col in features:
        df[col] = le.transform(df[col])

    X = df[features]
    y_home = df[label_home]
    y_away = df[label_away]

    model_home = RandomForestClassifier(n_estimators=100, random_state=42)
    model_home.fit(X, y_home)

    model_away = RandomForestClassifier(n_estimators=100, random_state=42)
    model_away.fit(X, y_away)

    

    predictions = save_predictions(
        matches, model_home, model_away, le,
        match_date=match_date_to_use,
        competition_code=competition_code
    )

    print(f"[INFO] Saved {len(predictions)} predictions for {competition_code}")

@shared_task
def cache_training_data():
    print("[CACHE] Starting training data caching")
    for comp in COMPETITIONS:
        key = f"training_data_{comp}"
        df = fetch_training_data_all_seasons(comp)

        if df is not None and not df.empty:
            cache.set(key, df, timeout=60 * 60 * 24 * 7)
            print(f"[CACHE] Cached {len(df)} records for {comp}")
        else:
            print(f"[CACHE] No data for {comp}")


@shared_task
def refresh_all_league_tables():
    for code in COMPETITIONS:
        print(f"[AUTO] Refreshing league table for {code}")
        get_league_table(code)

@shared_task
def update_metadata_task():
    for code in COMPETITIONS:
        fetch_and_cache_team_metadata()

@shared_task
def store_daily_top_pick():
    from .utils import get_top_predictions  # your prediction logic
    predictions = get_top_predictions(limit=10)
    store_top_pick_for_date(predictions)

@shared_task
def update_match_status_task():
    today = date.today()

    timed = MatchPrediction.objects.filter(match_date__gte=today).update(status="TIMED")
    finished = MatchPrediction.objects.filter(match_date__lt=today).update(status="FINISHED")

    return {
        "updated_timed": timed,
        "updated_finished": finished,
    }

@shared_task
def update_actual_results_for_competition(competition_code, match_date=None):
    if not match_date:
        match_date = date.today().strftime("%Y-%m-%d")

    results = fetch_actual_results(competition_code, match_date)
    print(f"[INFO] {len(results)} actual results found for {competition_code} on {match_date}")

    updated = 0
    for res in results:
        try:
            prediction = MatchPrediction.objects.get(
                home_team=res['home_team'],
                away_team=res['away_team'],
                match_date=match_date,
                competition=competition_code,
            )
            prediction.actual_home_goals = res['actual_home_goals']
            prediction.actual_away_goals = res['actual_away_goals']

            # Evaluate accuracy if predictions exist
            if prediction.predicted_home_goals is not None and prediction.predicted_away_goals is not None:
                predicted_result = (
                    "Home" if prediction.predicted_home_goals > prediction.predicted_away_goals
                    else "Away" if prediction.predicted_home_goals < prediction.predicted_away_goals
                    else "Draw"
                )
                prediction.is_accurate = (predicted_result == res["actual_result"])

            prediction.status = "FINISHED"
            prediction.save()
            updated += 1
        except MatchPrediction.DoesNotExist:
            print(f"[WARN] Prediction not found for: {res['home_team']} vs {res['away_team']}")

    print(f"[INFO] Updated {updated} match predictions with actual results.")
    return updated

@shared_task
def update_actual_results_all_competitions(match_date=None):
    from .tasks import COMPETITIONS
    for comp in COMPETITIONS:
        update_actual_results_for_competition.delay(comp, match_date)
