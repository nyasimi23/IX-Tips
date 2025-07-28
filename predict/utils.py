import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from django.core.cache import cache
from django.utils import timezone

from predict.constants import COMPETITIONS,get_team_metadata


# ========== API Config ==========
API_TOKEN = '7419be10abd14d7fb752e6fe6491e38f'  # Replace with your actual token
BASE_URL = 'https://api.football-data.org/v4'
HEADERS = {'X-Auth-Token': API_TOKEN}

# ========== API Data Fetching ==========
def fetch_season_matches(api_key, competition_code, season):
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches?season={season}"
    headers = {"X-Auth-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json().get("matches", [])
    else:
        print(f"[ERROR] Failed to fetch season {season} for {competition_code}: {response.status_code}")
        return []

def fetch_matches_by_season(api_key, competition_code, season_year):
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches?season={season_year}"
    headers = {"X-Auth-Token": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("matches", [])
    else:
        print(f"[WARN] No data for {competition_code} season {season_year}")
        return []

def fetch_competition_matches(competition_id, date_from=None, date_to=None):
    """
    Fetch matches from a specific competition within a date range.
    """
    url = f"{BASE_URL}/competitions/{competition_id}/matches"
    params = {}
    if date_from:
        params['dateFrom'] = date_from
    if date_to:
        params['dateTo'] = date_to

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get('matches', [])
    return []

def find_next_match_date(fetch_fn, api_key, competition_codes, past=False):
    today = datetime.today()
    direction = -1 if past else 1  # use -1 to go backwards in time
    for i in range(30):
        date = (today + timedelta(days=direction * i)).strftime("%Y-%m-%d")
        for comp in competition_codes:
            matches = fetch_fn(API_TOKEN,comp, date)
            if matches:
                print(f"[DEBUG] Found {len(matches)} match(es) on {date} for {comp}")
                return date
    print("[WARN] No matches found for testing.")
    return None

def fetch_matches_by_date(api_key, competition_code, match_date, retries=1, delay=8):
    url = "https://api.football-data.org/v4/matches"
    headers = {
        "X-Auth-Token": api_key

    }
    match_date_obj = datetime.strptime(match_date, "%Y-%m-%d")
    date_from = match_date_obj.strftime("%Y-%m-%d")
    date_to = (match_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    params = {
        "competitions": competition_code,
        "dateFrom": date_from,
        "dateTo": date_to
    }

    for attempt in range(retries):
        response = requests.get(url, headers=headers, params=params)

        print(f"[DEBUG] Attempt {attempt+1}: Request URL: {response.url}")
        print(f"[DEBUG] Status: {response.status_code}")
        print(f"[DEBUG] Response: {response.text[:300]}")

        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            if matches:
                print(f"[DEBUG] {len(matches)} matches found on {match_date} for {competition_code}")
                return matches
            else:
                print(f"[DEBUG] No matches yet, retrying in {delay}s...")
                time.sleep(delay)
        else:
            print(f"[ERROR] Failed to fetch matches: {response.status_code}")
            break
    
    return []  # Return empty if all retries fail
def get_actual_results(api_key, competition_code, match_date):
    matches = fetch_matches_by_date(api_key, competition_code, match_date)
    results = []
    for match in matches:
        if match['status'] == 'FINISHED':
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_goals = match['score']['fullTime']['home']
            away_goals = match['score']['fullTime']['away']
            if home_goals is not None and away_goals is not None:
                if home_goals > away_goals:
                    result = "Home Win"
                elif away_goals > home_goals:
                    result = "Away Win"
                else:
                    result = "Draw"
                results.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'actual_home_goals': home_goals,
                    'actual_away_goals': away_goals,
                    'actual_result': result
                })
    return results

# ========== Preprocessing and Model Training ==========

def preprocess_api_data(df):
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)
    X = df[["home_team", "away_team"]]
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    label_encoder = LabelEncoder()
    X_encoded = X.apply(label_encoder.fit_transform)
    return X_encoded, y_home, y_away, label_encoder


def train_models(X, y_home, y_away):
    # Encode teams
    encoder = LabelEncoder()
    all_teams = pd.concat([X['home_team'], X['away_team']]).unique()
    encoder.fit(all_teams)
    X_encoded = X.copy()
    X_encoded['home_team'] = encoder.transform(X_encoded['home_team'])
    X_encoded['away_team'] = encoder.transform(X_encoded['away_team'])

    model_home = RandomForestRegressor(n_estimators=100, random_state=42)
    model_away = RandomForestRegressor(n_estimators=100, random_state=42)

    model_home.fit(X_encoded, y_home)
    model_away.fit(X_encoded, y_away)

    return model_home, model_away, encoder

#====== Prediction Logic ==========

def predict_match(home_team, away_team, model_dict, encoder):
    X_new = pd.DataFrame([[home_team, away_team]], columns=["home_team", "away_team"])
    X_encoded = X_new.apply(encoder.transform)
    home_goals = int(np.round(model_dict["home"].predict(X_encoded)[0]))
    away_goals = int(np.round(model_dict["away"].predict(X_encoded)[0]))

    if home_goals > away_goals:
        result = "HOME"
    elif away_goals > home_goals:
        result = "AWAY"
    else:
        result = "DRAW"

    return result, f"{home_goals} - {away_goals}", home_goals, away_goals

def predict_match_outcome(home_team, away_team, models, label_encoder):
    model_home, model_away, model_result = models
    home_encoded = label_encoder.transform([home_team])[0]
    away_encoded = label_encoder.transform([away_team])[0]
    X = [[home_encoded, away_encoded]]

    pred_home_goals = round(model_home.predict(X)[0])
    pred_away_goals = round(model_away.predict(X)[0])

    if pred_home_goals > pred_away_goals:
        result = "Home Win"
    elif pred_away_goals > pred_home_goals:
        result = "Away Win"
    else:
        result = "Draw"

    return result, pred_home_goals, pred_away_goals

def find_next_available_match_date(api_key, competition_code, start_date, days_ahead=30):
    """
    Looks ahead up to `days_ahead` days to find the next date with at least one match.
    """
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": api_key}
    
    for i in range(days_ahead):
        check_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)).date().isoformat()
        params = {
            "competitions": competition_code,
            "dateFrom": check_date,
            "dateTo": check_date
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            matches = response.json().get("matches", [])
            if matches:
                return check_date, matches  # Return first valid date and matches
    return None, []  # No match found in that range
# ========== Utility ==========

def process_match_data(matches):
    data = []
    for match in matches:
        try:
            if match['status'] == 'FINISHED':
                data.append({
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_goals': match['score']['fullTime']['home'],
                    'away_goals': match['score']['fullTime']['away']
                })
        except Exception as e:
            print(f"[ERROR] Failed to process match: {e}")
            continue
    df = pd.DataFrame(data)
    print("[DEBUG] Processed DataFrame:\n", df.head())
    return df
def make_predictions(model_home, model_away, X, df):
    preds_home = model_home.predict(X)
    preds_away = model_away.predict(X)

    predictions = []
    for i in range(len(df)):
        predictions.append({
            "homeTeam": df.iloc[i]["home_team"],
            "awayTeam": df.iloc[i]["away_team"],
            "prediction": "Home Win" if preds_home[i] > preds_away[i] else "Away Win",
            "home_score": round(preds_home[i]),
            "away_score": round(preds_away[i]),
            "date": df.iloc[i]["utc_date"]
        })
    print(f"[DEBUG] Predictions: {predictions}")

    return predictions

def preprocess_match_data(matches, return_df=False):
    data = []
    for match in matches:
        try:
            data.append({
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "utc_date": match["utcDate"],
                "home_position": match["homeTeam"].get("position", 10),  # fallback
                "away_position": match["awayTeam"].get("position", 10),  # fallback
                "home_points": match["homeTeam"].get("points", 30),     # fallback
                "away_points": match["awayTeam"].get("points", 30),     # fallback
                "home_goals": match["score"]["fullTime"].get("home", 1),
                "away_goals": match["score"]["fullTime"].get("away", 1),
            })
        except KeyError as e:
            print(f"[ERROR] Missing key: {e} in match: {match}")

    df = pd.DataFrame(data)

    # Features
    X = df[["home_position", "away_position", "home_points", "away_points"]]
    y_home = df["home_goals"]
    y_away = df["away_goals"]
    
    if return_df:
        return X, y_home, y_away, df
    return X, y_home, y_away

def fetch_training_data(competition_code):
    API_KEY = "7419be10abd14d7fb752e6fe6491e38f"  # Replace with your secure token handling
    headers = {"X-Auth-Token": API_KEY}
    all_matches = []

    for season in range(2019, 2025):  # Example: seasons 2017–2023
        url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
        params = {"season": season}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            for match in matches:
                if match['status'] == 'FINISHED':
                    all_matches.append({
                        "home_team": match['homeTeam']['name'],
                        "away_team": match['awayTeam']['name'],
                        "home_goals": match['score']['fullTime']['home'],
                        "away_goals": match['score']['fullTime']['away']
                    })
        else:
            print(f"[WARN] Failed to fetch data for season {season}, status {response.status_code}")

    return pd.DataFrame(all_matches)

from .models import MatchPrediction
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def save_predictions(matches, model_home, model_away, le, match_date, competition_code, actual_result_map=None):
    saved_predictions = []

    for match in matches:
        home = match['homeTeam']['name']
        away = match['awayTeam']['name']
        match_id = match['id']

        if home not in le.classes_ or away not in le.classes_:
            print(f"[WARN] Skipping unknown teams: {home} or {away}")
            continue

        try:
            input_df = pd.DataFrame({'home_team': [home], 'away_team': [away]})
            input_df['home_team'] = le.transform(input_df['home_team'])
            input_df['away_team'] = le.transform(input_df['away_team'])

            pred_home = model_home.predict(input_df)[0]
            pred_away = model_away.predict(input_df)[0]

            prediction, _ = MatchPrediction.objects.update_or_create(
                match_id=match_id,
                defaults={
                    'match_date': match_date,
                    'competition': competition_code,
                    'home_team': home,
                    'away_team': away,
                    'predicted_home_goals': int(pred_home),
                    'predicted_away_goals': int(pred_away)
                }
            )

            if actual_result_map:
                result = actual_result_map.get((home, away))
                if result:
                    prediction.actual_home_goals = result['home_goals']
                    prediction.actual_away_goals = result['away_goals']

                    actual_result = 'draw'
                    if result['home_goals'] > result['away_goals']:
                        actual_result = 'home'
                    elif result['home_goals'] < result['away_goals']:
                        actual_result = 'away'

                    predicted_result = 'draw'
                    if pred_home > pred_away:
                        predicted_result = 'home'
                    elif pred_home < pred_away:
                        predicted_result = 'away'

                    prediction.is_accurate = actual_result == predicted_result

            prediction.save()
            saved_predictions.append(prediction)

        except Exception as e:
            print(f"[ERROR] Prediction failed for {home} vs {away}: {e}")

    return saved_predictions


def fetch_training_data_all_seasons(competition_code):
    cache_key = f"training_data_{competition_code}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        print(f"[CACHE] Loaded training data for {competition_code} from cache")
        return cached_data

    API_KEY = "7419be10abd14d7fb752e6fe6491e38f"  # Replace with your secure token handling
    headers = {"X-Auth-Token": API_KEY}
    all_matches = []

    for season in range(2019, 2025):  # Example: seasons 2017–2023
        url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
        params = {"season": season}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            matches = data.get("matches", [])
            for match in matches:
                if match['status'] == 'FINISHED':
                    all_matches.append({
                        "home_team": match['homeTeam']['name'],
                        "away_team": match['awayTeam']['name'],
                        "home_goals": match['score']['fullTime']['home'],
                        "away_goals": match['score']['fullTime']['away']
                    })
        else:
            print(f"[WARN] Failed to fetch data for season {season}, status {response.status_code}")

    df = pd.DataFrame(all_matches)
    cache.set(cache_key, df, timeout=60 * 60 * 24 * 7)  # Cache for 1 week
    return df

def get_league_table(competition):
    cache_key = f"standings_{competition}"  # ✅ Define this first

    # Set last updated time
    cache.set(f"{cache_key}_updated", timezone.now(), timeout=60 * 60 * 6)

    # Try cached table
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Fetch from API
    url = f"https://api.football-data.org/v4/competitions/{competition}/standings"
    headers = {"X-Auth-Token": API_TOKEN}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        table = data.get("standings", [])[0].get("table", [])
        cache.set(cache_key, table, timeout=60 * 60 * 6)
        return table
    else:
        print(f"[ERROR] API {response.status_code} {response.text}")
        return []


def fetch_and_cache_team_metadata():
    headers = {"X-Auth-Token": API_TOKEN}

    for comp_code, comp_name in COMPETITIONS.items():
        url = f"https://api.football-data.org/v4/competitions/{comp_code}/teams"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"[WARNING] Skipping {comp_code} - HTTP {response.status_code}")
                continue

            json_data = response.json()

            # Defensive check: skip if no teams
            teams = json_data.get("teams", [])
            if not teams:
                print(f"[WARNING] No teams returned for {comp_code}")
                continue

            # Fetch competition metadata (crest/logo)
            comp_meta = {
                "name": json_data.get("competition", {}).get("name", comp_name),
                "crest": json_data.get("competition", {}).get("emblem", "")
            }

            cache.set(f"competition_meta::{comp_code}", comp_meta, timeout=60 * 60 * 24 * 30)

            # Cache each team’s shortName and crest
            for team in teams:
                team_name = team["name"]
                team_meta = {
                    "shortName": team.get("shortName", team_name),
                    "crest": team.get("crest", ""),
                    "competition": comp_code
                }
                cache.set(f"team_meta::{team_name}", team_meta, timeout=60 * 60 * 24 * 30)

            print(f"[OK] Cached metadata for {comp_code} ({len(teams)} teams)")

        except Exception as e:
            print(f"[EXCEPTION] Error fetching teams for {comp_code}: {e}")

from datetime import date

def get_top_predictions(limit=10):
    today = date.today()
    matches = MatchPrediction.objects.filter(match_date__gte=today).order_by("match_date")

    picks_by_date = {}

    tip_priority = {
        "Over 2.5": 3,
        "GG": 2,
        "1": 1,
        "2": 1,
        "X": 0
    }

    for m in matches:
        tips = []
        meta_home = get_team_metadata(m.home_team)
        meta_away = get_team_metadata(m.away_team)

        margin = m.predicted_home_goals - m.predicted_away_goals

        if abs(margin) >= 1.5:
            if margin > 0:
                tips.append(("1", min(abs(margin) * 10, 40)))  # Home win
            else:
                tips.append(("2", min(abs(margin) * 10, 40)))  # Away win
        elif abs(margin) <= 0.4:
            tips.append(("X", 20))  # Draw

        if m.predicted_home_goals >= 1 and m.predicted_away_goals >= 1:
            tips.append(("GG", 25))

        total_goals = m.predicted_home_goals + m.predicted_away_goals
        if total_goals > 2.5:
            tips.append(("Over 2.5", min((total_goals - 2.5) * 12, 30)))

        if tips:
            # ✅ Force "Over 2.5" tip if total goals ≥ 4
            if total_goals >= 3:
                best_tip = ("Over 2.5", 100)
            else:
                best_tip = sorted(
                    tips,
                    key=lambda x: (x[1], tip_priority.get(x[0], 0)),
                    reverse=True
                )[0]

            match_day = m.match_date.strftime("%Y-%m-%d")
            picks_by_date.setdefault(match_day, []).append({
                "home_team": meta_home.get("shortName", m.home_team),
                "away_team": meta_away.get("shortName", m.away_team),
                "tip": best_tip[0],
                "confidence": f"{best_tip[1]:.0f}",
                "match_date": match_day,
            })

    return picks_by_date
    
def generate_predictions_for_date(date):
    return MatchPrediction.objects.filter(match_date=date, status="TIMED")


from .models import TopPick
def store_top_pick_for_date(predictions_by_date):
    all_picks = []

    for date_str, picks in predictions_by_date.items():
        try:
            match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Remove existing TopPicks for this date to avoid duplicates
        TopPick.objects.filter(match_date=match_date).delete()

        for p in picks:
            match = MatchPrediction.objects.filter(
                home_team=p["home_team"],
                away_team=p["away_team"],
                match_date=match_date
            ).first()

            actual_tip = None
            is_correct = None

            if match and match.status == "FINISHED":
                # Determine result tip from actual goals
                if match.actual_home_goals > match.actual_away_goals:
                    result_tip = "1"
                elif match.actual_home_goals < match.actual_away_goals:
                    result_tip = "2"
                else:
                    result_tip = "X"

                # Determine GG or Over 2.5
                gg = match.actual_home_goals >= 1 and match.actual_away_goals >= 1
                over_2_5 = (match.actual_home_goals + match.actual_away_goals) > 2.5

                # Map tip based on what user predicted
                if p["tip"] == "GG" and gg:
                    actual_tip = "GG"
                elif p["tip"] == "Over 2.5" and over_2_5:
                    actual_tip = "Over 2.5"
                else:
                    actual_tip = result_tip

                # Check if prediction is correct
                is_correct = actual_tip == p["tip"]

            all_picks.append(TopPick(
                match_date=match_date,
                home_team=p["home_team"],
                away_team=p["away_team"],
                tip=p["tip"],
                confidence=p["confidence"],
                actual_tip=actual_tip,
                is_correct=is_correct,
            ))

    if all_picks:
        TopPick.objects.bulk_create(all_picks)
        print(f"[TopPick] Stored {len(all_picks)} picks.")
    else:
        print("[TopPick] No picks to store.")
def update_actuals_for_top_picks(picks_qs):
    picks_to_update = picks_qs.filter(actual_tip__isnull=True)

    for pick in picks_to_update:
        print(f"\n🟡 Processing TopPick: {pick.match_date} | {pick.home_team} vs {pick.away_team}")

        # Apply metadata to TopPick teams
        pick_home_meta = get_team_metadata(pick.home_team)
        pick_away_meta = get_team_metadata(pick.away_team)

        pick_home = pick_home_meta.get("shortName", pick.home_team).lower()
        pick_away = pick_away_meta.get("shortName", pick.away_team).lower()

        # Loop over all MatchPredictions for that date
        match_candidates = MatchPrediction.objects.filter(match_date=pick.match_date)

        matched = None
        for match in match_candidates:
            match_home_meta = get_team_metadata(match.home_team)
            match_away_meta = get_team_metadata(match.away_team)

            match_home = match_home_meta.get("shortName", match.home_team).lower()
            match_away = match_away_meta.get("shortName", match.away_team).lower()

            if pick_home == match_home and pick_away == match_away:
                matched = match
                break

        if not matched:
            print(f"❌ No match found for TopPick on {pick.match_date}: {pick_home} vs {pick_away}")
            continue

        print(f"✅ Found: {matched.home_team} vs {matched.away_team} | Status: {matched.status}")

        if matched.actual_home_goals is not None and matched.actual_away_goals is not None:
            home_goals = matched.actual_home_goals
            away_goals = matched.actual_away_goals

            result_tip = (
                "1" if home_goals > away_goals else
                "2" if home_goals < away_goals else "X"
            )

            gg = home_goals >= 1 and away_goals >= 1
            over_2_5 = (home_goals + away_goals) > 2.5

            if pick.tip == "GG" and gg:
                actual_tip = "GG"
            elif pick.tip == "Over 2.5" and over_2_5:
                actual_tip = "Over 2.5"
            else:
                actual_tip = result_tip

            pick.actual_tip = actual_tip
            pick.is_correct = (pick.tip == actual_tip)
            pick.save()
            print(f"💾 Updated TopPick: actual_tip = {actual_tip}, is_correct = {pick.is_correct}")
        else:
            print("⚠️ Match found but actual goals missing.")
