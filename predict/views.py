# views.py (updated - uses The Odds API fallback for odds)
import csv
import difflib
import os
import re
import json
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from collections import defaultdict
from urllib.parse import quote

from django.conf import settings
from django.core.paginator import Paginator
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.contrib import messages
from django.core.cache import cache
from django.templatetags.static import static
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.template.loader import render_to_string
from django.views.decorators.http import require_GET

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from celery import current_app
from django_celery_beat.models import PeriodicTask

from .models import MatchOdds, MatchPrediction, TopPick
from .forms import ActualResultForm, PredictionForm, LivePredictionForm
from .utils import (
    fetch_matches_by_date,
    get_top_predictions as utils_get_top_predictions,
    predict_match_outcome,
    preprocess_api_data,
    #store_top_pick_for_date,
    train_models,
    get_league_table,
    fetch_training_data,
    fetch_matches_by_season,
    fetch_training_data_all_seasons,
    find_next_available_match_date,
    find_next_match_date,
    preprocess_match_data,
    process_match_data,
    update_actuals_for_top_picks,
    
    COMPETITIONS
)
from .generate_logo_mapping import TEAM_LOGOS

# -------------------------
# CONFIG
# -------------------------
API_KEY = getattr(settings, "FOOTBALL_DATA_API_KEY", os.getenv("FOOTBALL_DATA_API_KEY", "7419be10abd14d7fb752e6fe6491e38f"))
BASE_URL = "https://api.football-data.org/v4"

# ODDS provider (the-odds-api)
ODDS_API_KEY = getattr(settings, "ODDS_API_KEY", os.getenv("ODDS_API_KEY", "361d4c96a69e73be8db0953eca372dc1"))
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

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
competitions = COMPETITIONS
# Map competition codes to The Odds API sport keys (extend as needed)
COMPETITION_SPORT_MAP = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "ELC": "soccer_gbr_championship",
    "CL": "soccer_uefa_champs_league",
    "BSA": "soccer_brazil_serie_a",
    "CLI": "soccer_copa_libertadores",  # approximate
    "WC": "soccer_fifa_world_cup",
    # add more mappings as needed
}

# Cache timeout for odds (seconds)
ODDS_CACHE_TIMEOUT = 60 * 10  # 10 minutes
PREFERRED_BOOKMAKERS = ["1xBet", "Tipico"]
ODDS_API_KEY = "361d4c96a69e73be8db0953eca372dc1"

# -------------------------
# Utilities for team normalization / fuzzy matching
# -------------------------
def fetch_odds(sport_key):
    """Fetch odds for one competition"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "regions": "eu",
        "markets": "h2h,totals,btts",
        "oddsFormat": "decimal",
        "apiKey": ODDS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def extract_odds_from_bookmaker(bookmaker, game):
    odds_out = {
        "home": None,
        "draw": None,
        "away": None,
        "over25": None,
        "under25": None,
        "btts_yes": None,
    }
    for mk in bookmaker.get("markets", []):
        if mk["key"] == "h2h":
            for outcome in mk["outcomes"]:
                if outcome["name"] == game["home_team"]:
                    odds_out["home"] = outcome["price"]
                elif outcome["name"] == game["away_team"]:
                    odds_out["away"] = outcome["price"]
                elif outcome["name"].lower() in ["draw", "x"]:
                    odds_out["draw"] = outcome["price"]

        elif mk["key"] == "totals":
            for outcome in mk["outcomes"]:
                if outcome.get("point") == 2.5:
                    if outcome["name"].lower() == "over":
                        odds_out["over25"] = outcome["price"]
                    elif outcome["name"].lower() == "under":
                        odds_out["under25"] = outcome["price"]

        elif mk["key"] == "btts":
            for outcome in mk["outcomes"]:
                if outcome["name"].lower() == "yes":
                    odds_out["btts_yes"] = outcome["price"]

    return odds_out


def normalize_team_name(team, names):
    match = difflib.get_close_matches(team, names, n=1, cutoff=0.7)
    return match[0] if match else team


def update_odds_in_db(competition_code):
    """Fetch odds for a competition and update DB cache"""
    sport_key = COMPETITION_SPORT_MAP.get(competition_code)
    if not sport_key:
        return 0

    odds_data = fetch_odds(sport_key)
    saved_count = 0

    for game in odds_data:
        home, away = game["home_team"], game["away_team"]

        bookmaker = None
        for pref in PREFERRED_BOOKMAKERS:
            bookmaker = next((b for b in game.get("bookmakers", []) if b["title"] == pref), None)
            if bookmaker:
                break
        if not bookmaker and game.get("bookmakers"):
            bookmaker = game["bookmakers"][0]

        if bookmaker:
            odds = extract_odds_from_bookmaker(bookmaker, game)
            MatchOdds.objects.update_or_create(
                competition_code=competition_code,
                home_team=home,
                away_team=away,
                defaults={
                    "home_win": odds["home"],
                    "draw": odds["draw"],
                    "away_win": odds["away"],
                    "over25": odds["over25"],
                    "under25": odds["under25"],
                    "btts_yes": odds["btts_yes"],
                    "bookmaker": bookmaker["title"],
                },
            )
            saved_count += 1
    return saved_count


def update_all_odds():
    """Fetch and update odds for ALL competitions in COMPETITION_SPORT_MAP"""
    total_saved = 0
    for comp in COMPETITION_SPORT_MAP.keys():
        total_saved += update_odds_in_db(comp)
    return total_saved

def view_odds(request):
    competition = request.GET.get("competition", "EPL")
    sport_key = COMPETITION_SPORT_MAP.get(competition, "soccer_epl")

    # For testing, just grab all odds data for the competition
    odds_data = fetch_odds(sport_key)

    return render(request, "predict/view_odds.html", {
        "competition": competition,
        "sport_key": sport_key,
        "data": odds_data,  # full data for debugging in template
    })

def fetch_odds_for_match(match, competition_code="EPL"):
    sport_key = COMPETITION_SPORT_MAP.get(competition_code, "soccer_epl")
    odds_data = fetch_odds(sport_key)

    if not odds_data or "error" in odds_data:
        return None

    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]

    # Collect available names from API
    api_names = []
    for game in odds_data:
        api_names.extend([game["home_team"], game["away_team"]])

    # Normalize names
    home_norm = normalize_team_name(home, api_names)
    away_norm = normalize_team_name(away, api_names)

    for game in odds_data:
        if (game["home_team"] == home_norm and game["away_team"] == away_norm) or \
           (game["home_team"] == away_norm and game["away_team"] == home_norm):

            # ✅ Try preferred bookmakers first
            for pref in PREFERRED_BOOKMAKERS:
                for bookmaker in game.get("bookmakers", []):
                    if bookmaker["title"] == pref:
                        return extract_odds_from_bookmaker(bookmaker, game)

            # ✅ fallback: use the first available bookmaker
            if game.get("bookmakers"):
                return extract_odds_from_bookmaker(game["bookmakers"][0], game)

    return None

    
def get_top_predictions(limit=10):
    today = date.today()
    matches = MatchPrediction.objects.filter(
        match_date__gte=today
    ).select_related("odds")  # ✅ preload odds to avoid extra queries

    picks_by_date = {}
    tip_priority = {"Over 2.5": 3, "GG": 2, "1": 1, "2": 1, "X": 0}

    for m in matches:
        tips = []
        margin = m.predicted_home_goals - m.predicted_away_goals
        total_goals = m.predicted_home_goals + m.predicted_away_goals

        # --- Model-based tips ---
        if abs(margin) >= 1.5:
            tips.append(("1" if margin > 0 else "2", min(abs(margin) * 10, 40)))
        elif abs(margin) <= 0.4:
            tips.append(("X", 20))

        if m.predicted_home_goals >= 1 and m.predicted_away_goals >= 1:
            tips.append(("GG", 25))

        if total_goals > 2.5:
            tips.append(("Over 2.5", min((total_goals - 2.5) * 12, 30)))

        if not tips:
            continue

        # ✅ Force Over 2.5 if total goals ≥ 3
        if total_goals >= 3:
            best_tip = ("Over 2.5", 100)
        else:
            best_tip = sorted(
                tips, key=lambda x: (x[1], tip_priority.get(x[0], 0)), reverse=True
            )[0]

        # --- Get odds directly from related object ---
        odds_obj = getattr(m, "odds", None)
        odds_value = None
        if odds_obj:
            if best_tip[0] == "1":
                odds_value = odds_obj.home_win
            elif best_tip[0] == "2":
                odds_value = odds_obj.away_win
            elif best_tip[0] == "X":
                odds_value = odds_obj.draw
            elif best_tip[0] == "Over 2.5":
                odds_value = odds_obj.over_2_5
            elif best_tip[0] == "GG":
                odds_value = odds_obj.btts_yes

        date_str = m.match_date.strftime("%Y-%m-%d")
        pick = {
            "home_team": m.home_team,
            "away_team": m.away_team,
            "tip": best_tip[0],
            "confidence": int(best_tip[1]),
            "odds": odds_value,
            "match_date": date_str,
        }
        picks_by_date.setdefault(date_str, []).append(pick)

    # --- Sort and limit ---
    for date_str in picks_by_date:
        picks_by_date[date_str] = sorted(
            picks_by_date[date_str],
            key=lambda x: int(x["confidence"]),
            reverse=True,
        )[:limit]

    return picks_by_date
# -------------------------
# Helper: fetch actual results (existing)
# -------------------------
def fetch_actual_results(competition_code, match_date):
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        matches = response.json().get("matches", [])
        actual_results = []
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
        print(f"[ERROR] Fetch failed: {e}")
        return []


# -------------------------
# TRAINING helper (existing)
# -------------------------
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# -------------------------
# LIVE PREDICTIONS (updated to include odds)
# -------------------------
def live_predictions_by_date(request):
    predictions = []
    message = ""
    form = LivePredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        match_date = form.cleaned_data['match_date'].strftime("%Y-%m-%d")
        competition_code = form.cleaned_data['competition']

        matches = fetch_matches_by_date(API_KEY, competition_code, match_date)
        if not matches:
            message = "No matches found for selected date."
        else:
            try:
                train_df = fetch_training_data_all_seasons(competition_code)
            except Exception as e:
                return render(request, "predict/live_predictions.html", {
                    "form": form,
                    "message": f"[ERROR] Failed to fetch training data: {e}"
                })

            if train_df.empty:
                message = "No training data found for this competition."
            else:
                features = ['home_team', 'away_team']
                label_home = 'home_goals'
                label_away = 'away_goals'

                team_names = pd.concat([train_df['home_team'], train_df['away_team']]).unique()
                le = LabelEncoder()
                le.fit(team_names)

                for col in ['home_team', 'away_team']:
                    train_df[col] = le.transform(train_df[col])

                X = train_df[features]
                y_home = train_df[label_home]
                y_away = train_df[label_away]

                model_home = train_model(X, y_home)
                model_away = train_model(X, y_away)

                actual_results = fetch_actual_results(competition_code, match_date)
                actual_result_map = {
                    (res['home_team'], res['away_team']): res for res in actual_results
                }

                for match in matches:
                    home = match['homeTeam']['name']
                    away = match['awayTeam']['name']
                    match_id = match.get('id')

                    if home not in le.classes_ or away not in le.classes_:
                        print(f"[WARN] Skipping unknown teams: {home} or {away}")
                        continue

                    try:
                        input_df = pd.DataFrame({
                            'home_team': [home],
                            'away_team': [away]
                        })
                        input_df['home_team'] = le.transform(input_df['home_team'])
                        input_df['away_team'] = le.transform(input_df['away_team'])

                        pred_home = model_home.predict(input_df)[0]
                        pred_away = model_away.predict(input_df)[0]

                        result = actual_result_map.get((home, away))

                        # Fetch odds using The Odds API fallback
                        odds = fetch_odds_for_match(match, competition_code)

                        # Persist prediction + odds to DB
                        prediction, created = MatchPrediction.objects.update_or_create(
                            match_id=match_id or f"{home}-{away}-{match.get('utcDate','')}",
                            defaults={
                                'match_date': match['utcDate'][:10],
                                'competition': competition_code,
                                'home_team': home,
                                'away_team': away,
                                'predicted_home_goals': int(round(pred_home)),
                                'predicted_away_goals': int(round(pred_away)),
                            }
                        )

                        # ✅ Save odds in MatchOdds model
                        if odds:
                            MatchOdds.objects.update_or_create(
                                match=prediction,
                                defaults={
                                    "home_win": odds.get("home"),
                                    "draw": odds.get("draw"),
                                    "away_win": odds.get("away"),
                                    "over_2_5": odds.get("over25"),
                                    "under_2_5": odds.get("under25"),
                                    "btts_yes": odds.get("gg"),
                                    "btts_no": odds.get("ng"),
                                    "bookmaker": odds.get("bookmaker"),
                                }
                            )

                        

                        if result:
                            prediction.actual_home_goals = result['actual_home_goals']
                            prediction.actual_away_goals = result['actual_away_goals']
                            prediction.status = "FINISHED"
                            # mark accuracy
                            predicted_result = "Home" if pred_home > pred_away else ("Away" if pred_home < pred_away else "Draw")
                            prediction.is_accurate = (predicted_result == result["actual_result"])
                        prediction.save()
                        predictions.append(prediction)

                    except Exception as e:
                        print(f"[ERROR] Prediction failed for {home} vs {away}: {e}")
    else:
        form = LivePredictionForm()
    print(predictions)

    return render(request, "predict/live_predictions.html", {
        "form": form,
        "predictions": predictions,
        "message": message,
    })


# -------------------------
# UPDATED get_top_predictions (uses odds + confidence heuristics)
# -------------------------
from datetime import date
from .models import MatchPrediction
from .utils import get_team_metadata  # make sure this exists

# -------------------------
# store_top_pick_for_date (uses picks format above)
# -------------------------
def store_top_pick_for_date(predictions_by_date, specific_date=None):
    """
    Save top picks (from get_top_predictions-like structure) to TopPick DB.
    predictions_by_date: dict date_str -> list of picks
    """
    for date_str, picks in predictions_by_date.items():
        try:
            match_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue

        for p in picks:
            home_name = p["home_team"]
            away_name = p["away_team"]
            tip = p["tip"]
            confidence = p.get("confidence", 0)
            odds_val = p.get("odds", 0.0)

            TopPick.objects.update_or_create(
                match_date=match_date,
                home_team=home_name,
                away_team=away_name,
                defaults={
                    "tip": tip,
                    "confidence": confidence,
                    "odds": odds_val,
                    "actual_tip": None,
                    "is_correct": None,
                }
            )


# -------------------------
# Top picks view (updated to use our store/top logic)
# -------------------------
@require_GET
def top_picks_view(request):
    label_filter = request.GET.get("filter")
    match_date_str = request.GET.get("match_date")
    show_past = request.GET.get("past") == "1"

    today = date.today()

    if match_date_str:
        try:
            match_date = date.fromisoformat(match_date_str)
        except ValueError:
            match_date = today
    else:
        upcoming_dates = TopPick.objects.filter(match_date__gte=today).order_by("match_date").values_list("match_date", flat=True).distinct()
        match_date = upcoming_dates.first() if upcoming_dates.exists() else today

    # Fetch from DB
    if show_past:
        picks_qs = TopPick.objects.filter(match_date__lt=today).order_by("-match_date")
    else:
        picks_qs = TopPick.objects.filter(match_date=match_date)

    # Update actual results for picks if needed (existing util)
    update_actuals_for_top_picks(picks_qs)

    # Present picks (converted to dicts)
    picks = picks_qs.select_related("odds").values(
    "home_team",
    "away_team",
    "tip",
    "actual_tip",
    "is_correct",
    "confidence",
    "match_date",
    
    
)

    source = "cached"

    # If no picks stored, fallback to compute and store live picks
    if not picks and not show_past:
        predictions_by_date = get_top_predictions(limit=10)
        if match_date.strftime("%Y-%m-%d") in predictions_by_date:
            store_top_pick_for_date(predictions_by_date)
            picks_qs = TopPick.objects.filter(match_date=match_date)
            picks = list(picks_qs.values("home_team", "away_team", "tip", "actual_tip", "is_correct", "confidence", "odds", "match_date"))
            source = "live"

    if label_filter:
        picks = [p for p in picks if p.get("tip") == label_filter]

    return render(request, "predict/top_picks.html", {
        "prediction": picks,
        "filter_label": label_filter,
        "source": source,
        "selected_date": match_date,
        "show_past": show_past
    })


# -------------------------
# Remaining views (mostly unchanged) - results, training, admin dashboard, export, etc.
# -------------------------
@login_required
def admin_task_dashboard(request):
    tasks = PeriodicTask.objects.all()
    task_info = []
    for task in tasks:
        args = json.loads(task.args or "[]")
        kwargs = json.loads(task.kwargs or "{}")
        task_info.append({
            "name": task.name,
            "task": task.task,
            "enabled": task.enabled,
            "last_run_at": task.last_run_at,
            "interval": task.interval,
            "crontab": task.crontab,
            "args": args,
            "kwargs": kwargs,
            "last_triggered": task.date_changed,
        })

    cache_info = []
    for comp in COMPETITIONS:
        key = f"training_data_{comp}"
        df = cache.get(key)
        cache_info.append({
            "competition": comp,
            "cached": df is not None,
            "entries": len(df) if df is not None else 0
        })

    return render(request, "predict/admin_dashboard.html", {
        "tasks": task_info,
        "cache_info": cache_info,
        "competitions": COMPETITIONS
    })


@csrf_exempt
def trigger_task_now(request):
    if request.method == "POST":
        task_path = request.POST.get("task_path")
        try:
            current_app.send_task(task_path)
            return JsonResponse({"success": True, "message": f"{task_path} triggered successfully."})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    return JsonResponse({"success": False, "message": "Invalid request"})


@csrf_exempt
def refresh_cache_now(request):
    if request.method == "POST":
        comp = request.POST.get("competition")
        if comp:
            df = fetch_training_data_all_seasons(comp)
            if not df.empty:
                cache.set(f"training_data_{comp}", df, timeout=60 * 60 * 24 * 7)
                return JsonResponse({"success": True, "message": f"Cache refreshed for {comp}"})
    return JsonResponse({"success": False, "message": "Invalid request"})


@csrf_exempt
def clear_cache_now(request):
    if request.method == "POST":
        comp = request.POST.get("competition")
        if comp:
            cache.delete(f"training_data_{comp}")
            return JsonResponse({"success": True, "message": f"Cache cleared for {comp}"})
    return JsonResponse({"success": False, "message": "Invalid request"})


def results_view(request):
    matches = MatchPrediction.objects.filter(status="FINISHED").order_by('-match_date')
    for match in matches:
        match.correct = (match.predicted_home_goals is not None and match.predicted_away_goals is not None and
                         ((match.predicted_home_goals > match.predicted_away_goals and match.actual_home_goals > match.actual_away_goals) or
                          (match.predicted_home_goals < match.predicted_away_goals and match.actual_home_goals < match.actual_away_goals) or
                          (match.predicted_home_goals == match.predicted_away_goals and match.actual_home_goals == match.actual_away_goals)))
    return render(request, "predict/results.html", {"matches": matches})


def train_model_view(request):
    message = ""
    if request.method == "POST":
        competition_code = request.POST.get("competition")
        if not competition_code:
            message = "Please select a competition."
        else:
            seasons = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
            all_data = []
            for season in seasons:
                data = fetch_matches_by_season(API_KEY, competition_code, season)
                if data:
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df["home_team"] = df["homeTeam"].apply(lambda x: x["name"])
                        df["away_team"] = df["awayTeam"].apply(lambda x: x["name"])
                        df["home_goals"] = df["score"].apply(lambda x: x["fullTime"]["home"])
                        df["away_goals"] = df["score"].apply(lambda x: x["fullTime"]["away"])
                        all_data.append(df[["home_team", "away_team", "home_goals", "away_goals"]])

            if all_data:
                final_df = pd.concat(all_data, ignore_index=True)
                X, y_home, y_away, label_encoder = preprocess_api_data(final_df)
                model_dict = train_models(X, y_home, y_away)
                cache.set(f"{competition_code}_models", (model_dict, label_encoder), timeout=604800)
                message = f"Model trained and cached for {competitions.get(competition_code, competition_code)}."
            else:
                message = "No data available to train the model."

    return render(request, "predict/train_model.html", {
        "competitions": competitions,
        "message": message
    })


def cached_models_status(request):
    status = {}
    for code, name in competitions.items():
        key = f"{code}_models"
        status[name] = cache.get(key) is not None
    return JsonResponse(status)


def suggest_match_date(request):
    comp = request.GET.get("competition")
    date_str = request.GET.get("date")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", API_KEY)

    if not comp or not date_str:
        return JsonResponse({"error": "Missing parameters"}, status=400)

    next_date, matches = find_next_available_match_date(api_key, comp, date_str)
    return JsonResponse({
        "next_available_date": next_date,
        "match_count": len(matches),
    })


def view_predictions(request):
    competition = request.GET.get("competition")
    date_q = request.GET.get("date")
    predictions = MatchPrediction.objects.all()

    if competition:
        predictions = predictions.filter(competition=competition)

    if date_q:
        predictions = predictions.filter(match_date=date_q)

    return render(request, "predict/view_predictions.html", {
        "predictions": predictions,
        "competition": competition,
        "date": date_q
    })


def view_cache_status(request):
    cache_status = []
    for comp in COMPETITIONS:
        key = f"training_data_{comp}"
        df = cache.get(key)
        if df is not None:
            cache_status.append({
                "competition": comp,
                "cached": True,
                "entries": len(df)
            })
        else:
            cache_status.append({
                "competition": comp,
                "cached": False,
                "entries": 0
            })
    return render(request, "predict/cache_status.html", {"cache_status": cache_status})


def safe_logo_name(team_name):
    return quote(f"{team_name}.png")


def competition_logo(code):
    return static(f"logos/{code}.png")


TEAM_LOGO_DIR = os.path.join("static", "logos")
TEAM_LOGO_FILES = [f for f in os.listdir(TEAM_LOGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def fuzzy_match_logo(team_name):
    normalized_team = team_name.lower().replace("fc", "").replace(".", "").strip()
    name_map = {}
    for file in TEAM_LOGO_FILES:
        file_base = file.lower().replace("fc", "").replace(".", "").replace(".png", "").replace(".jpg", "").replace(".jpeg", "").strip()
        name_map[file_base] = file
    close_matches = difflib.get_close_matches(normalized_team, name_map.keys(), n=1, cutoff=0.6)
    if close_matches:
        matched_base = close_matches[0]
        return name_map[matched_base]
    return "default.png"


def get_team_metadata(name):
    return cache.get(f"team_meta::{name}", {"shortName": name, "crest": None})


NAME_TO_CODE = {v.lower(): k for k, v in competitions.items()}


def predictions_view(request):
    match_date = request.GET.get('match_date')
    predictions = MatchPrediction.objects.all().order_by('match_date')

    if match_date:
        predictions = predictions.filter(match_date=match_date)
    else:
        predictions = predictions.exclude(status="FINISHED")

    selected_code = request.GET.get("competition")
    if not selected_code:
        first_match = predictions.first()
        if first_match:
            selected_code = NAME_TO_CODE.get(first_match.competition.lower().strip(), "PL")
        else:
            selected_code = "PL"

    league_table = get_league_table(selected_code)
    top_predictions = get_top_predictions(limit=10)

    # Prefetch odds for all predictions in one query
    prediction_ids = [p.id for p in predictions]
    odds_map = {o.match_id: o for o in MatchOdds.objects.filter(match_id__in=prediction_ids)}
    display_data = []
    for p in predictions:
        meta_home = get_team_metadata(p.home_team)
        meta_away = get_team_metadata(p.away_team)

        comp_code = NAME_TO_CODE.get(p.competition.lower().strip(), "default")
        competition_logo_path = static(f"logos/{p.competition}.png")

        actual_result = "-:-"
        actual_winner = None
        if p.status == "FINISHED" and p.actual_home_goals is not None:
            actual_result = f"{p.actual_home_goals} - {p.actual_away_goals}"
            if p.actual_home_goals > p.actual_away_goals:
                actual_winner = "1"
            elif p.actual_home_goals < p.actual_away_goals:
                actual_winner = "2"
            else:
                actual_winner = "X"

        if (p.predicted_home_goals or 0) > (p.predicted_away_goals or 0):
            winner = "1"
        elif (p.predicted_home_goals or 0) < (p.predicted_away_goals or 0):
            winner = "2"
        else:
            winner = "X"

        odds = odds_map.get(getattr(p, "id", None), None)

        display_data.append({
            "home_team": meta_home.get("shortName", p.home_team),
            "away_team": meta_away.get("shortName", p.away_team),
            "predicted_home_goals": p.predicted_home_goals,
            "predicted_away_goals": p.predicted_away_goals,
            "match_date": p.match_date.strftime("%Y-%m-%d"),
            "match_time": p.match_date.strftime("%H:%M") if isinstance(p.match_date, datetime) else "",
            "competition": p.competition,
            "competition_code": comp_code,
            "status": p.status,
            "actual_home_goals": p.actual_home_goals,
            "actual_away_goals": p.actual_away_goals,
            "actual_result": actual_result,
            "home_logo": meta_home.get("crest", static("logos/default.png")),
            "away_logo": meta_away.get("crest", static("logos/default.png")),
            "competition_logo": competition_logo_path,
            "winner": winner,
            "actual_winner": actual_winner if p.status == "FINISHED" else None,
            "odds_home": getattr(p, "odds_home", None),
            "odds_draw": getattr(p, "odds_draw", None),
            "odds_away": getattr(p, "odds_away", None),
            "odds_gg": getattr(p, "odds_gg", None),
            "odds_over_25": getattr(p, "odds_over_25", None),
            "odds": odds,
        })

    for row in league_table:
        team_name = row["team"]["name"]
        meta = cache.get(f"team_meta::{team_name}", {})
        row["team"]["shortName"] = meta.get("shortName", team_name)
        row["team"]["crest"] = meta.get("crest", static("logos/default.png"))

    paginator = Paginator(display_data, 10)
    page_number = request.GET.get("page")
    paginated_predictions = paginator.get_page(page_number)
    return render(request, "predict/predictions_view.html", {
        "predictions": paginated_predictions,
        "league_table": league_table,
        "competitions": competitions,
        "selected_competition": selected_code,
        "selected_date": match_date,
        "page_obj": paginated_predictions,
        "top_predictions": top_predictions,
    })


# AJAX league table view (unchanged)
def ajax_league_table(request):
    comp = request.GET.get("competition", "PL")
    table = get_league_table(comp)
    for row in table:
        team = row.get("team", {})
        name = team.get("name", "")
        meta = cache.get(f"team_meta::{name}", {})
        team["shortName"] = meta.get("shortName") or team.get("shortName") or name
        team["crest"] = meta.get("crest") or team.get("crest") or static("logos/default.png")
    html = render_to_string("partials/league_table.html", {"league_table": table})
    return JsonResponse({"html": html})


def team_logos_preview(request):
    grouped_teams = defaultdict(list)

    # NOTE: cache.iter_keys may not exist depending on your cache backend
    # this portion retains earlier logic but might need adaptation for your cache
    for comp_code in COMPETITIONS:
        # scan keys in cache is backend-dependent; keep simple: attempt to load from known teams
        pass

    # Fallback simple preview (if cache keys scanning not available)
    preview_data = []
    for comp_code, comp_name in competitions.items():
        preview_data.append({
            "competition": comp_name,
            "competition_code": comp_code,
            "logo": static(f"logos/{comp_code}.png"),
            "teams": []
        })

    return render(request, "predict/team_logos_preview.html", {
        "preview_data": preview_data
    })


def match_team_logo(team_name):
    simplified_team_names = [f.lower().replace('.png', '').replace('.jpg', '').replace('.jpeg', '') for f in TEAM_LOGO_FILES]
    match = difflib.get_close_matches(team_name.lower(), simplified_team_names, n=1, cutoff=0.6)
    if match:
        for f in TEAM_LOGO_FILES:
            if match[0] in f.lower():
                return f
    return "default.png"


def flatten_competitions(comp_dict):
    return {code: name for region in comp_dict.values() for code, name in region.items()}


def league_table_view(request, competition_code):
    table = get_league_table(competition_code)
    for team in table:
        team_name = team["team"]["name"]
        team["team"]["logo"] = match_team_logo(team_name)

    competition_name = flatten_competitions(COMPETITIONS).get(competition_code, competition_code)

    return render(request, "predict/league_table.html", {
        "table": table,
        "competition_code": competition_code,
        "competition_name": competition_name,
        "competitions_grouped": COMPETITIONS,  # For dropdown with regions
        "competition_logo": static(f"logos/{competition_code}.png")
    })


@csrf_exempt
def refresh_league_table_cache(request):
    if request.method == "POST":
        comp = request.POST.get("competition")
        if comp:
            table = get_league_table(comp)  # Forces fresh fetch and recache
            return JsonResponse({"success": True, "message": f"Refreshed {comp}"})
    return JsonResponse({"success": False, "message": "Invalid request"})


def actual_results_view(request):
    form = ActualResultForm(request.GET or None)
    results = []

    if form.is_valid():
        comp = form.cleaned_data['competition']
        match_date_str = form.cleaned_data['match_date'].strftime('%Y-%m-%d')

        results = fetch_actual_results(comp, match_date_str)
        updated_count = 0

        for result in results:
            home = result["home_team"]
            away = result["away_team"]

            prediction = MatchPrediction.objects.filter(
                competition=comp,
                home_team=home,
                away_team=away,
                match_date=form.cleaned_data['match_date']
            ).first()

            if prediction:
                prediction.actual_home_goals = result["actual_home_goals"]
                prediction.actual_away_goals = result["actual_away_goals"]
                prediction.actual_result = result["actual_result"]
                prediction.actual_score = f"{result['actual_home_goals']} - {result['actual_away_goals']}"
                prediction.status = "FINISHED"
                prediction.save()
                updated_count += 1

        if updated_count:
            messages.success(request, f"{updated_count} match result(s) updated successfully.")
        else:
            messages.warning(request, "No matching predictions found to update.")

    return render(request, "predict/actual_results.html", {
        "form": form,
        "results": results,
    })


def refresh_top_picks(request):
    today = date.today()
    top_predictions = get_top_predictions(limit=10)
    store_top_pick_for_date(top_predictions)
    return redirect("top-picks_view")


def export_top_picks(request, format):
    match_date_str = request.GET.get("match_date")

    try:
        try:
            match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        except ValueError:
            match_date = datetime.strptime(match_date_str, "%B %d, %Y").date()
    except Exception as e:
        return HttpResponseBadRequest(f"Invalid date format: {e}")

    picks = TopPick.objects.filter(match_date=match_date)

    if format == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="top_picks_{match_date}.csv"'
        writer = csv.writer(response)
        writer.writerow(["Match Date", "Home", "Away", "Tip", "Confidence", "Odds", "Actual Tip", "Correct?"])
        for p in picks:
            writer.writerow([p.match_date, p.home_team, p.away_team, p.tip, p.confidence, getattr(p, "odds", ""), p.actual_tip, p.is_correct])
        return response

    elif format == "pdf":
        response = HttpResponse(content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="top_picks_{match_date}.pdf"'
        from reportlab.pdfgen import canvas
        p = canvas.Canvas(response)
        y = 800
        p.drawString(100, y, f"Top Picks - {match_date}")
        y -= 30
        for pick in picks:
            p.drawString(100, y, f"{pick.home_team} vs {pick.away_team} - Tip: {pick.tip} - Confidence: {pick.confidence}% - Odds: {getattr(p,'odds','')}")
            y -= 20
        p.showPage()
        p.save()
        return response

    else:
        return HttpResponse("Invalid format", status=400)


def backfill_viewids():
    matches = MatchPrediction.objects.all()
    for match in matches:
        if not getattr(match, "match_id", None):
            composite_id = f"{match.home_team}-{match.away_team}-{match.match_date}"
            # try to save to field name available (match.match_id or match.matchid)
            if hasattr(match, "match_id"):
                match.match_id = composite_id
            elif hasattr(match, "matchid"):
                match.matchid = composite_id
            match.save()
    print("Backfilling complete.")



from django.views.decorators.http import require_GET
from django.http import JsonResponse
from .models import MatchPrediction, TopPick

@require_GET
def api_predictions(request):
    competition = request.GET.get("competition")
    date_q = request.GET.get("date")

    predictions = MatchPrediction.objects.all()

    if competition:
        predictions = predictions.filter(competition=competition)

    if date_q:
        predictions = predictions.filter(match_date=date_q)

    data = [
        {
            "id": p.id,
            "competition": p.competition,
            "match_date": str(p.match_date),
            "home_team": p.home_team,
            "away_team": p.away_team,
            "predicted_home_goals": p.predicted_home_goals,
            "predicted_away_goals": p.predicted_away_goals,
            "actual_home_goals": p.actual_home_goals,
            "actual_away_goals": p.actual_away_goals,
            "status": p.status,
        }
        for p in predictions
    ]
    return JsonResponse({"predictions": data})

@require_GET
def api_top_picks(request):
    picks = TopPick.objects.all().order_by("-match_date")[:20]
    data = [
        {
            "home_team": p.home_team,
            "away_team": p.away_team,
            "tip": p.tip,
            "confidence": p.confidence,
            "odds": p.odds,
            "match_date": str(p.match_date),
            "is_correct": p.is_correct,
        }
        for p in picks
    ]
    return JsonResponse({"top_picks": data})

def league_table_api(request, competition_code):
    table = cache.get(f"league_table_{competition_code}", [])
    return JsonResponse({"table": table})
