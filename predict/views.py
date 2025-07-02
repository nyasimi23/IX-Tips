import csv
import difflib
import os
import re
import json
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict
from django.core.paginator import Paginator
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.contrib import messages
from django.core.cache import cache
from django.templatetags.static import static
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from urllib.parse import quote 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from celery import current_app
from django_celery_beat.models import PeriodicTask

from .models import MatchPrediction, TopPick
from .forms import ActualResultForm, PredictionForm, LivePredictionForm
from .utils import fetch_matches_by_date, get_top_predictions, predict_match_outcome, preprocess_api_data, store_top_pick_for_date, train_models, get_league_table, fetch_training_data
from .generate_logo_mapping import TEAM_LOGOS
from .utils import COMPETITIONS


API_KEY = "7419be10abd14d7fb752e6fe6491e38f"

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



# View to display cached training status and tasks
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
    # Fetch only matches that are finished (i.e., have actual results)
    matches = MatchPrediction.objects.filter(status="FINISHED").order_by('-match_date')

    # Prepare comparison data
    for match in matches:
        match.correct = match.predicted_result == match.actual_result

    return render(request, "predict/results.html", {"matches": matches})
from .utils import fetch_matches_by_season, fetch_training_data_all_seasons, find_next_available_match_date, find_next_match_date, get_league_table, make_predictions, preprocess_match_data, process_match_data  # Import this at top



def train_model_view(request):
    message = ""
    if request.method == "POST":
        competition_code = request.POST.get("competition")
        if not competition_code:
            message = "Please select a competition."
        else:
            seasons = [2019, 2020, 2021, 2022, 2023, 2024,2025]
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
    date = request.GET.get("date")
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "7419be10abd14d7fb752e6fe6491e38f")

    if not comp or not date:
        return JsonResponse({"error": "Missing parameters"}, status=400)

    next_date, matches = find_next_available_match_date(api_key, comp, date)
    return JsonResponse({
        "next_available_date": next_date,
        "match_count": len(matches),
    })

# views.py


# Utility function to fetch actual match results by competition and match date

import requests

BASE_URL = "https://api.football-data.org/v4"
 # 🔒 Ideally move this to settings or .env

def fetch_actual_results(competition_code, match_date):
    """
    Fetches actual results of matches for a specific competition and date.

    Parameters:
    - competition_code: The competition code (e.g., 'PL' for Premier League).
    - match_date: The date of the matches in 'YYYY-MM-DD' format.

    Returns:
    - A list of dictionaries containing match results, including teams, scores, and status.
    """
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

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
    
BASE_URL = "https://api.football-data.org/v4"


def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def live_predictions_by_date(request):
    predictions = []
    message = ""
    if request.method == "POST":
        form = LivePredictionForm(request.POST)
        if form.is_valid():
            match_date = form.cleaned_data['match_date'].strftime("%Y-%m-%d")
            competition_code = form.cleaned_data['competition']
            #status = "FINISHED" if match_date < datetime.today().date() else "TIMED"

            matches = fetch_matches_by_date(API_KEY,competition_code, match_date)
            if not matches:
                message = "No matches found for selected date."
            else:
                print(f"[DEBUG] {len(matches)} matches found on {match_date} for {competition_code}")

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
                        match_id = match['id']
                        

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
                            

                            prediction, created = MatchPrediction.objects.update_or_create(
                                match_id=match_id,
                                defaults={
                                    'match_date': match['utcDate'][:10],
                                    'competition': competition_code,
                                    'home_team': home,
                                    'away_team': away,
                                    'predicted_home_goals': int(pred_home),
                                    'predicted_away_goals': int(pred_away),
                                    #'status':status
                                    
                                }
                            )

                            if result:
                                prediction.actual_home_goals = result['home_goals']
                                prediction.actual_away_goals = result['away_goals']

                                # Evaluate prediction accuracy
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


def view_predictions(request):
    competition = request.GET.get("competition")
    date = request.GET.get("date")
    predictions = MatchPrediction.objects.all()

    if competition:
        predictions = predictions.filter(competition=competition)

    if date:
        predictions = predictions.filter(match_date=date)

    return render(request, "predict/view_predictions.html", {
        "predictions": predictions,
        "competition": competition,
        "date": date
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
    return quote(f"{team_name}.png")  # Encodes spaces and special characters

def competition_logo(code):
    return static(f"logos/{code}.png")

TEAM_LOGO_DIR = os.path.join("static", "logos")
TEAM_LOGO_FILES = [f for f in os.listdir(TEAM_LOGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def fuzzy_match_logo(team_name):
    # Normalize team name
    normalized_team = team_name.lower().replace("fc", "").replace(".", "").strip()

    # Prepare list of normalized filenames (without extension)
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
from datetime import datetime
def predictions_view(request):
    match_date = request.GET.get('match_date')
    predictions = MatchPrediction.objects.all().order_by('match_date')
    if match_date:
        predictions = predictions.filter(match_date=match_date)
    else:
        # ✅ Only show upcoming (non-finished) predictions by default
        predictions = predictions.exclude(status="FINISHED")
    # Determine selected competition from GET or fallback to first match
    selected_code = request.GET.get("competition")
    if not selected_code:
        first_match = predictions.first()
        if first_match:
            selected_code = NAME_TO_CODE.get(first_match.competition.lower().strip(), "PL")
        else:
            selected_code = "PL"

    # Fetch league table
    league_table = get_league_table(selected_code)
    top_predictions = get_top_predictions(limit=10)
    
    display_data = []
    for p in predictions:
        meta_home = get_team_metadata(p.home_team)
        meta_away = get_team_metadata(p.away_team)

        comp_code = NAME_TO_CODE.get(p.competition.lower().strip(), "default")
        comp_meta = cache.get(f"competition_meta::{comp_code}", {})
        competition_logo = comp_meta.get("emblem") or static(f"logos/{comp_code}.png")

        comp_meta = cache.get(f"competition_meta::{comp_code}", {})
        
        competition_logo = static(f"logos/{p.competition}.png")

        
        actual_result = "-:-"
        if p.status == "FINISHED" and p.actual_home_goals is not None:
            actual_result = f"{p.actual_home_goals} - {p.actual_away_goals}"
            if p.actual_home_goals > p.actual_away_goals:
                actual_winner = "1"
            elif p.actual_home_goals < p.actual_away_goals:
                actual_winner = "2"
            else:
                actual_winner = "X"

        if p.predicted_home_goals > p.predicted_away_goals:
            winner = "1"
        elif p.predicted_home_goals < p.predicted_away_goals:
            winner = "2"
        else:
            winner = "X"



        display_data.append({
            "home_team": meta_home.get("shortName", p.home_team),
            "away_team": meta_away.get("shortName", p.away_team),
            "predicted_home_goals": p.predicted_home_goals,
            "predicted_away_goals": p.predicted_away_goals,
            "match_date": p.match_date.strftime("%Y-%m-%d"),
            "match_time": p.match_date.strftime("%H:%M"),
            "competition": p.competition,
            "competition_code": comp_code,
            "status": p.status,
            "actual_home_goals": p.actual_home_goals,
            "actual_away_goals": p.actual_away_goals,

            "actual_result": actual_result,
            "home_logo": meta_home.get("crest", static("logos/default.png")),
            "away_logo": meta_away.get("crest", static("logos/default.png")),
            "competition_logo": competition_logo,
            "winner": winner,
            "actual_winner": actual_winner if p.status == "FINISHED" else None,
        })
        
      
    # Enhance league table with shortName + crest
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


# AJAX view
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

    for key in cache.iter_keys("team_meta::*"):
        meta = cache.get(key)
        if not meta:
            continue

        team_name = key.split("::")[1]
        competition = meta.get("competition", "Other")
        short = meta.get("shortName", team_name)
        crest = meta.get("crest", static("logos/default.png"))

        grouped_teams[competition].append({
            "team": team_name,
            "short": short,
            "crest": crest
        })

    # Prepare final structured data
    preview_data = []
    for comp_code, teams in grouped_teams.items():
        comp_name = COMPETITIONS.get(comp_code, comp_code)
        comp_logo = static(f"logos/{comp_code}.png")  # or fuzzy if needed
        preview_data.append({
            "competition": comp_name,
            "competition_code": comp_code,
            "logo": comp_logo,
            "teams": teams
        })

    return render(request, "predict/team_logos_preview.html", {
        "preview_data": preview_data
    })

TEAM_LOGO_DIR = os.path.join("static", "logos")
TEAM_LOGO_FILES = [f for f in os.listdir(TEAM_LOGO_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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
from django.utils.timezone import localtime
@csrf_exempt
def refresh_league_table_cache(request):
    last_updated = cache.get(f"standings_{selected}_updated")
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
        match_date = form.cleaned_data['match_date'].strftime('%Y-%m-%d')
        results = fetch_actual_results(comp, match_date)

    return render(request, "predict/actual_results.html", {
        "form": form,
        "results": results,
    })
from django.template.loader import render_to_string
from django.http import JsonResponse
from datetime import date
from django.views.decorators.http import require_GET

from .models import TopPick, MatchPrediction
from .utils import get_top_predictions
from django.db import transaction
from itertools import chain
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
        # Default to today or the next upcoming TopPick date
        upcoming_dates = TopPick.objects.filter(match_date__gte=today).order_by("match_date").values_list("match_date", flat=True).distinct()
        match_date = upcoming_dates.first() if upcoming_dates.exists() else today

    # Fetch from DB
    if show_past:
        picks_qs = TopPick.objects.filter(match_date__lt=today).order_by("-match_date")
    else:
        picks_qs = TopPick.objects.filter(match_date=match_date)

    picks = list(picks_qs.values("home_team", "away_team", "tip", "actual_tip", "is_correct", "confidence", "match_date"))

    source = "cached"

    # If no picks stored, fallback to live predictions
    if not picks and not show_past:
        predictions_by_date = get_top_predictions(limit=10)

        if match_date.strftime("%Y-%m-%d") in predictions_by_date:
            store_top_pick_for_date(predictions_by_date)  # stores to DB
            picks = predictions_by_date[match_date.strftime("%Y-%m-%d")]
            source = "live"

    # Apply tip filter (1, 2, GG, Over 2.5 etc.)
    if label_filter:
        picks = [p for p in picks if p.get("tip") == label_filter]

    return render(request, "predict/top_picks.html", {
        "prediction": picks,
        "filter_label": label_filter,
        "source": source,
        "selected_date": match_date,
        "show_past": show_past
    })
def refresh_top_picks(request):
    today = date.today()
    top_predictions = get_top_predictions(limit=10)
    store_top_pick_for_date(top_predictions, today)
    return redirect("top-picks_view")

from reportlab.pdfgen import canvas

def export_top_picks(request, format):
    
    match_date_str = request.GET.get("match_date")
    
    try:
        # Handle both ISO and human-readable formats
        try:
            match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        except ValueError:
            match_date = datetime.strptime(match_date_str, "%B %d, %Y").date()  # e.g. "July 12, 2025"
    except Exception as e:
        return HttpResponseBadRequest(f"Invalid date format: {e}")
    picks = TopPick.objects.filter(match_date=match_date)

    if format == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="top_picks_{match_date}.csv"'
        writer = csv.writer(response)
        writer.writerow(["Match Date", "Home", "Away", "Tip", "Confidence", "Actual Tip", "Correct?"])
        for p in picks:
            writer.writerow([p.match_date, p.home_team, p.away_team, p.tip, p.confidence, p.actual_tip, p.is_correct])
        return response

    elif format == "pdf":
        response = HttpResponse(content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="top_picks_{match_date}.pdf"'
        p = canvas.Canvas(response)
        y = 800
        p.drawString(100, y, f"Top Picks - {match_date}")
        y -= 30
        for pick in picks:
            p.drawString(100, y, f"{pick.home_team} vs {pick.away_team} - Tip: {pick.tip} - Confidence: {pick.confidence}%")
            y -= 20
        p.showPage()
        p.save()
        return response

    else:
        return HttpResponse("Invalid format", status=400)