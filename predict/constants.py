# Competition codes and names (adjust as needed)
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

# Reverse mapping for easy lookup
NAME_TO_CODE = {v.lower(): k for k, v in COMPETITIONS.items()}

# Football-Data.org API Token
API_TOKEN = "your_api_token_here"  # Replace with your actual token or load from env

# Cache expiration (in seconds)
TEAM_METADATA_CACHE_TIMEOUT = 60 * 60 * 24 * 30  # 30 days
STANDINGS_CACHE_TIMEOUT = 60 * 60 * 6  # 6 hours

# Cache keys (optional for centralization)
def team_meta_cache_key(team_name):
    return f"team_meta::{team_name}"

def competition_cached_key(comp_code):
    return f"competition_cached::{comp_code}"

def standings_cache_key(comp_code):
    return f"standings_{comp_code}"

def training_data_cache_key(comp_code):
    return f"training_data_{comp_code}"

def get_team_metadata(name):
    return cache.get(f"team_meta::{name}", {"shortName": name, "crest": None})

from django.core.cache import cache