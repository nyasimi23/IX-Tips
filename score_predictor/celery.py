import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'score_predictor.settings')

app = Celery("score_predictor")
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Consolidated CELERY_BEAT_SCHEDULE
app.conf.beat_schedule = {
    'cache-models-every-tuesday': {
        'task': 'predict.tasks.train_and_cache_models',
        'schedule': crontab(hour=1, minute=0, day_of_week='tue'),  # 1:00 AM
    },
    'predict-weekly-matches': {
        'task': 'predict.tasks.update_predictions_and_cache',
        'schedule': crontab(hour=2, minute=0, day_of_week='tue'),  # 2:00 AM
    },
    'run-weekly-predictions-every-tuesday': {
        'task': 'predict.tasks.weekly_predict_and_store_matches',
        'schedule': crontab(hour=6, minute=0, day_of_week='tue'),  # 6:00 AM
    },
    'predict-next-matches-weekly': {
        'task': 'predict.tasks.predict_next_fixtures',
        'schedule': crontab(hour=4, minute=0, day_of_week='tue'),  # 4:00 AM
    },
    'run-staggered-predictions-every-tuesday': {
        'task': 'predict.tasks.trigger_staggered_scheduling',
        'schedule': crontab(hour=20, minute=26, day_of_week=4),  # Adjust as needed
    },
    'cache-training-data-weekly': {
        'task': 'predict.tasks.cache_training_data',
        'schedule': crontab(hour=17, minute=27, day_of_week=4),  
    },
    
    'refresh-league-standings': {
        'task': 'predict.tasks.refresh_all_league_tables',
        'schedule': crontab(minute=1,),
    },
    "update_metadata_hourly": {
        "task": "predict.tasks.update_metadata_task",
        "schedule": crontab(minute=0, hour='*'),  # every hour
    },
    
    "update-match-status-and-results-daily": {
        "task": "predict.tasks.update_match_status_and_results",
        "schedule": crontab(hour=3, minute=0),  # runs daily at 03:00 AM
    },
}
