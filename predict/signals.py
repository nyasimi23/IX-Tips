from django_celery_beat.models import PeriodicTask, CrontabSchedule
from django.utils.timezone import now
import json
from django.db.models.signals import post_migrate
from django.dispatch import receiver

def setup_scheduled_tasks():
    # 1. Weekly Prediction Task — Tuesday at 9:00 AM
    t1_schedule, _ = CrontabSchedule.objects.get_or_create(
        minute='0',
        hour='9',
        day_of_week='2',
        day_of_month='*',
        month_of_year='*',
        timezone='Africa/Nairobi'
    )
    PeriodicTask.objects.get_or_create(
        crontab=t1_schedule,
        name='Weekly Match Predictions',
        task='predict.tasks.update_predictions_and_cache',
        defaults={'start_time': now()}
    )

    # 2. Daily Results Update — Every day at 11:00 PM
    t2_schedule, _ = CrontabSchedule.objects.get_or_create(
        minute='0',
        hour='23',
        day_of_week='*',
        day_of_month='*',
        month_of_year='*',
        timezone='Africa/Nairobi'
    )
    PeriodicTask.objects.get_or_create(
        crontab=t2_schedule,
        name='Daily Results Update',
        task='predict.tasks.update_actual_results',
        defaults={'start_time': now()}
    )
@receiver(post_migrate)
def create_periodic_tasks(sender, **kwargs):
    # Avoid running when migrating unrelated apps
    if sender.name != 'predict':
        return

    try:
        setup_scheduled_tasks()
    except Exception as e:
        print(f"Error setting up scheduled tasks: {e}")
