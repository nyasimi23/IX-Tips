from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Run weekly predictions'

    def handle(self, *args, **kwargs):
        from predict.tasks import update_predictions_and_cache
        update_predictions_and_cache.delay()
