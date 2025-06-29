from django.contrib import admin
from .models import TopPick

@admin.register(TopPick)
class TopPickAdmin(admin.ModelAdmin):
    list_display = ("match_date", "home_team", "away_team", "tip", "confidence", "created_at")
    list_filter = ("match_date", "tip")
    search_fields = ("home_team", "away_team")

# Register your models here.
