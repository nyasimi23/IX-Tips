from django.db import models

class MatchPrediction(models.Model):
    competition = models.CharField(max_length=100)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    match_date = models.DateField()
    predicted_result = models.CharField(max_length=10, null=True, blank=True)
    predicted_score = models.CharField(max_length=10, null=True, blank=True)
    actual_result = models.CharField(max_length=10, null=True, blank=True)
    actual_score = models.CharField(max_length=10, null=True, blank=True)
    gg = models.CharField(max_length=3, null=True, blank=True)  # Both teams scored
    agg = models.CharField(max_length=3, null=True, blank=True)  # Aggregate goals condition
    ov = models.CharField(max_length=10, null=True, blank=True)  # Over/Under 1.5 goals
    average_goals_category = models.CharField(max_length=10, null=True, blank=True)  # Predicted goals category
    status = models.CharField(max_length=20, null=True, blank=True)  # Match status (e.g., SCHEDULED, FINISHED)

    def __str__(self):
        return f"{self.competition}: {self.home_team} vs {self.away_team} on {self.match_date}"
