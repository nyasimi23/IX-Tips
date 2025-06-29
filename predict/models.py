from django.db import models

class MatchPrediction(models.Model):
    match_id = models.IntegerField(default=0)
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
    predicted_home_goals = models.IntegerField(null=True, blank=True)
    predicted_away_goals = models.IntegerField(null=True, blank=True)
    actual_home_goals = models.IntegerField(null=True, blank=True)
    actual_away_goals = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.competition}: {self.home_team} vs {self.away_team} on {self.match_date}"

class TopPick(models.Model):
    match_date = models.DateField()
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    tip = models.CharField(max_length=50)  # e.g. '1', '2', 'X', 'Over 2.5', 'GG'
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    actual_tip = models.CharField(max_length=50, blank=True, null=True)  # new
    is_correct = models.BooleanField(null=True,blank=True) 

    class Meta:
        unique_together = ('match_date', 'home_team', 'away_team')
        ordering = ['match_date']

    def __str__(self):
        return f"{self.match_date} | {self.home_team} vs {self.away_team} - {self.tip} ({self.confidence}%)"