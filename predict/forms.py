# predict/forms.py

from django import forms
from datetime import date

from predict.constants import COMPETITIONS

COMPETITION_CHOICES = [
    ('PL', 'Premier League'),
    ('PD', 'La Liga'),
    ('SA', 'Serie A'),
    ('BL1', 'Bundesliga'),
    ('FL1', 'Ligue 1'),
    ('BSA', 'Brasileirão Série A'),
    ('DED', 'Eredivisie'),
    ('PPL', 'Primeira Liga'),
    ('ELC', 'Championship'),
    ('CL', 'UEFA Champions League'),
    ('EC', 'European Championship'),
    ('CLI', 'Copa Libertadores'),
    ('WC', 'FIFA World Cup'),


]

class PredictionForm(forms.Form):
    match_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        initial=date.today,
        label="Match Date"
    )
    competition = forms.ChoiceField(
        choices=COMPETITION_CHOICES,
        label="Competition"
    )


from django import forms

class LivePredictionForm(forms.Form):
    match_date = forms.DateField(widget=forms.DateInput(attrs={"type": "date"}))
    competition = forms.ChoiceField(choices=[
        ("PL", "Premier League"),
        ("SA", "Serie A"),
        ("PD", "La Liga"),
        ("BL1", "Bundesliga"),
        ("FL1", "Ligue 1"),
        ("DED", "Eredivisie"),
        ("PPL", "Primeira Liga"),
        ("ELC", "Championship"),
        ("BSA", "Brazil Série A"),
        ("CL", "UEFA Champions League"),
        ("EC", "European Championship"),
        ("CLI", "Copa Libertadores"),
        ("WC", "FIFA World Cup"),
        
    ])

class ActualResultForm(forms.Form):
    match_date = forms.DateField(initial=date.today, widget=forms.DateInput(attrs={'type': 'date'}))
    competition = forms.ChoiceField(choices=[(k, v) for k, v in COMPETITIONS.items()])