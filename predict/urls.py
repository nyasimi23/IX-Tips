from django.urls import path
from . import views

urlpatterns = [
    path('matchday_predictions/', views.matchday_predictions, name='matchday_predictions'),
]
