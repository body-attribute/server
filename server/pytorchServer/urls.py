from django.urls import path

from . import views

urlpatterns = [
    path('', views.predict_index, name='predict_index')
]
