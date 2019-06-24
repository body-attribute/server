from django.urls import path

from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict_index'),
    path('show/', views.show)
]
