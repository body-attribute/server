from django.urls import path

from . import views

urlpatterns = [
    path('predict/', views.predict_index, name='predict_index'),
    path('post/', views.Post.as_view()),
    path('show/', views.show)
]
