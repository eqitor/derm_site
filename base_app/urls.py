from django.urls import path
from . import views

app_name = 'base_app'
urlpatterns = [
    path('', views.mainpage, name='mainpage'),
    path('restricted', views.restricted, name='restricted'),
]