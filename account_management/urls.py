from django.urls import path
from . import views

app_name = 'account_management'
urlpatterns = [
    path('login_page/', views.login_page, name='login_page'),
    path('logout_user/', views.logout_user, name='logout_user'),
    path('register_page/', views.register_page, name='register_page'),
    path('processing_history/', views.processing_history, name='processing_history')
]