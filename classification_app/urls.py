from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *


app_name = 'classification_app'

urlpatterns = [
    path('image_upload', examination_image_view, name='image_upload'),
    path('success', success, name='success'),
    path('examination_image_for_processing', examination_image_for_processing, name='examination_image_for_processing'),
    path('success_for_processing', success_for_processing, name='success_for_processing'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
