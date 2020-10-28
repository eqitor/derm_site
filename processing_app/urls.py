from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from .views import *


app_name = 'processing_app'

urlpatterns = [
    path('image_upload', processing_image_upload_view, name='processing_image_upload'),
    path('image_processing', processing, name='processing'),
    path('results', results, name='results'),
]

if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
