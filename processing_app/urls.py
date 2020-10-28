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
    # path('save', save, name='save'),
    path('remove_illumination', remove_illumination, name='remove_illumination'),
    path('contour_image', contour_image, name='contour_image'),
    path('colour_params', colour_params, name='colour_params'),
    path('axes_image', axes_image, name='axes_image'),
    path('results', results, name='results'),
]

if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
