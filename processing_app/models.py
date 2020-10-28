from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from datetime import timedelta, datetime
from PIL import Image
import numpy as np

def update_filename(instance, filename):
    return f'images/{instance.id}/{filename}'


class ImageProc(models.Model):
    id = models.AutoField(primary_key=True)
    user_field = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    created = models.DateTimeField(blank=True, default=datetime.now)

    image = models.ImageField(upload_to=update_filename)
    image_clahe = models.ImageField(upload_to=update_filename)
    image_axes = models.ImageField(upload_to=update_filename)
    image_contours = models.ImageField(upload_to=update_filename)
    image_illumination_removed = models.ImageField(upload_to=update_filename)

    # colors of image
    white_color = models.FloatField(null=True, default=-1)
    red_color = models.FloatField(null=True, default=-1)
    light_brown_color = models.FloatField(null=True, default=-1)
    dark_brown_color = models.FloatField(null=True, default=-1)
    blue_gray_color = models.FloatField(null=True, default=-1)
    black_color = models.FloatField(null=True, default=-1)

    def __str__(self):
        return str(self.id)




