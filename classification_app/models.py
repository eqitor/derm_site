from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

from datetime import timedelta, datetime
from PIL import Image
import numpy as np


def update_filename(instance, filename):
    """Function specifies method of saving uploaded image to file and database"""
    return f'temp_images/{instance.id}/{filename}'


class Examination(models.Model):
    """Model of examination. It's used to store temporary images of classification function"""
    user_field = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to=update_filename)
    created = models.DateTimeField(blank=True, default=datetime.now)
    diagnose = models.CharField(max_length=10, null=True)

    def __str__(self):
        return str(self.id)
