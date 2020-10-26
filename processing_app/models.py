from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from datetime import timedelta, datetime
from PIL import Image
import numpy as np


class ProcessedImage(models.Model):
    user_field = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='images/')
    created = models.DateTimeField(blank=True, default=datetime.now)

    def __str__(self):
        return str(self.id)


class TempImage(models.Model):
    image = models.ImageField(upload_to='images/temp/')
    created = models.DateTimeField(blank=True, default=datetime.now)

    def __str__(self):
        return "Temp " + str(self.id)




