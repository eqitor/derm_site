from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from datetime import timedelta, datetime
from PIL import Image
import numpy as np


class Examination(models.Model):
    user_field = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='images/')
    created = models.DateTimeField(blank=True, default=datetime.now)
    diagnose = models.CharField(max_length=10, null=True)

    def __str__(self):
        return str(self.id)

