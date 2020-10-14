from django.db import models
from datetime import timedelta, datetime


class Examination(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='images/')