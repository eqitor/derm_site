from django.db import models
from datetime import timedelta, datetime
from PIL import Image
import numpy as np

class Examination(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='images/')
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

