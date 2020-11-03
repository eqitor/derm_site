from django import forms
from .models import *


class ExaminationForm(forms.ModelForm):
    """Form used to create new examination (upload image to classify)"""
    class Meta:
        model = Examination
        fields = ['image']
