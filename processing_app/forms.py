from django import forms
from .models import *

class TempImageForm(forms.ModelForm):

    class Meta:
        model = TempImage
        fields = ['image']
