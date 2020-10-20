from django import forms
from .models import *

class ExaminationForm(forms.ModelForm):

    class Meta:
        model = Examination
        fields = ['image']
