from django import forms
from .models import *


class ImageProcForm(forms.ModelForm):

    class Meta:
        model = ImageProc
        fields = ['image']
