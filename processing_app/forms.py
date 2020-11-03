from django import forms
from .models import *


class ImageProcForm(forms.ModelForm):
    """Form used to create a new ImageProc db object"""
    class Meta:
        model = ImageProc
        fields = ['image']


class EditDataForm(forms.Form):
    """Form used to edit data of ImageProc"""
    patient_name = forms.CharField(label='Nazwa pacjenta', max_length=30)
    description = forms.CharField(label='Opis...', max_length=300, widget=forms.Textarea)

