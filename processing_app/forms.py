from django import forms
from .models import *


class ImageProcForm(forms.ModelForm):

    class Meta:
        model = ImageProc
        fields = ['image']


class EditDataForm(forms.Form):
    patient_name = forms.CharField(label='Nazwa pacjenta', max_length=30)
    description = forms.CharField(label='Opis...', max_length=300, widget=forms.Textarea)

