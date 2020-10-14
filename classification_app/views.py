from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *


# Create your views here.
def examination_image_view(request):
    if request.method == 'POST':
        form = ExaminationForm(request.POST, request.FILES)

        if form.is_valid():
            print(form.instance)
            form.save()
            return redirect('success')
    else:
        form = ExaminationForm()
    return render(request, 'classification_app/examination_image_form.html', {'form': form})


def success(request):
    return HttpResponse('successfully uploaded')
