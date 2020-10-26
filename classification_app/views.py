from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *

from PIL import Image
import numpy as np
from .classification_backend import classify_image, ENABLE_CLASSIFICATION
from image_processing import get_processing_image


def look_for_enable_classification(fun):
    def wrapper(*args, **kwargs):
        if ENABLE_CLASSIFICATION:
            rv = fun(*args, **kwargs)
            return rv
        else:
            return HttpResponse("Classification unavailable.")

    return wrapper


# Create your views here.

@look_for_enable_classification
def examination_image_view(request):
    if request.method == 'POST':
        form = ExaminationForm(request.POST, request.FILES)

        if form.is_valid():
            image_to_process = get_processing_image(request.FILES['image'].file)

            classification_result = classify_image(image_to_process)
            print(classification_result)

            if classification_result[0] == 1:
                request.session['classification_result'] = 'malignant'
            else:
                request.session['classification_result'] = 'benign'

            if request.user.is_authenticated:
                post = form.save(commit=False)
                post.user_field = request.user
                post.diagnose = request.session['classification_result']
                form.save()

            return redirect('classification_app:success')
    else:
        form = ExaminationForm()
    return render(request, 'classification_app/examination_image_form.html', {'form': form})


@look_for_enable_classification
def success(request):
    context = {'result': request.session["classification_result"]}
    return render(request, 'classification_app/success.html', context)
