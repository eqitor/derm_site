from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse

from PIL import Image
import numpy as np

from .forms import *
from processing_app.models import ImageProc

from .classification_backend import classify_image, ENABLE_CLASSIFICATION
from image_processing import get_processing_image


def look_for_enable_classification(fun):
    """Decorator that checks if classification function is enabled before running other function"""
    def wrapper(*args, **kwargs):
        if ENABLE_CLASSIFICATION:
            rv = fun(*args, **kwargs)
            return rv
        else:
            return HttpResponse("Classification unavailable.")

    return wrapper


@look_for_enable_classification
def examination_image_view(request):
    """View used in classification_app to upload and crop image to classify."""
    if request.method == 'POST':
        form = ExaminationForm(request.POST, request.FILES)

        if form.is_valid():
            # create and separated db object to provide ID before image saving
            db_object = Examination.objects.create()
            db_object.save()

            # saving image to new created db object
            db_object.image = request.FILES['image']
            db_object.user_field = request.user
            db_object.save()

            # saving image data in session
            request.session['image'] = db_object.image.url
            request.session['id'] = db_object.id
            return JsonResponse({'message': 'works'})

    else:
        form = ExaminationForm()
        context = {'form': form}
        return render(request, 'classification_app/examination_image_form.html', context)


@look_for_enable_classification
def examination_image_for_processing(request):
    """View used in classification_app if image came from processing_app (omits image uploading and cropping)"""
    db_object = ImageProc.objects.get(id=request.session['id'])

    # prepare image to classification
    image_to_process = get_processing_image(db_object.image)

    # image classification
    classification_result = classify_image(image_to_process)

    if classification_result[0] == 1:
        request.session['classification_result'] = 'malignant'
    else:
        request.session['classification_result'] = 'benign'

    db_object.classification_result = request.session['classification_result']
    db_object.save()

    return redirect('classification_app:success_for_processing')


@look_for_enable_classification
def success(request):
    """View shows result of classification."""
    if request.session["classification_result"] == 'benign':
        result = 'łagodny'
    elif request.session["classification_result"] == 'malignant':
        result = 'złośliwy'
    request.user.profile.classifications += 1
    request.user.save()
    context = {'result': result}
    return render(request, 'classification_app/success.html', context)


@look_for_enable_classification
def success_for_processing(request):
    """View shows result of classification of image from processing app."""
    if request.session["classification_result"] == 'benign':
        result = 'łagodny'
    elif request.session["classification_result"] == 'malignant':
        result = 'złośliwy'
    request.user.profile.classifications += 1
    request.user.save()
    context = {'result': result}
    return render(request, 'classification_app/success_for_processing.html', context)


@look_for_enable_classification
def run_processing(request):
    """View performs image processing. Used with cropped images."""
    db_object = Examination.objects.get(id=request.session['id'])
    image_to_process = get_processing_image(db_object.image)

    classification_result = classify_image(image_to_process)

    if classification_result[0] == 1:
        request.session['classification_result'] = 'malignant'
    else:
        request.session['classification_result'] = 'benign'

    db_object.classification_result = request.session['classification_result']
    db_object.save()
    print('redirection to success ' + request.session['classification_result'])
    return redirect('classification_app:success')
