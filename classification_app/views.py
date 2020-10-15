from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *

from PIL import Image
import numpy as np
from .classification_backend import classify_image

def get_processing_image(uploaded_image):
    pil_img = Image.open(uploaded_image)
    cv_img = np.array(pil_img)
    return cv_img


# Create your views here.
def examination_image_view(request):
    if request.method == 'POST':
        form = ExaminationForm(request.POST, request.FILES)

        if form.is_valid():

            image_to_process = get_processing_image(request.FILES['image'].file)

            classification_result = classify_image(image_to_process)
            print(str(classification_result[0]))
            request.session['classification_result'] = str(classification_result[0])
            # TODO: To niżej (save) tylko dla zalogowanych
            #form.save()
            return redirect('success')
    else:
        form = ExaminationForm()
    return render(request, 'classification_app/examination_image_form.html', {'form': form})


def success(request):
    return HttpResponse(f'Classification result = {request.session["classification_result"]}')
