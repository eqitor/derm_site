from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import *
from .models import ProcessedImage
from django.conf import settings
from django.core.files.base import ContentFile

from PIL import Image
import numpy as np
from image_processing import get_processing_image, get_pil_image, remove_uneven_illumination,\
    create_contours_image


def processing_image_upload_view(request):
    if request.method == 'POST':
        form = TempImageForm(request.POST, request.FILES)

        if form.is_valid():

            saved = form.save()
            request.session['image'] = saved.image.url

            return redirect('processing_app:processing')
    else:
        form = TempImageForm()
    return render(request, 'processing_app/processing_image_upload.html', {'form': form})


def processing(request):
    context = {
        'image_url': request.session['image'],
    }

    return render(request, 'processing_app/processing.html', context)


@login_required
def save(request):
    processed_image = ProcessedImage(user_field=request.user, image=request.session['image'])
    processed_image.save()
    return redirect('processing_app:processing')


def remove_illumination(request):
    image = get_processing_image(str(settings.BASE_DIR) + request.session['image'])
    image_name = str(request.session['image']).split('/')[-1]
    removed_illumination = remove_uneven_illumination(image)
    removed_illumination_pil = get_pil_image(removed_illumination)

    removed_illumination_pil_temp = TempImage()
    removed_illumination_pil_temp.image.save('noilumination ' + image_name,
                                             ContentFile(removed_illumination_pil), save=False)
    removed_illumination_pil_temp.save()

    request.session['image'] = removed_illumination_pil_temp.image.url

    return redirect('processing_app:processing')


def contour_image(request):
    image = get_processing_image(str(settings.BASE_DIR) + request.session['image'])
    image_name = str(request.session['image']).split('/')[-1]
    image_contours = create_contours_image(image)
    image_contours_pil = get_pil_image(image_contours)
    image_contours_pil_temp = TempImage()
    image_contours_pil_temp.image.save('contours ' + image_name, ContentFile(image_contours_pil), save=False)
    image_contours_pil_temp.save()

    request.session['image'] = image_contours_pil_temp.image.url

    return redirect('processing_app:processing')
