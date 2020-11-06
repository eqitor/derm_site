from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.base import ContentFile

from .forms import *
from .models import ImageProc
from image_processing import get_processing_image, get_pil_image, remove_uneven_illumination,\
    create_contours_image, colour_quantification, create_axes_image, clahe_image, asymmetry_quantification


@login_required
def processing_image_upload_view(request):
    """View used in processing_app to upload image.."""
    if request.method == 'POST':
        form = ImageProcForm(request.POST, request.FILES)

        if form.is_valid():
            # create and separated db object to provide ID before image saving
            db_object = ImageProc.objects.create()
            db_object.save()

            # saving image to new created db object
            db_object.image = request.FILES['image']
            db_object.user_field = request.user
            db_object.save()

            # saving image data in session
            request.session['image'] = db_object.image.url
            request.session['id'] = db_object.id

            return redirect('processing_app:processing')
    else:
        form = ImageProcForm()
    return render(request, 'processing_app/processing_image_upload.html', {'form': form})


def processing(request):
    """View computes image parameters and additional images."""
    image = get_processing_image(str(settings.BASE_DIR) + request.session['image'])
    image_name = str(request.session['image']).split('/')[-1]
    db_object = ImageProc.objects.get(id=request.session['id'])

    # create no illumination image
    removed_illumination = remove_uneven_illumination(image)
    removed_illumination_pil = get_pil_image(removed_illumination)
    db_object.image_illumination_removed.save('no_ilumination ' + image_name,
                                              ContentFile(removed_illumination_pil), save=False)

    # create contours
    image_contours = create_contours_image(image.copy())
    image_contours_pil = get_pil_image(image_contours)
    db_object.image_contours.save('contours ' + image_name, ContentFile(image_contours_pil), save=False)

    # create axes
    image_axes = create_axes_image(image.copy())
    image_axes_pil = get_pil_image(image_axes)
    db_object.image_axes.save('axes ' + image_name, ContentFile(image_axes_pil), save=False)

    # create CLAHE
    image_clahe = clahe_image(image.copy())
    image_clahe_pil = get_pil_image(image_clahe)
    db_object.image_clahe.save('clahe ' + image_name, ContentFile(image_clahe_pil), save=False)

    colour_features = colour_quantification(image)

    db_object.white_color = colour_features['WHITE']
    db_object.red_color = colour_features['RED']
    db_object.light_brown_color = colour_features['LIGHT_BROWN']
    db_object.dark_brown_color = colour_features['DARK_BROWN']
    db_object.blue_gray_color = colour_features['BLUE_GRAY']
    db_object.black_color = colour_features['BLACK']

    asymmetry_features = asymmetry_quantification(image, enable_processing_features=True)

    db_object.a_p_feature = asymmetry_features['a_p']
    db_object.b_p_feature = asymmetry_features['b_p']
    db_object.a_b_feature = asymmetry_features['a_b']
    db_object.b_b_feature = asymmetry_features['b_b']
    db_object.area_p_feature = asymmetry_features['A_p']
    db_object.area_c_feature = asymmetry_features['A_c']
    db_object.solidity_feature = asymmetry_features['solidity']
    db_object.extent_feature = asymmetry_features['extent']
    db_object.equivalent_diameter_feature = asymmetry_features['equivalent diameter']
    db_object.circularity_feature = asymmetry_features['circularity']
    db_object.p_p_feature = asymmetry_features['p_p']
    db_object.b_p_a_p_feature = asymmetry_features['b_p/a_p']
    db_object.b_b_a_b_feature = asymmetry_features['b_b/a_b']
    db_object.entropy_feature = asymmetry_features['entropy']

    db_object.save()

    request.user.profile.processed_images += 1
    request.user.save()

    return redirect('processing_app:results', request.session['id'])


@login_required
def results(request, image_id):
    """View shows results of processing."""
    db_object = ImageProc.objects.get(id=image_id)

    # redirect if user isn't owner of image
    if db_object.user_field != request.user:
        return redirect('base_app:restricted')

    context = {
        'id': db_object.id,
        'date': str(db_object.created)[0:16],
        'patient_name': db_object.patient_name,
        'description': db_object.description,
        'classification_result': db_object.classification_result,
        'image': db_object.image.url,
        'image_clahe': db_object.image_clahe.url,
        'image_axes': db_object.image_axes.url,
        'image_contours': db_object.image_contours.url,
        'image_illumination_removed': db_object.image_illumination_removed.url,
        'white': format(db_object.white_color * 100, '.2f'),
        'red': format(db_object.red_color * 100, '.2f'),
        'light_brown': format(db_object.light_brown_color * 100, '.2f'),
        'dark_brown': format(db_object.dark_brown_color * 100, '.2f'),
        'blue_gray': format(db_object.blue_gray_color * 100, '.2f'),
        'black': format(db_object.black_color * 100, '.2f'),
        'a_p': format(db_object.a_p_feature, '.2f'),
        'b_p': format(db_object.b_p_feature, '.2f'),
        'a_b': format(db_object.a_b_feature, '.2f'),
        'b_b': format(db_object.b_b_feature, '.2f'),
        'area_p': format(db_object.area_p_feature, '.2f'),
        'area_c': format(db_object.area_c_feature, '.2f'),
        'solidity': format(db_object.solidity_feature, '.2f'),
        'extent': format(db_object.extent_feature, '.2f'),
        'equivalent_diameter': format(db_object.equivalent_diameter_feature, '.2f'),
        'circularity': format(db_object.circularity_feature, '.2f'),
        'p_p': format(db_object.p_p_feature, '.2f'),
        'b_p/a_p': format(db_object.b_p_a_p_feature, '.2f'),
        'b_b/a_b': format(db_object.b_b_a_b_feature, '.2f'),
        'entropy': format(db_object.entropy_feature, '.2f'),
    }

    return render(request, 'processing_app/results.html', context)


def edit_data(request):
    """View shows form to edit processed image data."""
    if request.method == 'POST':
        form = EditDataForm(request.POST)
        if form.is_valid():
            db_object = ImageProc.objects.get(id=request.session['id'])
            db_object.description = form.cleaned_data['description']
            db_object.patient_name = form.cleaned_data['patient_name']
            db_object.save()

            return redirect('processing_app:results', request.session['id'])
    else:
        form = EditDataForm()

    return render(request, 'processing_app/edit_data.html', {'form': form})
