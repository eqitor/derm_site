from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

from .forms import CreateUserForm
from processing_app.models import ImageProc


def login_page(request):
    """View of application login page, creates forms and makes data validation."""
    if request.user.is_authenticated:
        return redirect('base_app:mainpage')
    else:
        if request.method == 'POST':
            username = request.POST.get('input-login')
            password = request.POST.get('input-password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('base_app:mainpage')
            else:
                messages.info(request, 'Podano nieprawidłowy login lub hasło.')

        context = {}
        return render(request, 'account_management/login_page.html', context)


def logout_user(request):
    """View of logout function, logs user out."""
    logout(request)
    return redirect('base_app:mainpage')


def register_page(request):
    """View of register function, creates register form and makes data validation"""
    if request.user.is_authenticated:
        return redirect('base_app:mainpage')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Pomyślnie utworzono użytkownika {}. Teraz możesz się zalogować!'.format(user))
                return redirect('account_management:login_page')

        context = {'form':form}
        return render(request, 'account_management/register_page.html', context)


@login_required
def processing_history(request):
    """View of page with user processed images history, collects user processed images and returns urls in context"""
    examinations = []

    for examination in ImageProc.objects.filter(user_field=request.user).order_by('created'):
        examinations_row = [examination.created,
                      examination.patient_name,
                      examination.description,
                      examination.image.url]

        examinations.append(examinations_row)

    context = {
        'examinations': examinations,
    }

    return render(request, 'account_management/processing_history.html', context)

