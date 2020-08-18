from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout


def login_page(request):

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
    logout(request)
    return redirect('base_app:mainpage')

