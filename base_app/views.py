from django.shortcuts import render, HttpResponse


def mainpage(request):
    """View of main page of application."""
    context = {}
    return render(request, 'base_app/mainpage.html', context)
