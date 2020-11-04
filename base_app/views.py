from django.shortcuts import render, HttpResponse


def mainpage(request):
    """View of main page of application."""
    context = {}
    return render(request, 'base_app/mainpage.html', context)


def restricted(request):
    """View occurs when access to page is restricted."""
    context = {}
    return render(request, 'base_app/restricted.html', context)