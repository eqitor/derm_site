from django.shortcuts import render, HttpResponse

def mainpage(request):
    context = {}
    return render(request, 'base_app/mainpage.html', context)
