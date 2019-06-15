from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import ListView
from . import models
# Create your views here.


def predict_index(request):
    if request.method == 'POST':
        img = models.predict_index(Url=request.FILES.get('img'))
        img.save()
        context = {'img': img}
        return render(request, 'result.html', context=context)
    return render(request, 'predict.html')

class Post(ListView):
    model = models.Post
    template_name = 'home.html'

def show(request):
    imgs = models.predict_index.objects.all()[0]
    context = {'img': imgs}
    return render(request, 'show.html', context)