from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.


def predict_index(request):
    if request.method == 'POST':
        print(request.FILES.get('img'))
        print(1)
    return render(request, 'predict.html')
