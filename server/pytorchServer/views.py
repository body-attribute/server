from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import ListView
from . import models

import os
import mat4py
from PIL import Image
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms, models
from . import helper
from collections import OrderedDict
import numpy as np

# Create your views here.

class Predict():
    def __init__(self):
        self.attributes = [
            'backpack', 'bag', 'handbag', 'clothes', 'down', 'up',
            'hair', 'hat', 'gender', 'upblack', 'upwhite','upred',
            'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen', 'downblack',
            'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue',
            'downgreen', 'downbrown', 'young','teenager','adult', 'old'
        ]
        self.resnet = models.resnet18(pretrained=False)
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 256)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(256, 128)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(128, 30)),
                          ('output', nn.Sigmoid())
                          ]))
        self.resnet.fc = classifier
        self.resnet.load_state_dict(torch.load('C:/Users/ly/.torch/models/Market-unPreTrainedResNet18.pth'))
        self.transforms = transforms.Compose([transforms.Resize((224, 224)), 
                                              transforms.ToTensor()])

    def pre(self, img):
        img = self.transforms(img)
        print(img.shape)
        # helper.imshow(img, normalize=False)
        img = torch.unsqueeze(img, 0)
        print(img.shape)
        with torch.no_grad():
            res = self.resnet(img)
            res = res.numpy()
        attrs = []
        print(res[0])
        for i in range(len(res[0])):
            if res[0][i] >= 0.5:
                attrs.append(self.attributes[i])
        return attrs

P = Predict()

def predict(request):
    if request.method == 'POST':
        # img = models.Image(Url=request.FILES.get('img'))
        # img.save()
        # context = {'img': img}
        img = request.FILES.get('img')
        print(type(img), img.name)
        im = Image.open(img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        print(im.size)
        print(P.pre(im))

        # return render(request, 'result.html', context=context)
    return render(request, 'predict.html')


def show(request):
    imgs = models.Image.objects.all()[0]
    context = {'img': imgs}
    return render(request, 'result.html', context)



# attributes = [
#     'backpack', 'bag', 'handbag', 'clothes', 'down', 'up',
#     'hair', 'hat', 'gender', 'upblack', 'upwhite','upred',
#     'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen', 'downblack',
#     'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue',
#     'downgreen', 'downbrown', 'young','teenager','adult', 'old'
# ]