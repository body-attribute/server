from django.db import models

# Create your models here.
class Image(models.Model):
    Url = models.ImageField(upload_to='images/')
