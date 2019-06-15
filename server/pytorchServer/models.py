from django.db import models

# Create your models here.
class predict_index(models.Model):
    Url = models.ImageField(upload_to='images/')

class Post(models.Model):
    title = models.TextField()
    cover = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.title