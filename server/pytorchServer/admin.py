from django.contrib import admin
from .models import predict_index, Post

# Register your models here.

admin.site.register(predict_index)
admin.site.register(Post)