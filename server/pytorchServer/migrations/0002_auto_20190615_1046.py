# Generated by Django 2.2.1 on 2019-06-15 02:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pytorchServer', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='predict_index',
            old_name='Img',
            new_name='Url',
        ),
    ]
