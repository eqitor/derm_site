# Generated by Django 3.1 on 2020-10-28 09:51

from django.db import migrations, models
import processing_app.models


class Migration(migrations.Migration):

    dependencies = [
        ('processing_app', '0004_auto_20201028_1042'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageproc',
            name='image',
            field=models.ImageField(upload_to=processing_app.models.update_filename),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_axes',
            field=models.ImageField(upload_to=processing_app.models.update_filename),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_clahe',
            field=models.ImageField(upload_to=processing_app.models.update_filename),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_contours',
            field=models.ImageField(upload_to=processing_app.models.update_filename),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_illumination_removed',
            field=models.ImageField(upload_to=processing_app.models.update_filename),
        ),
    ]