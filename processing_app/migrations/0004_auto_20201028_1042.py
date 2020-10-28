# Generated by Django 3.1 on 2020-10-28 09:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing_app', '0003_auto_20201028_1037'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageproc',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image',
            field=models.ImageField(upload_to='images/<django.db.models.fields.AutoField>'),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_axes',
            field=models.ImageField(upload_to='images/<django.db.models.fields.AutoField>'),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_clahe',
            field=models.ImageField(upload_to='images/<django.db.models.fields.AutoField>'),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_contours',
            field=models.ImageField(upload_to='images/<django.db.models.fields.AutoField>'),
        ),
        migrations.AlterField(
            model_name='imageproc',
            name='image_illumination_removed',
            field=models.ImageField(upload_to='images/<django.db.models.fields.AutoField>'),
        ),
    ]
