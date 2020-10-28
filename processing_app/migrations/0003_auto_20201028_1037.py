# Generated by Django 3.1 on 2020-10-28 09:37

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('processing_app', '0002_tempimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageProc',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(blank=True, default=datetime.datetime.now)),
                ('image', models.ImageField(upload_to='images/<built-in function id>')),
                ('image_clahe', models.ImageField(upload_to='images/<built-in function id>')),
                ('image_axes', models.ImageField(upload_to='images/<built-in function id>')),
                ('image_contours', models.ImageField(upload_to='images/<built-in function id>')),
                ('image_illumination_removed', models.ImageField(upload_to='images/<built-in function id>')),
                ('user_field', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RemoveField(
            model_name='processedimage',
            name='user_field',
        ),
        migrations.DeleteModel(
            name='TempImage',
        ),
        migrations.DeleteModel(
            name='ProcessedImage',
        ),
    ]
