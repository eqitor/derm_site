from django.apps import AppConfig
from .classification_backend import prepare_classifier

class ClassificationAppConfig(AppConfig):
    name = 'classification_app'

    def ready(self):
        """Running SVM classifier traning"""
        prepare_classifier()