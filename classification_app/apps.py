from django.apps import AppConfig
from .classification_backend import prepare_svm_classifier


class ClassificationAppConfig(AppConfig):
    name = 'classification_app'

    def ready(self):
        """Running SVM classifier training (classifier selector)"""
        prepare_svm_classifier()
