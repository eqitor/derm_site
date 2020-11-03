from django.apps import AppConfig
from .classification_backend import prepare_svm_classifier, prepare_tree_classifier


class ClassificationAppConfig(AppConfig):
    name = 'classification_app'

    def ready(self):
        """Running SVM or tree classifier training (classifier selector)"""
        classifier_decision = 0
        while not(classifier_decision == 1 or classifier_decision == 2):
            classifier_decision = int(input("Select classifier:\n 1 SVM \n 2 DecisionTree \n [1/2]"))
        print(f"Selected {classifier_decision}")
        if classifier_decision == 1:
            prepare_svm_classifier()
        elif classifier_decision == 2:
            prepare_tree_classifier()
