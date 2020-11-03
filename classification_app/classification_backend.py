import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.svm import SVC
from joblib import dump, load

import image_processing as ip


# Classification settings

ENABLE_CLASSIFICATION = False  # if True, classification function is enabled


def prepare_svm_classifier():
    """Function creates new classifier and input standardizer, or loads old one. Function creates simple interface
    for classifier selection and saves new classifier and standardizer as joblib file."""
    global ENABLE_CLASSIFICATION
    print("Checking for SVM classifier...")

    # Files loading
    try:
        svm = load('./static/svm.joblib')
        svm_features = load('./static/svm_features.joblib')
        standardizer = load('./static/svm_stdsc.joblib')
    except FileNotFoundError:
        print(f"Cannot load classifier, features or scaler, creating new one...")
    else:
        print("SVM loaded successfully!")
        print(f"SVM info: {svm.get_params()}")
        print(f"Usef features: {svm_features}")
        ENABLE_CLASSIFICATION = True
        return

    # New classifier teaching procedure
    column_names = ['A_p', 'A_c', 'solidity', 'extent', 'equivalent diameter', 'circularity', 'p_p', 'b_p/a_p',
                    'b_b/a_b', 'entropy', 'u', 'o2', 'WHITE', 'RED', 'LIGHT_BROWN', 'DARK_BROWN', 'BLUE_GRAY',
                    'BLACK', 'B_mean', 'B_variance', 'B_min', 'B_max', 'G_mean', 'G_variance', 'G_min', 'G_max',
                    'R_mean', 'R_variance', 'R_min', 'R_max', 'RG_mean', 'RB_mean', 'GB_mean', 'E_LR', 'E_TB',
                    'E_TL_BR', 'E_TR_BL', 'H_LR', 'H_TB', 'H_TL_BR', 'H_TR_BL', 'Cor_LR', 'Cor_TB', 'Cor_TL_BR',
                    'Cor_TR_BL', 'Con_LR', 'Con_TB', 'Con_TL_BR', 'Con_TR_BL', 'E_mean', 'H_mean', 'Cor_mean',
                    'Con_mean', 'Class']

    # Loading dataset
    try:
        data = pd.read_csv('./static/datasets/dataset.csv', delimiter=';', names=column_names)
    except FileNotFoundError:
        print("Dataset for SVM not found!")
        ENABLE_CLASSIFICATION = False
        return

    # y = classes, X = attributes
    X = data.drop('Class', axis=1)
    y = data['Class']

    rank_range = X.shape[1]  # rank_range is amount of features as default

    selector = SelectKBest(chi2, k=rank_range)
    selector.fit(X, y)  # feature selection
    selected_columns_indexes = selector.get_support(indices=True)
    selected_columns_dataframe = X.iloc[:, selected_columns_indexes]

    # sorting columns starting with best rank
    scores = selector.scores_
    pvalues = selector.pvalues_
    zipped_list = [[float(s), str(f), int(i), float(p)] for s, f, i, p in
                   zip(scores, selected_columns_dataframe.columns, selected_columns_indexes, pvalues)]
    zipped_list = sorted(zipped_list, key=lambda x: x[0], reverse=True)

    zipped_list_columns = list(np.array(zipped_list)[:, 1])  # sort results

    X_sorted_df = selected_columns_dataframe[zipped_list_columns]  # sorted columns as dataframe
    X_sorted_np = np.array(X_sorted_df)  # sorted columns as numpy array

    standardizer = StandardScaler()

    # variables made to searching best classifier
    best_score = 0
    best_svm = None

    # NOTE: parameters ranges was selected experimentally
    for gamma in np.arange(66 - 5, 66 + 5, 1):
        for C in np.arange(6 - 5, 6 + 5, 1):

            rkf = RepeatedKFold(n_splits=2, n_repeats=5)  # repeated cross validation object

            for number_of_features in range(1, rank_range):
                average_score = 0

                # data standardization
                standardizer.fit(X_sorted_np[:, :number_of_features])
                X_sorted_np_std = standardizer.transform(X_sorted_np[:, :number_of_features])

                # data selected with cross validation method
                splitted_data = rkf.split(X_sorted_np_std)

                # cross validation iterations
                for train_index, test_index in splitted_data:
                    # splitting train and test data
                    X_train, X_test = X_sorted_np_std[train_index], X_sorted_np_std[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # SVC (SVM) classifier
                    svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)
                    svm.fit(X_train, y_train)

                    # counting average score
                    average_score += (svm.score(X_test, y_test) / 10)

                # selecting best score
                if svm.score(X_test, y_test) > best_score:
                    best_svm = svm
                    best_score = svm.score(X_test, y_test)
                    best_features = zipped_list_columns[:number_of_features]

    ENABLE_CLASSIFICATION = True

    print(f"Classificator created, score: {best_score}")
    print(f"Used features: {best_features}")

    best_stdsc = StandardScaler()
    best_stdsc.fit(X_sorted_np[:, :len(best_features)])

    # save classifier and standardizer to joblib files
    dump(best_svm, r"./static/svm.joblib")
    dump(best_features, r"./static/svm_features.joblib")
    dump(best_stdsc, r"./static/svm_stdsc.joblib")


def classify_image(img):
    """Function performs classification of image with given classifier
    @img - input image in np array format
    @return classification_result - string value with classification result (benign or malignant)"""
    try:
        features = load('./static/svm_features.joblib')
        classifier = load('./static/svm.joblib')
        stdsc = load('./static/svm_stdsc.joblib')
    except FileNotFoundError as ex:
        print(f"Something went wrong {ex.args}")
        return

    quantification_results = ip.specified_quantification(img, features)
    quantification_results_std = stdsc.transform(np.array(quantification_results).reshape(1, -1))
    classification_result = classifier.predict(quantification_results_std)

    return classification_result
