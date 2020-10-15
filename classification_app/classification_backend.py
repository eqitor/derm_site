import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.svm import SVC
from joblib import dump, load
import image_processing as ip


# Classification settings

ENABLE_CLASSIFICATION = False





def prepare_classifier():

    global ENABLE_CLASSIFICATION
    print("Checking for SVM classifier...")

    try:
        # Wczytywanie klasyfikatora
        svm = load('./static/svm.joblib')
        svm_features = load('./static/svm_features.joblib')
        stdsc = load('./static/stdsc.joblib')
    except FileNotFoundError as ex:
        print(f"Cannot load classifier, features or scaler, creating new one...")
    else:
        print("SVM loaded successfully!")
        print(f"SVM info: {svm.get_params()}")
        print(f"Usef features: {svm_features}")
        ENABLE_CLASSIFICATION = True
        return

    # Przygotowanie nagłówków tabeli
    column_names = ['A_p', 'A_c', 'solidity', 'extent', 'equivalent diameter', 'circularity', 'p_p', 'b_p/a_p',
                    'b_b/a_b', 'entropy', 'u', 'o2', 'WHITE', 'RED', 'LIGHT_BROWN', 'DARK_BROWN', 'BLUE_GRAY',
                    'BLACK', 'B_mean', 'B_variance', 'B_min', 'B_max', 'G_mean', 'G_variance', 'G_min', 'G_max',
                    'R_mean', 'R_variance', 'R_min', 'R_max', 'RG_mean', 'RB_mean', 'GB_mean', 'E_LR', 'E_TB',
                    'E_TL_BR', 'E_TR_BL', 'H_LR', 'H_TB', 'H_TL_BR', 'H_TR_BL', 'Cor_LR', 'Cor_TB', 'Cor_TL_BR',
                    'Cor_TR_BL', 'Con_LR', 'Con_TB', 'Con_TL_BR', 'Con_TR_BL', 'E_mean', 'H_mean', 'Cor_mean',
                    'Con_mean', 'Class']

    try:
        data = pd.read_csv('./static/datasets/dataset.csv', delimiter=';', names=column_names)
    except FileNotFoundError:
        print("Dataset for SVM not found!")
        ENABLE_CLASSIFICATION = False
        return

    # Podział na klasy i atrybuty
    X = data.drop('Class', axis=1)
    y = data['Class']



    # rank_range - ilość cech w rankingu
    rank_range = X.shape[1]

    # obiekt selekcji cech
    selector = SelectKBest(chi2, k=rank_range)

    # wybór cech
    selector.fit(X, y)

    # indeksy wybranych kolumn
    cols = selector.get_support(indices=True)

    # nowy zbiór w postaci DataFrame
    features_df_new = X.iloc[:, cols]

    # sortowanie zbioru pod względem parametru score
    scores = selector.scores_
    pvalues = selector.pvalues_
    zipped_list = [[float(s), str(f), int(i), float(p)] for s, f, i, p in
                   zip(scores, features_df_new.columns, cols, pvalues)]
    zipped_list = sorted(zipped_list, key=lambda x: x[0], reverse=True)

    # posortowane wyniki
    zipped_list_columns = list(np.array(zipped_list)[:, 1])

    # posortowany zbiór w postaci np
    X_sorted_df = features_df_new[zipped_list_columns]
    X_sorted_np = np.array(X_sorted_df)

    ## Obiekt odpowiedzialny za standaryzację
    stdsc = StandardScaler()


    best_score = 0
    best_svm = None

    # TODO: Znajdź lepszą metodę wyszukiwania
    for gamma in np.arange(66 - 5, 66 + 5, 1):
        for C in np.arange(6 - 5, 6 + 5, 1):

            # inicjalizacja obiektu do walidacji krzyżwej
            rkf = RepeatedKFold(n_splits=2, n_repeats=5)

            # pętla określająca ilosć użytych cech
            for num_of_features in range(1, rank_range):

                # inicjalizacja zmiennej zliczającej średnią dokładność klasyfikatora
                iteration_score = 0

                # standaryzacja wybranych cech
                stdsc.fit(X_sorted_np[:, :num_of_features])
                X_sorted_np_std = stdsc.transform(X_sorted_np[:, :num_of_features])

                # podział danych za pomocą 5 razy powtórzonej 2-krotnej walidacji krzyżowej
                splitted_data = rkf.split(X_sorted_np_std)

                # pętla określająca próbki z walidacji krzyżowej
                for train_index, test_index in splitted_data:
                    # przygotowanie danych trenujących i testowych
                    X_train, X_test = X_sorted_np_std[train_index], X_sorted_np_std[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # inicjalizacja klasyfikatora
                    svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)

                    # uczenie klasyfikatora za pomocą danych trenujących
                    svm.fit(X_train, y_train)

                    # zliczanie średniej dokładnośći
                    iteration_score += (svm.score(X_test, y_test) / 10)

                if svm.score(X_test, y_test) > best_score:
                    best_svm = svm
                    best_score = svm.score(X_test, y_test)
                    best_features = zipped_list_columns[:num_of_features]


    ENABLE_CLASSIFICATION = True
    print(f"Classificator created, score: {best_score}")
    print(f"Used features: {best_features}")

    best_stdsc = StandardScaler()
    best_stdsc.fit(X_sorted_np[:, :len(best_features)])

    dump(best_svm, r"./static/svm.joblib")
    dump(best_features, r"./static/svm_features.joblib")
    dump(best_stdsc, r"./static/stdsc.joblib")

# TODO: Dekorator ktory bedzie blokowal
def classify_image(img):

    try:
        svm_features = load('./static/svm_features.joblib')
        svm = load('./static/svm.joblib')
        stdsc = load('./static/stdsc.joblib')
    except FileNotFoundError as ex:
        print("something wrong")
        return ex.args

    quantification_results = ip.specified_quantification(img, svm_features)
    quantification_results_std = stdsc.transform(np.array(quantification_results).reshape(1, -1))
    classification_result = svm.predict(quantification_results_std)

    return classification_result



