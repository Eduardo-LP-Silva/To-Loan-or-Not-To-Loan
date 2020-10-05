import csv
import pandas as pd
import sklearn
import scipy
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
import data_preparation


def build_prediction(clf):
    with open('./files/prediction.csv', 'w', newline='') as predictions:
        pred_writer = csv.writer(predictions, delimiter=',')
        pred_writer.writerow(['Id', 'Predicted'])

        data_preparation.arrange_complete_data(False)
        x = load_data(False)
        loan_ids = x['loan_id'].copy()
        x.drop(['loan_id'], axis=1, inplace=True)
        y_pred = clf.predict(x)

        for i, row in x.iterrows():
            pred_writer.writerow([int(loan_ids[i]), int(y_pred[i])])

def load_data(train):
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = data_preparation.col_names.copy()

    if train:
        feature_cols = header[1 : len(header) - 1]
        x = data[feature_cols]
        y = data.status

        return x, y
    else:
        header.pop()
        return data[header]

def createMatrix(val):
    mat = []
    for i in val:
        mat.append([i])
    return mat

def k_nn():
    data_preparation.arrange_complete_data(True)
    x, y = load_data(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    print('X TRAIN')
    print(x_train)
    print('X TEST')
    print(x_test)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    print(clf)

    y_expect = y_test
    y_pred = clf.predict(x_test)

    print(y_pred)
    build_prediction(clf)

    print(metrics.classification_report(y_expect, y_pred))

k_nn()
