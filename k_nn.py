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


def build_prediction(pred):
    with open('./files/loan_test.csv') as loans, open('./files/prediction.csv', 'w', newline='') as prediction:
        loan_reader = csv.reader(loans, delimiter=';')
        prediction_writer = csv.writer(prediction, delimiter=';')

        prediction_writer.writerow(['Id', 'Prediction'])

        count = 0
        for row in loan_reader:
            if row[0] != 'loan_id':
                prediction_writer.writerow([row[0],pred[count]])
                count = count + 1
        print('Built Prediction File')

def createMatrix(val):
    mat = []
    for i in val:
        mat.append([i])
    return mat

def k_nn():
    clean_loans = pd.read_csv('./files/loan_train_clean.csv', delimiter=';')
    clean_loans.columns = ['loan_id', 'amount', 'duration', 'dist. no. of inhabitants', 'dist. no. of municipalities with inhabitants < 499', 'dist. no. of municipalities with inhabitants 500-1999', 'dist. no. of municipalities with inhabitants 2000-9999', 'dist. no. of municipalities with inhabitants >10000', 'dist. no. of cities', 'dist. ratio of urban inhabitants', 'dist. average salary', 'dist. unemploymant rate', 'dist. no. of enterpreneurs per 1000 inhabitants', 'dist. no. of commited crimes', 'status']

    # TODO: ADD NORMALIZATION

    # x_train = data_preparation.normalize_data()
    train_amount_values = clean_loans.amount.values
    # x_train = createMatrix(train_amount_values)
    x_train = data_preparation.normalize_train_data()
    y_train = clean_loans.status.values

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    print(clf)

    test_loans = pd.read_csv('./files/loan_test.csv', delimiter=';')
    test_loans.columns = ['loan_id', 'account_id', 'date', 'amount','duration', 'payments', 'status']
    test_amount_values = test_loans.amount.values
    # x_test = createMatrix(test_amount_values)
    x_test = data_preparation.normalize_test_data()
    y_test = test_loans.status.values

    y_expect = y_test
    y_pred = clf.predict(x_test)

    print(y_pred)
    build_prediction(y_pred)

data_preparation.arrange_data()
k_nn()
