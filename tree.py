import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data_preparation as dp
import data_understanding

def load_data(train):
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = dp.col_names.copy()

    if train:
        feature_cols = header[: len(header) - 1]
        x = data[feature_cols]
        y = data.status

        return x, y
    else:
        header.pop()
        return data[header]

def build_model():
    dp.arrange_complete_data(True)
    x, y = load_data(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    return clf

def run_model(clf):
    with open('./files/prediction.csv', 'w', newline='') as predictions:
        pred_writer = csv.writer(predictions, delimiter=',')
        pred_writer.writerow(['Id', 'Predicted'])

        dp.arrange_complete_data(False)
        x = load_data(False)
        y_pred = clf.predict(x)

        for i, row in x.iterrows():
            pred = -2

            if int(y_pred[i]) == 0:
                pred = 1
            elif int(y_pred[i]) == 1:
                pred = -1
            
            pred_writer.writerow([int(row['loan_id']), pred])

def main():
    clf = build_model()
    run_model(clf)

if __name__ == '__main__':
    main()