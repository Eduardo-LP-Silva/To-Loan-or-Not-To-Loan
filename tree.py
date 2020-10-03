import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
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

    clf = DecisionTreeClassifier(min_samples_split=2)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    get_feature_importance(clf)

    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    visualize_tree(clf)

    return clf

def get_feature_importance(clf):
    print('\n--- Feature Importance ---\n')

    for i in range(len(clf.feature_importances_) - 1):
        print(dp.col_names[i] + ': ' + '%.2f' % (clf.feature_importances_[i] * 100) + '%')

def visualize_tree(clf):
    fig = plt.figure(figsize=(100, 100))
    tree.plot_tree(clf, feature_names=dp.col_names.copy()[: len(dp.col_names) - 1], class_names=['0', '1'], 
        filled=True)
    fig.savefig('./figures/decision_tree.png')

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
    #run_model(clf)

if __name__ == '__main__':
    main()