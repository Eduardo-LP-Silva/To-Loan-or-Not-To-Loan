import csv
import itertools
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import data_preparation as dp
import data_understanding

# Loads data from complete_data.csv and returns it in the form of a pandas data frame, depending on the mode (train or test)
def load_data(train):
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = dp.col_names.copy()

    if train:
        feature_cols = header[1 : len(header) - 1]
        x = data[feature_cols]
        y = data.status

        return x, y
    else:
        header.pop()
        return data[header]

# Builds the random forest model and calculates the accuracy and AUC score
def build_model():
    dp.arrange_complete_data(True)
    x, y = load_data(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    negative_percent_train = (len(y_train[y_train.values == -1]) / len(y_train)) * 100
    negative_percent_test = (len(y_test[y_test.values == -1]) / len(y_test)) * 100
    positive_test_cases = len(y_test[y_test.values == 1])
    negative_test_cases = len(y_test[y_test.values == -1])
    
    print('\nPositive training cases:' + str(len(y_train[y_train.values == 1])))
    print('Negative training cases:' + str(len(y_train[y_train.values == -1])))
    print('Positive test cases:' + str(positive_test_cases))
    print('Negative test cases:' + str(negative_test_cases))

    print('\nTraining negative cases ratio: %.2f' % negative_percent_train + '%')
    print('Test negative cases ratio: %.2f' % negative_percent_test + '%')

    x_train_balanced, y_train_balanced = balance_train_dataset(x_train, y_train, 1.7)

    print('\nTraining cases: ' + str(len(x_train_balanced)))
    print('Test cases: ' + str(len(x_test)))

    clf = RandomForestClassifier(max_features=None, min_samples_split=2, min_samples_leaf=5, max_depth=None, random_state=42)
    clf.fit(x_train_balanced, y_train_balanced)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['Rejected', 'Approved'])
    
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    get_feature_importance(clf)

    dummy_classifier(x_train_balanced, y_train_balanced, x_test, y_test)

    print('Accuracy: %.1f' % (metrics.accuracy_score(y_test, y_pred) * 100) + '%')
    print('AUC Score: %.2f' % roc_auc_score(y_test, prob_y))

    return clf

# Builds, trains and evaluates a dummy classifier
def dummy_classifier(x_train, y_train, x_test, y_test):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    print('\nDummy Score: %.2f' % (dummy.score(x_test, y_test) * 100) + '%')

# Balances the training set given a ratio
def balance_train_dataset(x_train, y_train, ratio):
    x_train_majority = x_train[y_train.values == 1]
    y_train_majority = y_train[y_train.values == 1]

    x_train_minority = x_train[y_train.values == -1]
    y_train_minority = y_train[y_train.values == -1]

    sample_no = int(len(x_train_minority.index) * ratio)

    x_train_majority_downsampled = resample(x_train_majority, replace=False, n_samples=sample_no, random_state=123)
    y_train_majority_downsampled = resample(y_train_majority, replace=False, n_samples=sample_no, random_state=123)

    x_train_balanced = pd.concat([x_train_majority_downsampled, x_train_minority])
    y_train_balanced = pd.concat([y_train_majority_downsampled, y_train_minority])

    print('\nPositive | Negative training X: ' + str(len(x_train_balanced[y_train_balanced.values == 1])) + ' | ' + 
        str(len(x_train_balanced[y_train_balanced.values == -1])))
    print('Positive | Negative training Y: ' + str(len(y_train_balanced[y_train_balanced.values == 1])) + ' | ' + 
        str(len(y_train_balanced[y_train_balanced.values == -1])))

    return x_train_balanced, y_train_balanced

# Displays the relevance of each predictive attribute in respect to the final prediction
def get_feature_importance(clf):
    print('\n--- Feature Importance ---\n')

    for i in range(len(clf.feature_importances_)):
        print(dp.col_names[i + 1] + ': ' + '%.2f' % (clf.feature_importances_[i] * 100) + '%')

# Saves an image representing one of the forest's decision trees
def visualize_tree(clf):
    fig = plt.figure(figsize=(100, 100))
    tree.plot_tree(clf.estimators_[0], feature_names=dp.col_names.copy()[1 : len(dp.col_names) - 1], class_names=['0', '1'], 
        filled=True)
    fig.savefig('./figures/decision_tree.png')

# Plots a confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    #plt.show()
    plt.savefig('./figures/decision_tree_confusion_matrix.png')

# Runs the model on the competition data
def run_model(clf):
    with open('./files/prediction.csv', 'w', newline='') as predictions:
        pred_writer = csv.writer(predictions, delimiter=',')
        pred_writer.writerow(['Id', 'Predicted'])

        dp.arrange_complete_data(False)
        x = load_data(False)
        loan_ids = x['loan_id'].copy()
        x.drop(['loan_id'], axis=1, inplace=True)
        y_pred = clf.predict(x)

        for i, row in x.iterrows():
            pred_writer.writerow([int(loan_ids[i]), int(y_pred[i])])

def main():
    clf = build_model()
    visualize_tree(clf)
    run_model(clf)

if __name__ == '__main__':
    main()