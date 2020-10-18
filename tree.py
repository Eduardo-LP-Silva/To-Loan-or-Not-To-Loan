import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import data_preparation as dp
import data_understanding as du

# Loads data from complete_data.csv and returns it in the form of a pandas data frame, depending on the mode (train or test)
def load_data(train):
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = list(dp.complete_data_row.keys()).copy()

    if train:
        feature_cols = header[1 : len(header) - 1]
        x = data[feature_cols]
        y = data.status

        return x, y
    else:
        return data[header]

# Loads data from complete_data.csv and returns it in the form of a pandas data frame with loan_ids
def load_loan_data():
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = list(dp.complete_data_row.keys()).copy()
    return data[header]

# Returns the year of a loan request given the loan_id
def get_loan_year(loan_id):
    with open('./files/loan_train.csv') as loans:
        loan_reader = csv.reader(loans, delimiter=';')
        next(loan_reader)

        for row in loan_reader:

            if len(row) == 7 and int(row[0]) == loan_id:
                date = row[2]
                return int(date[:2])

# Splits data into new train and test files having in consideration the loan year
def rewrite_loans(data, train_years, test_years):
    with open('./files/loan_year_train.csv', 'w', newline='') as loan_year_train, open('./files/loan_year_test.csv', 'w', newline='') as loan_year_test:

        train_writer = csv.writer(loan_year_train, delimiter=';')
        test_writer = csv.writer(loan_year_test, delimiter=';')
        header = list(dp.complete_data_row.keys()).copy()
        train_writer.writerow(header)
        test_writer.writerow(header)

        for train_year in train_years:
            for index, row in data.iterrows():
                if get_loan_year(row['loan_id']) == train_year:
                    train_writer.writerow(row)


        for test_year in test_years:
            for index, row in data.iterrows():
                if get_loan_year(row['loan_id']) == test_year:
                    test_writer.writerow(row)

# Returns x_train, y_train, x_test, y_test splitted accordingly to the train and test dates
def date_split():
    train_data = pd.read_csv('./files/loan_year_train.csv', header=0, delimiter=';')
    test_data = pd.read_csv('./files/loan_year_test.csv', header=0, delimiter=';')
    header = list(dp.complete_data_row.keys()).copy()

    feature_cols = header[1 : len(header) - 1]
    x_train = train_data[feature_cols]
    y_train = train_data.status

    x_test = test_data[feature_cols]
    y_test = test_data.status

    return x_train, y_train, x_test, y_test


# Builds the random forest model and calculates the accuracy and AUC score
def build_model():
    dp.arrange_complete_data(True, True)
    x, y = load_data(True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    # data = load_loan_data()
    # rewrite_loans(data, [95], [96])
    # x_train, y_train, x_test, y_test = date_split()

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

    x_train_balanced, y_train_balanced = balance_train_dataset(x_train, y_train, 1)

    print('\nTraining cases: ' + str(len(x_train_balanced)))
    print('Test cases: ' + str(len(x_test)))

    clf = RandomForestClassifier(max_features='sqrt', criterion='gini', min_samples_split=2, min_samples_leaf=5,
        max_depth=None, n_estimators=500, random_state=42)
    clf.fit(x_train_balanced, y_train_balanced)
    y_pred = clf.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    du.plot_confusion_matrix(cm, ['Rejected', 'Approved'], 'Decision Tree')

    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    get_feature_importance(clf)

    dummy_classifier(x_train_balanced, y_train_balanced, x_test, y_test)

    print('Accuracy: %.1f' % (calc_accuracy(y_test, y_pred) * 100) + '%')
    print('AUC Score: %.2f' % calc_auc(clf, x_test, y_test))

    #hyper_parameter_grid_search(x_train_balanced, y_train_balanced, x_test, y_test)

    return clf

def hyper_parameter_grid_search(x_train, y_train, x_test, y_test):
    param_grid = {
        'max_features': ['sqrt', None],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [300, 500, 900],
        'criterion': ['gini', 'entropy']
    }

    clf = RandomForestClassifier(max_features=None, max_depth=None, random_state=42)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    print('\n--- Best Params ---')
    print(grid_search.best_params_)

    y_pred = grid_search.predict(x_test)

    print('Accuracy: %.1f' % (calc_accuracy(y_test, y_pred) * 100) + '%')
    print('AUC Score: %.2f' % calc_auc(grid_search.best_estimator_, x_test, y_test))

# Returns a model's AUC
def calc_auc(clf, x_test, y_test):
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    return roc_auc_score(y_test, prob_y)

# Returns a model's accuracy
def calc_accuracy(y_test, y_pred):
    return metrics.accuracy_score(y_test, y_pred)

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
        print(list(dp.complete_data_row.keys())[i + 1] + ': ' + '%.2f' % (clf.feature_importances_[i] * 100) + '%')

# Saves an image representing one of the forest's decision trees
def visualize_tree(clf):
    fig = plt.figure(figsize=(100, 100))
    tree.plot_tree(clf.estimators_[0], feature_names=list(dp.complete_data_row.keys()).copy()[1 : len(dp.complete_data_row.keys()) - 1], class_names=['0', '1'],
        filled=True)
    fig.savefig('./figures/decision_tree.png')
    plt.close()

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
