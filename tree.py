import csv
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import data_preparation as dp
import data_understanding as du

# Set holding the correlated features removed during training
corr_feats = set()

# Loads data from complete_data.csv and returns it in the form of a pandas data frame, depending on the mode (train or test)
def load_data(train):
    data = pd.read_csv('./files/complete_data.csv', header=0, delimiter=';')
    header = list(dp.complete_data_row.keys()).copy()

    if train:
        feature_cols = header[1 : len(header) - 1]
        x = data[feature_cols]
        y = data.status

        global corr_feats
        x, corr_feats = dp.remove_correlated_attributes(x)

        return x, y
    else:
        return data[header].drop(labels=corr_feats, axis=1)

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
def build_model(hp_grid_search=False):
    dp.arrange_complete_data(True, True)
    x, y = load_data(True)

    # Best Weights: {-1: 1, 1: 7}
    clf = RandomForestClassifier(max_features='log2', criterion='gini', min_samples_split=5, min_samples_leaf=2,
        max_depth=None, n_estimators=500, class_weight={-1: 1, 1: 7}, random_state=42)
    clf_original = clone(clf)

    # clf = hyper_parameter_randomized_search()

    # data = load_loan_data()
    # rewrite_loans(data, [95], [96])
    # x_train, y_train, x_test, y_test = date_split()

    x_train, x_test, y_train, y_test = strat_train_test_split(x, y, 0.2)
    x_train_balanced, y_train_balanced = smote_and_undersample(x_train, y_train, 0.5, 3)

    print('\nTraining cases: ' + str(len(x_train_balanced)))
    print('Test cases: ' + str(len(x_test)))

    clf.fit(x_train_balanced, y_train_balanced)
    y_pred = clf.predict(x_test)
    eval_trained_model(clf, x_train_balanced, y_train_balanced, x_test, y_test, y_pred)
    evaluate_model_kfold(clf_original, x_train_balanced, y_train_balanced, 10)

    if hp_grid_search:
        hyper_parameter_grid_search(x_train_balanced, y_train_balanced, x_test, y_test)

    return clf

# Evaluates a (untrained) model with repeated stratified k-folds
def evaluate_model_kfold(clf, x, y, k=5):
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=42)

    scores = cross_validate(clf, x, y, scoring=['roc_auc'], cv=cv,
        n_jobs=-1)

    print('\n--- Repeated Stratified K-Fold Average Performance ---')

    for key, vals in scores.items():
        if key.startswith('test_'):
            print('%s: %.5f' % (key, np.mean(vals)))

# Evaluates a trained model given its training and test data sets, as well as its predictions
def eval_trained_model(clf, x_train, y_train, x_test, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    du.plot_confusion_matrix(cm, ['Rejected', 'Approved'], 'Decision Tree')

    get_feature_importance(clf)
    dummy_classifier(x_train, y_train, x_test, y_test)

    print('\n--- Fitted Model Performance ---')
    print('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
    print('Precision: %.2f' % (precision_score(y_test, y_pred)))
    print('Recall: %.2f' % (recall_score(y_test, y_pred)))
    print('F1: %.2f' % (f1_score(y_test, y_pred)))
    print('AUC Score: %.5f' % calc_auc(clf, x_test, y_test))

# Train / Test Stratified Dataset Split
def strat_train_test_split(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1, stratify=y)
    train_counter = Counter(y_train)
    test_counter = Counter(y_test)

    negative_percent_train = (train_counter[-1] / len(y_train)) * 100
    negative_percent_test = (test_counter[-1] / len(y_test)) * 100

    print('\nAfter Stratification')
    print('Positive training cases:' + str(train_counter[1]))
    print('Negative training cases:' + str(train_counter[-1]))
    print('Positive test cases:' + str(test_counter[1]))
    print('Negative test cases:' + str(test_counter[-1]))

    print('\nTraining negative cases ratio: %.2f' % negative_percent_train + '%')
    print('Test negative cases ratio: %.2f' % negative_percent_test + '%')

    return x_train, x_test, y_train, y_test

# Executes a hyper parameter randomized search on a Random Forest
def hyper_parameter_randomized_search():
    param_grid = {
        'max_features': ['sqrt', None, 'log2'],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [300, 500, 900],
        'criterion': ['gini', 'entropy']
    }

    clf = RandomForestClassifier(max_features=None, max_depth=None, random_state=42)
    randomized_search = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    return randomized_search

# Executes a hyper parameter grid search on a Random Forest
def hyper_parameter_grid_search(x_train, y_train, x_test, y_test):
    param_grid = {
        'max_features': ['sqrt', None, 'log2'],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [300, 500, 900],
        'criterion': ['gini', 'entropy']
    }

    clf = RandomForestClassifier(max_features=None, max_depth=None, random_state=42)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    print('\n--- Hyper Parameter Grid Search Results ---')
    print(grid_search.best_params_)

    y_pred = grid_search.predict(x_test)

    print('Accuracy: %.1f' % (accuracy_score(y_test, y_pred) * 100) + '%')
    print('AUC Score: %.2f' % calc_auc(grid_search.best_estimator_, x_test, y_test))

# Returns a model's AUC
def calc_auc(clf, x_test, y_test):
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    return roc_auc_score(y_test, prob_y)

# Builds, trains and evaluates a dummy classifier
def dummy_classifier(x_train, y_train, x_test, y_test):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    print('\nDummy Score: %.2f' % (dummy.score(x_test, y_test)))

def smote_and_undersample(x_train, y_train, ratio, k_neighbors=5):
    sm = BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=42)
    x_upsampled, y_upsampled = sm.fit_resample(x_train, y_train)
    y_counter = Counter(y_upsampled)

    print('\nAfter SMOTE (1/2)')
    print('Positive, Negative training Y: %s' % y_counter)

    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    x_undersampled, y_undersampled = rus.fit_resample(x_upsampled, y_upsampled)
    y_counter = Counter(y_undersampled)

    print('After Undersampling (2/2)')
    print('Positive, Negative training Y: %s' % y_counter)

    return x_undersampled, y_undersampled

# Balances the training set given a ratio
def undersample_majority_class(x_train, y_train, ratio):
    x_train_majority = x_train[y_train.values == 1]
    y_train_majority = y_train[y_train.values == 1]

    x_train_minority = x_train[y_train.values == -1]
    y_train_minority = y_train[y_train.values == -1]

    sample_no = int(len(x_train_minority.index) * ratio)

    x_train_majority_downsampled = resample(x_train_majority, replace=False, n_samples=sample_no, random_state=123)
    y_train_majority_downsampled = resample(y_train_majority, replace=False, n_samples=sample_no, random_state=123)

    x_train_balanced = pd.concat([x_train_majority_downsampled, x_train_minority])
    y_train_balanced = pd.concat([y_train_majority_downsampled, y_train_minority])

    print('\nAfter Undersampling')
    print('Positive | Negative training X: ' + str(len(x_train_balanced[y_train_balanced.values == 1])) + ' | ' +
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
    plot_tree(clf.estimators_[0], feature_names=list(dp.complete_data_row.keys()).copy()[1 : len(dp.complete_data_row.keys()) - 1], class_names=['0', '1'],
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
        #y_pred = clf.predict(x)
        y_prob = clf.predict_proba(x)

        for i, row in x.iterrows():
            pred_writer.writerow([int(loan_ids[i]), y_prob[i][1]])

def main():
    parser = argparse.ArgumentParser(description='Random Forest Classifier')
    parser.add_argument('-t', dest='test', action='store_true', default=False, help='Generate Kaggle test set predictions')
    parser.add_argument('-v', dest='vis_tree', action='store_true', default=False, help='Generate image of the Decision Tree')
    parser.add_argument('-g', dest='grid_search', action='store_true', default=False, help='Perform hyper-parameter grid search')
    args = parser.parse_args()
    clf = build_model(args.grid_search)

    if args.vis_tree:
        visualize_tree(clf)

    if args.test:
        run_model(clf)

if __name__ == '__main__':
    main()
