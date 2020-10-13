import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import data_preparation as dp

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

# Builds the k nearest neighbor model and calculates the accuracy and AUC score
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

    x_train_balanced, y_train_balanced = balance_train_dataset(x_train, y_train, 1)

    print('\nTraining cases: ' + str(len(x_train_balanced)))
    print('Test cases: ' + str(len(x_test)))

    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train_balanced, y_train_balanced)
    y_pred = clf.predict(x_test)

    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    # Print classification report
    print(classification_report(y_test, y_pred))
    print('Accuracy: %.1f' % (calc_accuracy(y_test, y_pred) * 100) + '%')
    print('AUC Score: %.2f' % calc_auc(clf, x_test, y_test))

    # best_clf = hyper_parameter_grid_search(x_train_balanced, y_train_balanced, x_test, y_test)
    #
    # #Fit the model
    # best_model = best_clf.fit(x_train_balanced, y_train_balanced)
    #
    # print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    # print('Best p:', best_model.best_estimator_.get_params()['p'])
    # print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    # new_y_pred = best_clf.predict(x_test)
    #
    # prob_y = best_clf.predict_proba(x_test)
    # prob_y = [p[1] for p in prob_y]
    #
    # # Print classification report
    # print(classification_report(y_test, new_y_pred))
    # print('Accuracy: %.1f' % (calc_accuracy(y_test, new_y_pred) * 100) + '%')
    # print('AUC Score: %.2f' % calc_auc(best_clf, x_test, y_test))

    return clf

def hyper_parameter_grid_search(x_train_balanced, y_train_balanced, x_test, y_test):
    # List Hyperparameters that we want to tune.
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]

    # Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # Create new KNN object
    knn_2 = KNeighborsClassifier()
    # Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)

    return clf

# Returns a model's AUC
def calc_auc(clf, x_test, y_test):
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    return roc_auc_score(y_test, prob_y)

# Returns a model's accuracy
def calc_accuracy(y_test, y_pred):
    return metrics.accuracy_score(y_test, y_pred)


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
    run_model(clf)

if __name__ == '__main__':
    main()
