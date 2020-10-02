
import csv
import pandas as pd # Usei este pq ja conhecia para ser mais rapido, dps posso dar refactor para fazer com o csv tambem
from pandas import Series, DataFrame
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
import data_understanding

def arrange_data():
    with open('./files/loan_train.csv') as loans, open('./files/loan_train_clean.csv', 'w', newline='') as loans_new, open('./files/account.csv') as accounts, open('./files/district.csv') as districts:
        loan_reader = csv.reader(loans, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        dist_reader = csv.reader(districts, delimiter=';')
        loan_writer = csv.writer(loans_new, delimiter=';')
        attr_data = data_understanding.analyse_data()
        next(loan_reader)

        loan_writer.writerow(['loan_id', 'amount', 'duration', 'dist. no. of inhabitants',
            'dist. no. of municipalities with inhabitants < 499', 'dist. no. of municipalities with inhabitants 500-1999',
            'dist. no. of municipalities with inhabitants 2000-9999', 'dist. no. of municipalities with inhabitants >10000',
            'dist. no. of cities', 'dist. ratio of urban inhabitants', 'dist. average salary',
            'dist. unemploymant rate 96', 'dist. no. of enterpreneurs per 1000 inhabitants',
            'dist. no. of commited crimes 96', 'status'])

        for row in loan_reader:
            if len(row) == 7:
                id = int(row[0])
                pred_attrs = {'amount': int(row[3]), 'duration': int(row[4]), 'payments': int(row[5])}
                status = int(row[6])
                outlier = False

                # check if any attribute is an outlier
                for pred_attr in pred_attrs.keys():
                    if(pred_attrs[pred_attr] < attr_data[pred_attr][0]
                        or pred_attrs[pred_attr] > attr_data[pred_attr][1]):
                        outlier = True
                        break

                if not outlier:
                    #CHECK: what is date in account? Is it relevant? Same for date in loan
                    acc_id = int(row[1])

                    accounts.seek(0)
                    next(acc_reader)

                    for account in acc_reader:
                        if int(account[0]) == acc_id and len(account) == 4:
                            dist_id = int(account[1])

                            districts.seek(0)
                            next(dist_reader)

                            '''
                            District info
                            code; name; region; no. of inhabitants; no. of municipalities with inhabitants < 499;
                            no. of municipalities with inhabitants 500-1999;
                            no. of municipalities with inhabitants 2000-9999;
                            no. of municipalities with inhabitants >10000;
                            no. of cities; ratio of urban inhabitants; average salary; unemploymant rate '95;
                            unemploymant rate '96; no. of enterpreneurs per 1000 inhabitants;
                            no. of commited crimes '95; no. of commited crimes '96
                            '''
                            for district in dist_reader:
                                if int(district[0]) == dist_id and len(district) == 16:
                                    #amount = duration * payments, discard payments or duration since they're correlated
                                    loan_writer.writerow([row[0], row[3], row[4], district[3], district[4], district[5],
                                        district[6], district[7], district[8], district[9], district[10], district[12],
                                        district[13], district[15], row[6]])
                else:
                    print('Loan ' + row[0] + ': Outlier detected, skiping')

def normalize_train_data():
    clean_loans = pd.read_csv('./files/loan_train_clean.csv', delimiter=';')
    clean_loans.columns = ['loan_id', 'amount', 'duration', 'dist. no. of inhabitants', 'dist. no. of municipalities with inhabitants < 499', 'dist. no. of municipalities with inhabitants 500-1999', 'dist. no. of municipalities with inhabitants 2000-9999', 'dist. no. of municipalities with inhabitants >10000', 'dist. no. of cities', 'dist. ratio of urban inhabitants', 'dist. average salary', 'dist. unemploymant rate', 'dist. no. of enterpreneurs per 1000 inhabitants', 'dist. no. of commited crimes', 'status']
    train_amount = clean_loans.amount
    amount_matrix = train_amount.values.reshape(-1,1)
    normalized = preprocessing.MinMaxScaler()
    normalized_x_train = normalized.fit_transform(amount_matrix)
    return normalized_x_train

def normalize_test_data():
    test_loans = pd.read_csv('./files/loan_test.csv', delimiter=';')
    test_loans.columns = ['loan_id', 'account_id', 'date', 'amount','duration', 'payments', 'status']
    test_amount = test_loans.amount
    amount_matrix = test_amount.values.reshape(-1,1)
    normalized = preprocessing.MinMaxScaler()
    normalized_x_test = normalized.fit_transform(amount_matrix)
    return normalized_x_test

# arrange_data()
# normalize_data()
