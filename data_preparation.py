
import csv
import pandas as pd
from pandas import Series, DataFrame
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
import data_understanding as du

# column headers for final training / testing data
col_names = ['loan_id', 'amount', 'payments',
    'dist. no. of inhabitants', 'dist. average salary', 'dist. unemploymant rate 96', 'dist. no. of commited crimes 96',
    'status']

# Generates new development csv with all relevant data from most csv's
def arrange_complete_data(train):
    attr_data = du.analyse_data()
    clean_data(attr_data)
    loan_path, card_path, transaction_path = '', '', ''

    if train:
        loan_path = './files/loan_train.csv'
        card_path = './files/card_train.csv'
        transaction_path = './files/trans_train.csv'
    else:
        loan_path = './files/loan_test.csv'
        card_path = './files/card_test.csv'
        transaction_path = './files/trans_test.csv'
    
    with open(loan_path) as loans, open('./files/complete_data.csv', 'w', newline='') as complete_data_file, open('./files/account.csv') as accounts, open(card_path) as cards, open('./files/district.csv') as districts, open('./files/disp_clean.csv') as dispositions, open(transaction_path) as transactions: 
        loan_reader = csv.reader(loans, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        dist_reader = csv.reader(districts, delimiter=';')
        disp_reader = csv.reader(dispositions, delimiter=';')
        cards_reader = csv.reader(cards, delimiter=';')
        complete_data_writer = csv.writer(complete_data_file, delimiter=';')
        next(loan_reader)

        header = col_names.copy()

        if not train:
            header.pop()

        complete_data_writer.writerow(header)

        for loan in loan_reader:
            acc_id = int(loan[1])
            account = du.get_account(accounts, acc_reader, acc_id)

            if len(account) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - ACCOUNT NOT FOUND FOR ID ' + str(acc_id))
                    return

            dist_id = int(account[1])
            district = du.get_district(districts, dist_reader, dist_id)

            if len(district) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - DISTRICT NOT FOUND FOR ID ' + str(dist_id))
                    return

            # choose only relevant district data
            dist_data = [district[3], district[10], district[12], district[15]]

            acc_dispositions = du.get_dispositions(dispositions, disp_reader, acc_id)

            if len(acc_dispositions) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - DISPOSITION(S) NOT FOUND FOR ACCOUNT ' + str(acc_id))
                    return
                    
            # Transactions - Add avg monthly balance and avg transaction value for account
            # Transactions - Current balance based on last transaction before loan
            data_row = [loan[0], loan[3], loan[5]]
            data_row.extend(dist_data)

            if train:
                data_row.append(loan[6])

            complete_data_writer.writerow(data_row)

# One hot encodes a single data piece given the possible set of labels 
def one_hot_encode(labels, data):
    i = 0
    encoded = []

    for label in labels:
        if data == label:
            encoded.append(1)

            for j in range(len(labels) - 1 - i):
                encoded.append(0)
            break
        else:
            encoded.append(0)
            i += 1

    return encoded

# Splits development csv data in two sets: training (2/3, equal number of cases) and testing (1/3)
# ATTENTION: sklearn's train_test_split function should be used instead when possible
def arrange_train_test_data(attr_data):
    with open('./files/complete_data.csv') as complete_data_file, open('./files/train.csv', 'w', newline='') as train_file, open('./files/test.csv', 'w', newline='') as test_file:
        complete_data_reader = csv.reader(complete_data_file, delimiter=';')
        train_writer = csv.writer(train_file, delimiter=';')
        test_writer = csv.writer(test_file, delimiter=';')

        total_loans = attr_data['loan_status_appr'] + attr_data['loan_status_rej']
        train_no = (total_loans * 2) // 3
        test_no = total_loans - train_no
        train_data = []
        train_data_status = [0, 0] # Approved / rejected loan status count
        i = 1

        next(complete_data_reader)

        for dev_row in complete_data_reader:
            if i <= train_no:
                # Train data is not immediately written to csv because it needs to be balanced first
                train_data.append(dev_row)

                # status must always be last attribute
                if int(dev_row[len(dev_row) - 1]) == 0:
                    train_data_status[0] += 1
                else:
                    train_data_status[1] += 1
            else:
                # test data can be written to csv since it must not be altered
                test_writer.writerow(dev_row)
                     
            i += 1

        # make it so there's an equal number of positive and negative training cases
        max_cases = min(train_data_status)
        train_data_status = [0, 0]

        for train_row in train_data:
            status = int(train_row[len(train_row) - 1])

            if status == 0 and train_data_status[0] < max_cases:
                train_data_status[0] += 1
                train_writer.writerow(train_row)
            elif status == 1 and train_data_status[1] < max_cases:
                train_data_status[1] += 1
                train_writer.writerow(train_row)
                        
# Copies and changes some csv data to new, 'cleaned' files
def clean_data(attr_data):
    clean_dispositions(attr_data)

# Copies dispositions complete records and changes type to binary form
def clean_dispositions(attr_data):
    with open('./files/disp.csv') as dispositions, open('./files/disp_clean.csv', 'w', newline='') as dispositions_clean:
        disp_reader = csv.reader(dispositions, delimiter=';')
        disp_writer = csv.writer(dispositions_clean, delimiter=';')
        disp_writer.writerow(next(disp_reader))

        for disp in disp_reader:
            if len(disp) == 4:
                disp_type = disp[3]

                if disp_type == 'OWNER':
                    disp[3] = 1
                elif disp_type == 'DISPONENT':
                    disp[3] = 0

                disp_writer.writerow(disp)

# Copies loans training complete records and changes the status attribute to binary form
def clean_loans(attr_data):
    with open('./files/loan_train.csv') as loans, open('./files/loan_train_clean.csv', 'w', newline='') as loans_new:
        loan_reader = csv.reader(loans, delimiter=';')
        loan_writer = csv.writer(loans_new, delimiter=';')
        loan_writer.writerow(next(loan_reader))

        for row in loan_reader:
            if len(row) == 7:
                status = int(row[6])
                pred_attrs = {'amount': int(row[3]), 'duration': int(row[4]), 'payments': int(row[5])}
                
                # convert to standard binary classification (-1 is positive class)
                if status == -1:
                    row[6] = 1
                elif status == 1:
                    row[6] = 0

                loan_writer.writerow(row)

def normalize_list(list):
    matrix = list.reshape(-1,1)
    normalized = preprocessing.MinMaxScaler()
    normalized_x = normalized.fit_transform(matrix)
    return normalized_x

def create_normalized_matrix(amount, duration):
    norm_matrix = []
    count = 0
    for am in amount:
        norm_matrix.append([am[0], duration[count][0]])
        count = count + 1
    return norm_matrix

def normalize_train_data():
    clean_loans = pd.read_csv('./files/loan_train_clean.csv', delimiter=';')
    clean_loans.columns = ['loan_id', 'amount', 'duration', 'dist. no. of inhabitants', 'dist. no. of municipalities with inhabitants < 499', 'dist. no. of municipalities with inhabitants 500-1999', 'dist. no. of municipalities with inhabitants 2000-9999', 'dist. no. of municipalities with inhabitants >10000', 'dist. no. of cities', 'dist. ratio of urban inhabitants', 'dist. average salary', 'dist. unemploymant rate', 'dist. no. of enterpreneurs per 1000 inhabitants', 'dist. no. of commited crimes', 'status']

    normalized_amount = normalize_list(clean_loans.amount.values)
    normalized_duration = normalize_list(clean_loans.duration.values)
    normalized_x_train = create_normalized_matrix(normalized_amount, normalized_duration)
    return normalized_x_train

def normalize_test_data():
    test_loans = pd.read_csv('./files/loan_test.csv', delimiter=';')
    test_loans.columns = ['loan_id', 'account_id', 'date', 'amount','duration', 'payments', 'status']
    normalized_amount = normalize_list(test_loans.amount.values)
    normalized_duration = normalize_list(test_loans.duration.values)
    normalized_x_test = create_normalized_matrix(normalized_amount, normalized_duration)
    return normalized_x_test

def main():
    arrange_complete_data(True)
    # arrange_data()
    # normalize_data()

if __name__ == '__main__':
    main()