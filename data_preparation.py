import csv
import argparse
from os import path
import pandas as pd
import numpy as np
from sklearn import preprocessing
import data_understanding as du

# Dictionary with the values for one row of the complete data set
complete_data_row = {}

# Generates new development csv with all relevant data from most csv's
def arrange_complete_data(train, clean=False, outlier_removal=False):
    complete_data_row.clear()
    attr_data = du.analyse_data()

    if clean or outlier_removal or not (path.isfile('./files/client_clean.csv') 
        and path.isfile('./files/disp_clean.csv') and path.isfile('district_clean.csv') 
        and path.isfile('loan_train_clean.csv') and path.isfile('trans_test_clean.csv') 
        and path.isfile('trans_train_clean.csv')):
        clean_data(attr_data, outlier_removal)

    loan_path, card_path, transaction_path = '', '', ''

    if train:
        loan_path = './files/loan_train_clean.csv'
        card_path = './files/card_train.csv'
        transaction_path = './files/trans_train_clean.csv'
    else:
        loan_path = './files/loan_test.csv'
        card_path = './files/card_test.csv'
        transaction_path = './files/trans_test_clean.csv'

    with open(loan_path) as loans, open('./files/complete_data.csv', 'w', newline='') as complete_data_file:
        loan_reader = csv.reader(loans, delimiter=';')
        complete_data_writer = csv.writer(complete_data_file, delimiter=';')
        accounts = pd.read_csv('./files/account.csv', sep=';', header=0, index_col=False)
        districts = pd.read_csv('./files/district_clean.csv', sep=';', header=0, index_col=False)
        transactions = pd.read_csv(transaction_path, sep=';', header=0, index_col=False, low_memory=False)
        dispositions = pd.read_csv('./files/disp_clean.csv', sep=';', header=0, index_col=False)
        clients = pd.read_csv('./files/client_clean.csv', sep=';', header=0, index_col=False)
        cards = pd.read_csv(card_path, sep=';', header=0, index_col=False)

        next(loan_reader)

        for loan in loan_reader:
            fill_loan_info(loan)

            acc_id = int(loan[1])
            loan_date = loan[2]

            account = du.get_account(accounts, acc_id)

            if len(account) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - ACCOUNT NOT FOUND FOR ID ' + str(acc_id))
                    return

            fill_account_info(account, loan_date)

            dist_id = int(account[1])
            district = du.get_district(districts, dist_id)

            if len(district) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - DISTRICT NOT FOUND FOR ID ' + str(dist_id))
                    return

            fill_district_info(district)
            acc_dispositions = du.get_dispositions(dispositions, acc_id)

            if len(acc_dispositions) == 0:
                if train:
                    continue
                else:
                    print('ERROR IN TESTING - DISPOSITION(S) NOT FOUND FOR ACCOUNT ' + str(acc_id))
                    return

            fill_disposition_info(acc_dispositions)
            fill_client_info(clients, acc_dispositions, acc_id, loan_date)
            fill_card_info(cards, acc_dispositions)
            fill_transaction_info(transactions, acc_id, loan_date)

            if train:
                complete_data_row['status'] = loan[6]

            # Hasn't started writing yet
            if complete_data_file.tell() == 0:
                complete_data_writer.writerow(complete_data_row.keys())

            complete_data_writer.writerow(complete_data_row.values())
        
        if train:
            plot_complete_data_graphs()

def plot_complete_data_graphs():
    complete_data = pd.read_csv('./files/complete_data.csv', sep=';', header=0, index_col=False).astype({'status': 'category'})
    complete_data.replace({'status': {-1: 'Unsuccessful', 1: 'Successful'}, 'gender': {0: 'Male', 1: 'Female'}}, inplace=True)
    status_palette = {'Successful': 'green', 'Unsuccessful': 'red'}

    du.plot_scatter(complete_data, 'avg_k_missing', 'avg_monthly_income', 
        'avg_k_missing_monthly_income_avg_withdrawals', hue='status', size='avg_withdrawals', 
        palette=status_palette)

    du.plot_scatter(complete_data, 'avg_credits_cash', 'avg_other_bank_collections', 
        'avg_credits_cash_avg_other_bank_collections', hue='status', palette=status_palette)

    du.plot_scatter(complete_data, "no. of commited crimes '96", 'average salary', 
        'crimes96_avg_salary_cities', hue='status', size='no. of cities', palette=status_palette)

# Removes correlated attributes
def remove_correlated_attributes(x, thresh=0.8):
    corr_mat = x.corr()
    corr_feats = set()
    print('\n--- Correlated Features ---')

    for i in range(len(corr_mat.columns)):
        for j in range(i):
            corr_val = corr_mat.iloc[i, j]
            col_name = corr_mat.columns[i]

            if abs(corr_val) > thresh:
                corr_feats.add(col_name)

                if col_name in complete_data_row:
                    complete_data_row.pop(col_name)

                print('%s & %s: %.2f' % (col_name, corr_mat.columns[j], corr_val))
                break

    du.plot_correlation_matrix(corr_mat, 'correlation_matrix')

    return x.drop(labels=corr_feats, axis=1), corr_feats

def fill_loan_info(loan):
    complete_data_row['loan_id'] = loan[0]
    complete_data_row['amount'] = loan[3]
    complete_data_row['duration'] = loan[4]
    complete_data_row['payments'] = loan[5]

def fill_account_info(account, loan_date):
    ld = du.parse_date(loan_date)
    complete_data_row['account_age'] = ld[0] - du.parse_date(str(account['date']))[0]

def fill_district_info(district):
    complete_data_row['no. of inhabitants'] = district['no. of inhabitants']
    complete_data_row['no. of municipalities with inhabitants < 499'] = district['no. of municipalities with inhabitants < 499 ']
    complete_data_row['no. of municipalities with inhabitants 500-1999'] = district['no. of municipalities with inhabitants 500-1999']
    complete_data_row['no. of municipalities with inhabitants 2000-9999'] = district['no. of municipalities with inhabitants 2000-9999 ']
    complete_data_row['no. of municipalities with inhabitants >10000'] = district['no. of municipalities with inhabitants >10000 ']
    complete_data_row['no. of cities'] = district['no. of cities ']
    complete_data_row['ratio of urban inhabitants'] = district['ratio of urban inhabitants ']
    complete_data_row['average salary'] = district['average salary ']
    complete_data_row["unemploymant rate '95"] = district["unemploymant rate '95 "]
    complete_data_row["unemploymant rate '96"] = district["unemploymant rate '96 "]
    complete_data_row['no. of enterpreneurs per 1000 inhabitants'] = district['no. of enterpreneurs per 1000 inhabitants ']
    complete_data_row["no. of commited crimes '95"] = district["no. of commited crimes '95 "]
    complete_data_row["no. of commited crimes '96"] = district["no. of commited crimes '96 "]

def fill_disposition_info(acc_dispositions):
    complete_data_row['disposition_no'] = len(acc_dispositions)
    pass

def fill_client_info(clients, acc_dispositions, acc_id, loan_date):
    owner = du.get_acc_owner(acc_id, acc_dispositions, clients)
    loan_owner_age = du.calculate_loan_client_age(str(owner['birth_number']), loan_date)

    complete_data_row['age'] = loan_owner_age
    complete_data_row['gender'] = owner['gender']

def fill_card_info(cards, acc_dispositions):
    card_types = du.get_card_types_no(cards, acc_dispositions)
    complete_data_row['card_no'] = sum(card_types.values())
    complete_data_row['junior_card_no'] = card_types['junior']
    complete_data_row['classic_card_no'] = card_types['classic']
    complete_data_row['gold_card_no'] = card_types['gold']

def fill_transaction_info(transactions, acc_id, loan_date):
    ld = du.parse_date(loan_date)
    acc_trans = du.get_acc_transactions(transactions, acc_id)
    last_trans = du.get_acc_last_transactions(acc_trans, ld)

    sd_trans = du.get_sd_acc_transactions(acc_trans)
    negative_balance_no = len([trans for trans in last_trans if trans['balance'] <= 0])

    attrs = {
        'balance': [trans['balance'] for trans in last_trans],
        'withdrawals': [trans['amount'] for trans in last_trans if trans['type'] == 'withdrawal'],
        'withdrawals_cash': [trans['amount'] for trans in last_trans if trans['type'] == 'withdrawal in cash'],
        'credits': [trans['amount'] for trans in last_trans if trans['type'] == 'credit'],
        'withdrawals_cash_op': [trans['amount'] for trans in last_trans if trans['operation'] == 'withdrawal in cash'],
        'remittances': [trans['amount'] for trans in last_trans if trans['operation'] == 'remittance to another bank'],
        'credit_card_withdrawals': [trans['amount'] for trans in last_trans if trans['operation'] == 'credit card withdrawal'],
        'credits_cash': [trans['amount'] for trans in last_trans if trans['operation'] == 'credit in cash'],
        'other_bank_collections': [trans['amount'] for trans in last_trans if trans['operation'] == 'collection from another bank'],
        'k_missing': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'missing'],
        'interest_credited': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'interest credited'],
        'household': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'household'],
        'statement_payments': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'payment for statement'],
        'insurrance_payments': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'insurrance payment'],
        'sanctions': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'sanction interest if negative balance'],
        'pension': [trans['amount'] for trans in last_trans if trans['k_symbol'] == 'old-age pension']
    }

    complete_data_row['transaction_no'] = len(acc_trans)
    complete_data_row['last_balance'] = last_trans[0]['balance']
    complete_data_row['negative_balance_no'] = negative_balance_no
    complete_data_row['standard_deviation_transactions'] = sd_trans
    complete_data_row['avg_monthly_income'] = du.calc_avg_monthly_income(acc_trans)

    for key, value in attrs.items():
        complete_data_row['avg_' + key] = np.mean(value) if len(value) > 0 else 0
        complete_data_row[key + '_no'] = len(value)

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

# Copies and changes some csv data to new, 'cleaned' files
def clean_data(attr_data, outlier_removal=False):
    clean_clients()
    clean_dispositions()
    clean_transactions(attr_data['trans_op_mode'], attr_data['trans_k_mode'], attr_data['Transaction_amount_thresh'],
        attr_data['Transaction_balance_thresh'], outlier_removal=outlier_removal)
    clean_districts(attr_data['dist_avg_95_ur'], attr_data['dist_avg_95_cr'])
    clean_loans(attr_data['Loans_amount_thresh'], attr_data['Loans_duration_thresh'], attr_data['Loans_payments_thresh'],
        outlier_removal=outlier_removal)

# Cleans transactions files
def clean_transactions(op_mode, k_mode, amount_thresh, balance_thresh, outlier_removal=False):
    with open('./files/trans_train.csv') as transactions_train, open('./files/trans_train_clean.csv', 'w', newline='') as transactions_train_clean, open('./files/trans_test.csv') as transactions_test, open('./files/trans_test_clean.csv', 'w', newline='') as transactions_test_clean:
        read_files = [transactions_train, transactions_test]
        write_files = [transactions_train_clean, transactions_test_clean]

        for i in range(len(read_files)):
            reader = csv.reader(read_files[i], delimiter=';')
            writer = csv.writer(write_files[i], delimiter=';')
            writer.writerow(next(reader))

            for trans in reader:
                if len(trans) == 10:
                    amount = float(trans[5])
                    balance = float(trans[6])

                    if (outlier_removal and (amount < amount_thresh[0] or amount > amount_thresh[1]
                        or balance < balance_thresh[0] or balance > balance_thresh[1])):
                        continue

                    if not trans[4] or trans[4].isspace(): # Operation
                        trans[4] = 'interest' # K-Symbol match

                    if not trans[7] or trans[7].isspace(): # K-Symbol
                        trans[7] = 'missing' #or k_mode

                    writer.writerow(trans)


# Replaces missing values in the 95 unemployment and crime rates columns with their average
def clean_districts(avg_95_ur, avg_95_cr):
    with open('./files/district.csv') as districts, open('./files/district_clean.csv', 'w', newline='') as districts_clean:
        dist_reader = csv.reader(districts, delimiter=';')
        dist_writer = csv.writer(districts_clean, delimiter=';')
        dist_writer.writerow(next(dist_reader))

        for dist in dist_reader:
            if len(dist) == 16:
                try:
                    float(dist[11])
                except ValueError:
                    dist[11] = avg_95_ur

                try:
                    float(dist[14])
                except ValueError:
                    dist[14] = avg_95_cr

                dist_writer.writerow(dist)

# Copies dispositions complete records and changes type to binary form
def clean_dispositions():
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

# Copies client ids and districts and adds their normalized date of birth and gender
def clean_clients():
    with open('./files/client.csv') as clients, open('./files/client_clean.csv', 'w', newline='') as clients_clean:
        client_reader = csv.reader(clients, delimiter=';')
        client_writer = csv.writer(clients_clean, delimiter=';')
        header = ['client_id', 'birth_number', 'district_id', 'gender']
        client_writer.writerow(header)
        next(client_reader)

        for client in client_reader:
            if len(client) == 3:
                gender = du.get_client_gender(client[1])
                dob = du.normalize_client_dob(client[1])
                client[1] = dob
                client.append(gender)

                client_writer.writerow(client)

# Copies loans training complete records and removes outliers
def clean_loans(amount_thrs, duration_thrs, payments_thrs, outlier_removal=False):
    with open('./files/loan_train.csv') as loans, open('./files/loan_train_clean.csv', 'w', newline='') as loans_new:
        loan_reader = csv.reader(loans, delimiter=';')
        loan_writer = csv.writer(loans_new, delimiter=';')
        loan_writer.writerow(next(loan_reader))

        for row in loan_reader:
            if len(row) == 7:
                amount = int(row[3])
                duration = int(row[4])
                payments = int(row[5])

                if (outlier_removal and (amount < amount_thrs[0] or amount > amount_thrs[1]
                    or duration < duration_thrs[0] or duration > duration_thrs[1]
                    or payments < payments_thrs[0] or payments > payments_thrs[1])):
                    continue
                
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
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('-c', dest='clean', action='store_true', default=False, help='Clean CSVs')
    parser.add_argument('-o', dest='outlier', action='store_true', default=False, help='Remove Outliers')
    args = parser.parse_args()

    arrange_complete_data(True, args.clean, args.outlier)
    # arrange_data()
    # normalize_data()

if __name__ == '__main__':
    main()
