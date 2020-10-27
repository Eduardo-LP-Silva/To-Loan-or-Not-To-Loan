import csv
import argparse
import operator
import datetime
import itertools
from matplotlib.pyplot import axis, colormaps, xticks
from numpy.core.defchararray import count
from numpy.core.fromnumeric import sort
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import datetime
from datetime import date
import statistics

plt.rcParams['font.size'] = 8.0

# Data to be passed to preparation
attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'trans_op_mode': '', 'trans_k_mode': ''}

# Analyses csv's data and produces respective statistics
def analyse_data(clients=False, detailed=False):
    calc_missing_values()
    analyse_loans()
    analyse_accounts(detailed)
    analyse_cards()
    analyse_dispositions()
    analyse_districts()
    analyse_transactions()

    if clients:
        analyse_clients()

    return attr_data

# Analyses the transactions csv and produces related statistics and metrics
def analyse_transactions():
    transactions = pd.read_csv('./files/trans_train.csv', sep=';', header=0, index_col=False, low_memory=False)
    transactions_test = pd.read_csv('./files/trans_test.csv', sep=';', header=0, index_col=False, low_memory=False)
    transactions['operation'].fillna('Missing', inplace=True)
    transactions['k_symbol'].replace('^\s+$', 'Missing', regex=True, inplace=True)
    transactions['k_symbol'].fillna('Missing', inplace=True)
    trans_attrs = {'amount': transactions['amount'].tolist(), 'balance': transactions['balance'].tolist()}

    trans_missing_op = transactions[transactions['operation'] == 'Missing']
    trans_missing_k = transactions[transactions['k_symbol'] == 'Missing']
    trans_household_k = transactions[transactions['k_symbol'] == 'household']
    trans_date_years = pd.Series([parse_date(str(date))[0] for date in transactions['date']])
    trans_test_date_years = pd.Series([parse_date(str(date))[0] for date in transactions_test['date']])

    plot_bar(transactions['type'], 'Transaction Types')
    plot_bar(transactions['operation'], 'Transaction Operations')
    plot_bar(transactions['k_symbol'], 'Transaction K Symbols')
    plot_bar(trans_missing_op['type'], 'Transaction Missing Operation Types')
    plot_bar(trans_missing_op['k_symbol'], 'Transaction Missing Operation K Symbols')
    plot_bar(trans_missing_k['type'], 'Transaction Missing K Types')
    plot_bar(trans_missing_k['operation'], 'Transaction Missing K Operations')
    plot_bar(transactions[transactions['k_symbol'] == 'interest credited']['operation'], 'Transaction Interest K Operations')
    plot_bar(transactions[transactions['operation'] == 'withdrawal in cash']['type'], 'Transaction Withdrawal in Cash Operation Types')
    plot_bar(trans_household_k['type'], 'Transaction Household K Types')
    plot_bar(trans_household_k['operation'], 'Transaction Household K Operations')
    plot_bar(trans_date_years, 'Transactions Date (Years)', stacked=False, double_precision=False)
    plot_bar(trans_test_date_years, 'Transactions (Test) Date (Years)', stacked=False, double_precision=False)
    plot_box(trans_attrs, 'Transaction', save_thresholds=True)

    attr_data['trans_op_mode'] = transactions['operation'].value_counts().idxmax()
    attr_data['trans_k_mode'] = transactions['k_symbol'].value_counts().idxmax()

# Analyses the clients csv and produces the number of accounts per client plot
def analyse_clients():
    clients = pd.read_csv('./files/client.csv', sep=';', header=0, index_col=False)
    dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
    accounts = pd.read_csv('./files/account.csv', sep=';', header=0, index_col=False)

    client_dobs = clients['birth_number'].values

    genders = pd.Series([get_client_gender(str(birthdate)) for birthdate in client_dobs])
    client_dob_years = pd.Series([parse_date(str(birthdate))[0] for birthdate in client_dobs])
    client_account_no = pd.Series([len(get_client_accounts(client_id, dispositions, accounts)) 
        for client_id in clients['client_id'].values])

    plot_bar(genders, 'Clients Gender', rename_cols={0: 'Male', 1: 'Female'})
    plot_bar(client_account_no, 'Accounts per Client')
    plot_bar(client_dob_years, 'Clients Year of Birth', stacked=False, double_precision=False)

# Analyses districts csv and produces box charts for each relevant attribute
def analyse_districts():
    with open('./files/district.csv') as districts:
        attrs = {'inh': [], 'mun_lt_499': [], 'mun_500_1999': [], 'mun_2000_9999': [], 'mun_gt_10000': [],
            'cities': [], 'urban_inh_r': [], 'avg_salary': [], 'unemp_95': [], 'unemp_96': [], 'enterp_p_1000': [],
            'crimes_95': [], 'crimes_96': []}
        dist_reader = csv.reader(districts, delimiter=';')
        next(dist_reader)

        for row in dist_reader:
            if len(row) == 16:
                attrs['inh'].append(int(row[3]))
                attrs['mun_lt_499'].append(int(row[4]))
                attrs['mun_500_1999'].append(int(row[5]))
                attrs['mun_2000_9999'].append(int(row[6]))
                attrs['mun_gt_10000'].append(int(row[7]))
                attrs['cities'].append(int(row[8]))
                attrs['urban_inh_r'].append(float(row[9]))
                attrs['avg_salary'].append(int(row[10]))

                try:
                    attrs['unemp_95'].append(float(row[11]))
                except ValueError:
                    pass

                attrs['unemp_96'].append(float(row[12]))
                attrs['enterp_p_1000'].append(int(row[13]))

                try:
                    attrs['crimes_95'].append(float(row[14]))
                except ValueError:
                    pass

                attrs['crimes_96'].append(int(row[15]))

        attr_data['dist_avg_95_ur'] =  sum(attrs['unemp_95']) / len(attrs['unemp_95'])
        attr_data['dist_avg_95_cr'] =  sum(attrs['crimes_95']) / len(attrs['crimes_95'])
        plot_box(attrs, 'Districts')

# Analyses dispositions csv and produces disposition type pie chart
def analyse_dispositions():
    dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
    plot_bar(dispositions['type'], 'Disposition Types')

# Analyses training cards csv and produces card type pie chart
def analyse_cards():
    cards = pd.read_csv('./files/card_train.csv', sep=';', header=0, index_col=False)
    cards_test = pd.read_csv('./files/card_test.csv', sep=';', header=0, index_col=False)

    cards_issued_date = pd.Series([parse_date(str(date))[0] for date in cards['issued']])
    cards_test_issued_date = pd.Series([parse_date(str(date))[0] for date in cards_test['issued']])

    plot_bar(cards['type'], 'Card Type')
    plot_bar(cards_issued_date, 'Cards Issued Date (Years)', stacked=False, double_precision=False)
    plot_bar(cards_test_issued_date, 'Cards (Test) Issued Date (Years)', stacked=False, double_precision=False)

# Analyses accounts csv and produces various statistics regarding them
def analyse_accounts(detailed):
    accounts = pd.read_csv('./files/account.csv', sep=';', header=0, index_col=False)
    accs_date = pd.Series([parse_date(str(date))[0] for date in accounts['date']])

    plot_bar(accounts['frequency'], 'Account Issuance Frequency')
    plot_bar(accs_date, 'Accounts Creation Date (Years)', stacked=False, double_precision=False)

    if detailed:
        dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
        cards = pd.read_csv('./files/card_train.csv', sep=';', header=0, index_col=False)
        loans = pd.read_csv('./files/loan_train.csv', sep=';', header=0, index_col=False)

        acc_dispositions = [get_dispositions(dispositions, acc_id) for acc_id in accounts['account_id'].values]
        acc_loans = [get_account_loans(loans, acc_id) for acc_id in accounts['account_id'].values]
        disps_no = pd.Series([len(acc_disps) for acc_disps in acc_dispositions])
        acc_owner_card = pd.Series([get_owner_card(cards, acc_disps) for acc_disps in acc_dispositions])
        acc_cards_no = pd.Series([sum(get_card_types_no(cards, account_dispositions).values()) 
            for account_dispositions in acc_dispositions])
        acc_loan_no = pd.Series([len(account_loans) for account_loans in acc_loans])

        plot_bar(disps_no, 'Account Dispositions No')
        plot_bar(acc_owner_card, 'Account Owner Card Type')
        plot_bar(acc_cards_no, 'Account Card Number')
        plot_bar(acc_loan_no, 'Loans Per Account')

# Analyses training loans csv and produces relevant attributes box chart and loan status pie chart
def analyse_loans():
    attrs = {'amount': [], 'duration': [], 'payments': []}
    status_disp = pd.DataFrame([[0, 0], [0, 0]], columns=['1', '2'], index=['Unsuccessful', 'Successful'])
    loans = pd.read_csv('./files/loan_train.csv', sep=';', header=0, index_col=False)
    loans_test = pd.read_csv('./files/loan_test.csv', sep=';', header=0, index_col=False)
    dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
    clients = pd.read_csv('./files/client.csv', sep=';', header=0, index_col=False)
    age_dist = {'0-19':0, '20-29':0,'30-39':0, '40-49':0, '50-59':0, '60-69':0, '70+':0}
    gender_dist = {0: 0, 1: 0}

    for _, loan in loans.iterrows():
        if len(loan) == 7:
            status = loan['status']
            row = -1

            acc_id = loan['account_id']
            owner = get_acc_owner(acc_id, dispositions, clients)
            owner_loan_age = calculate_loan_client_age(str(owner['birth_number']), str(loan['date']))
            gender = get_client_gender(str(owner['birth_number']))
            gender_dist[gender] += 1

            if owner_loan_age < 20:
                age_dist['0-19'] += 1
            elif owner_loan_age >= 20 and owner_loan_age < 30:
                age_dist['20-29'] += 1
            elif owner_loan_age >= 30 and owner_loan_age < 40:
                age_dist['30-39'] += 1
            elif owner_loan_age >= 40 and owner_loan_age < 50:
                age_dist['40-49'] += 1
            elif owner_loan_age >= 50 and owner_loan_age < 60:
                age_dist['50-59'] += 1
            elif owner_loan_age >= 60 and owner_loan_age < 70:
                age_dist['60-69'] += 1
            elif owner_loan_age >= 70:
                age_dist['70+'] += 1

            for attr_key in attrs.keys():
                attrs[attr_key].append(loan[attr_key])

            disp_no = str(len(dispositions[dispositions['account_id'] == loan['account_id']]))

            if status == 1:
                attr_data['loan_status_appr'] += 1
                row = 'Successful'
            else:
                attr_data['loan_status_rej'] += 1
                row = 'Unsuccessful'

            status_disp.at[row, disp_no] += 1

    age_dist_series = pd.DataFrame(columns=age_dist.keys())
    age_dist_series.loc[0] = age_dist.values()

    total_loans = attr_data['loan_status_appr'] + attr_data['loan_status_rej']
    suc_loan_percent = attr_data['loan_status_appr'] / total_loans * 100
    unsuc_loan_percent = attr_data['loan_status_rej'] / total_loans * 100

    loan_status = pd.DataFrame(columns=['Successful', 'Unsuccessful'])
    loan_status.loc[0] = [suc_loan_percent, unsuc_loan_percent]

    gender_total = gender_dist[0] + gender_dist[1]
    male_percent = gender_dist[0] / gender_total * 100
    female_percent = gender_dist[1] / gender_total * 100

    gender_df = pd.DataFrame(columns=['Male', 'Female'])
    gender_df.loc[0] = [male_percent, female_percent]

    loans_date = pd.Series([parse_date(str(date))[0] for date in loans['date']])
    loans_test_date = pd.Series([parse_date(str(date))[0] for date in loans_test['date']])

    plot_bar(gender_df, 'Loan Genders', count_values=False)
    plot_bar(loans_date, 'Loans Date (Years)', stacked=False, double_precision=False)
    plot_bar(loans_test_date, 'Loans (Test) Date (Years)', stacked=False, double_precision=False)
    plot_bar(status_disp[['1', '2']], 'Disposition No. & Status', single_col=False, count_values=False, 
        double_precision=False)
    plot_bar(age_dist_series, 'Clients Age at Loan Request', count_values=False, double_precision=False)
    plot_bar(loan_status, 'Loan Status', count_values=False)
    plot_box(attrs, 'Loans', save_thresholds=True)

# Calculates (necessary) missing and / or not loan linked values
def calc_missing_values():
    with open('./files/loan_train.csv') as loans, open('./files/card_train.csv') as cards:
        loans_reader = csv.reader(loans, delimiter=';')
        cards_reader = csv.reader(cards, delimiter=';')
        accounts = pd.read_csv('./files/account.csv', sep=';', header=0, index_col=False)
        districts = pd.read_csv('./files/district.csv', sep=';', header=0, index_col=False)
        dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
        next(loans_reader)

        missing_vals_count = {'missing_districts': 0, 'missing_loans': 0, 'missing_dispositions': 0,
            'missing_accounts': 0}
        missing_vals = {}

        for row in loans_reader:
            if len(row) == 7:
                acc_id = int(row[1])
                missing_vals = {'missing_districts': True, 'missing_dispositions': True, 'missing_accounts': True}
                account = get_account(accounts, acc_id)

                if len(account) > 0:
                    missing_vals['missing_accounts'] = False
                    dist_id = int(account[1])
                    district = get_district(districts, dist_id)
                    dispositions_list = get_dispositions(dispositions, acc_id)

                    if len(district) > 0:
                        missing_vals['missing_districts'] = False

                    if len(dispositions_list) > 0:
                        missing_vals['missing_dispositions'] = False
            else:
                missing_vals_count['missing_loans'] += 1

            for key in missing_vals.keys():
                if missing_vals[key]:
                    missing_vals_count[key] += 1

        print('\n--- Missing Required ID Matches ---')
        for key in missing_vals_count.keys():
            print(key + ': ' + str(missing_vals_count[key]))
        print('\n')

# Calculates the average monthly income associated with an account, given its transactions
def calc_avg_monthly_income(acc_transactions):
    acc_transactions = acc_transactions.sort_values(by=['date'])
    monthly_balances = [[0]]
    i = 0
    previous_date = parse_date(str(acc_transactions.iloc[0]['date'])) if len(acc_transactions) > 0 else None

    for _, trans in acc_transactions.iterrows():

        if trans['type'] == 'credit' or trans['operation'] == 'credit in cash' or trans['operation'] == 'collection from another bank' or trans['k_symbol'] == 'old-age pension':
            date = parse_date(str(trans['date']))

            if date[0] != previous_date[0] or date[1] != previous_date[1]:
                monthly_balances[i] = np.mean(np.array(monthly_balances[i]))
                monthly_balances.append([trans['balance']])
                i += 1
            else:
                monthly_balances[i].append(trans['balance'])

            previous_date = date

    if type(monthly_balances[i]) == list:
        monthly_balances[i] = np.mean(np.array(monthly_balances[i]))

    return sum(monthly_balances) / len(monthly_balances)

# Parses a YYMMDD date to tuple format
def parse_date(date):
    year = '19' + date[:2]
    month = date[2:4]
    day = date[4:]

    return (int(year), int(month), int(day))

# Returns the last transactions of an account before a given date, from most recent to oldest
def get_acc_last_transactions(transactions, date):
    d1 = datetime.datetime(*date)
    prior_trans = []

    for index, trans in transactions.iterrows():
        d2 = datetime.datetime(*parse_date(str(trans['date'])))

        if d2 < d1:
            prior_trans.append(trans)

    prior_trans.sort(key=operator.attrgetter('date'), reverse=True)

    return prior_trans

# Calculates standard deviation for an account's transactions amount
def get_sd_acc_transactions(transactions):
    return statistics.pstdev(transactions['amount'])

# Calculates standard deviation for an account's balance
def get_sd_acc_balance(transactions):
    return statistics.pstdev(transactions['balance'])

# Returns the transactions associated with an account
def get_acc_transactions(transactions_df, acc_id):
    return transactions_df.loc[transactions_df['account_id'] == acc_id]

# Returns the accounts associated with a given client
def get_client_accounts(client_id, dispositions, accounts):
    accs = []
    client_disps = dispositions[dispositions['client_id'] == client_id]

    for _, disp in client_disps.iterrows():
        acc = accounts[accounts['account_id'] == disp['account_id']].iloc[0]
        accs.append(acc)

    return accs

# Returns the client data given a client_id
def get_client(clients_file, clients_reader, client_id):
    clients_file.seek(0)
    next(clients_file)

    for client in clients_reader:
        if len(client) == 4 and int(client[0]) == int(client_id):
            return client

    return []

# Returns the owner's info of a given account
def get_acc_owner(account_id, dispositions, clients):
    client_id = dispositions[dispositions['account_id'] == account_id]['client_id'].iloc[0]

    return clients[clients['client_id'] == client_id].iloc[0]

# Returns the age of a client
def get_client_age(birthdate):
    day = int(birthdate[4:])
    month = int(birthdate[2:4])
    year = int(birthdate[:2])

    if int(month) > 12:
        month -= 50

    today = date.today()
    return today.year - (1900 + year) - ((today.month, today.day) < (month, day))

# Returns the gender of a client
def get_client_gender(birthdate):
    month = birthdate[2:4]

    if int(month) > 12:
        return 1 # female

    return 0 # male

def normalize_client_dob(client_dob):
    client_dob_month = int(client_dob[2:4])

    if client_dob_month > 12:
        client_dob = client_dob[:2] + str(client_dob_month - 50).zfill(2) + client_dob[4:]

    return client_dob

# Returns the age a certain client had when he asked for the loan, given the client's date of birth and the loan date
def calculate_loan_client_age(client_dob, loan_date):
    loan_date = parse_date(loan_date)
    client_dob = parse_date(normalize_client_dob(client_dob))

    return loan_date[0] - client_dob[0]

# Returns the loans associated with a given account
def get_account_loans(loans, acc_id):
    return loans[loans['account_id'] == acc_id]

# Returns the (junior_card_no, classic_card_no, gold_card_no) associated with an account given its associated dispositions
def get_card_types_no(cards, acc_dispositions):
    card_types = {'junior': 0, 'classic': 0, 'gold': 0}

    for _, disposition in acc_dispositions.iterrows():
        disp_id = disposition['disp_id']
        c_types = cards[cards['disp_id'] == disp_id]['type'].values

        for c_type in c_types:
            card_types[c_type] += 1

    return card_types

# Returns the (client_id, disposition_id) of the owner of an account given the ASSOCIATED dispositions
def get_account_owner_info(dispositions):
    acc_owner = dispositions[dispositions['type'] == 'OWNER']
    acc_owner_disp = acc_owner if len(acc_owner) > 0 else dispositions[dispositions['type'] == 1]
    acc_owner_disp = acc_owner_disp.iloc[0]

    return (acc_owner_disp.at['client_id'], acc_owner_disp.at['disp_id'])

# Returns the card type (or none) of the owner of an account given the ASSOCIATED dispositions
def get_owner_card(cards, acc_dispositions):
    owner = get_account_owner_info(acc_dispositions)
    owner_card = cards[cards['disp_id'] == owner[1]]

    if len(owner_card) > 0:
        return owner_card.iloc[0]['type']
    else:
        return 'none'

# Returns the dispositions associated with an account given an account id
def get_dispositions(dispositions, acc_id):
    return dispositions[dispositions['account_id'] == acc_id]

# Returns a district given a district id
def get_district(districts, dist_id):
    return districts[districts['code '] == dist_id].iloc[0]

# Returns an account given an account id
def get_account(accounts, acc_id):
    return accounts[accounts['account_id'] == acc_id].iloc[0]

# Plots a scatter plot
def plot_cat(data, x, y, hue, title, palette=None):
    if palette:
        plot = sn.catplot(x=x, y=y, hue=hue, data=data, palette=palette)
    else:
        plot = sn.catplot(x=x, y=y, hue=hue, data=data)
        
    plot.savefig('./figures/%s.png' % title)
    plt.close()
    
# Plots a heatmap corresponding to a correlation matrix
def plot_correlation_matrix(corr_mat):
    plt.figure(figsize=(50, 50))
    sn.heatmap(corr_mat, annot=True)
    plt.savefig('./figures/correlation_matrix.png')
    plt.close()

# Plots a confusion matrix
def plot_confusion_matrix(cm, classes, title):
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
    plt.savefig('./figures/' + title + '_confusion_matrix.png')
    plt.close()

# Draws a stacked bar plot given a pandas dataframe / series
def plot_bar(df, title, stacked=True, count_values=True, single_col=True, double_precision=True, 
    rename_cols=None):
    freq = []

    if count_values:
        if stacked:
            freq = df.value_counts(normalize=True, dropna=False) * 100
        else:
            freq = df.value_counts(dropna=False).sort_index()

        df = freq.to_frame().T

        if rename_cols:
            df.rename(rename_cols, inplace=True, axis='columns')

    axes = None

    if single_col:
        axes = df.plot(kind='bar', stacked=stacked, rot=0, title=title, xticks=[])
    else:
        axes = df.plot(kind='bar', stacked=stacked, rot=0, title=title)
    
    for rect in axes.patches:
        if rect.get_height() > 0:
            label = ''
            x = 0
            color = ''
            font_weight = ''

            if double_precision:
                label = '%.2f' % rect.get_height() + '%'
            else:
                label = '%i' % int(rect.get_height())

            if single_col and stacked:
                x = rect.get_x() + rect.get_width() * 1.2
                color = 'black'
                font_weight = 'regular'
            else:
                x = rect.get_x() + rect.get_width() / 2.
                color = 'white'
                font_weight = 'bold'

            axes.text(x, rect.get_y() + rect.get_height() / 2., label, 
                ha = 'center', va = 'center', fontweight=font_weight, color=color)

    fig = axes.get_figure()
    fig.savefig('./figures/%s.png' % title)
    plt.close()

# Draws a box chart based on a set of numerical attributes
def plot_box(attrs, title, save_thresholds=False):
    for attr in attrs.keys():
        attr_array = attrs[attr]
        r = plt.boxplot(attr_array, vert=False)
        minThresh = r['whiskers'][0].get_xdata()[1]
        maxThresh = r['whiskers'][1].get_xdata()[1]

        thresholds = (minThresh, maxThresh)
        attr_name = title + '_' + attr

        if save_thresholds:
            attr_data[attr_name + '_thresh'] = thresholds

        print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)),
            'Avg: ' + str(sum(attr_array) // len(attr_array)), 'Min.Thresh: ' +  str(thresholds[0]),
            'Max.Thresh ' + str(thresholds[1]), sep=' | ')

        plt.title(title + ' - ' + attr)
        plt.savefig('./figures/' + attr_name + '_box.png')
        plt.close()

# Draws a pie chart based on a set of sizes / numerical data and respective labels
def plot_pie(sizes, labels, title):
    _, loan_chart = plt.subplots()
    loan_chart.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.tight_layout(pad=6.0)
    loan_chart.axis('equal')
    plt.title(title)
    plt.savefig('./figures/' +  title + '.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Data Analysis')
    parser.add_argument('-c', dest='clients', action='store_true', default=False, help='Analyse Clients')
    parser.add_argument('-d', dest='detailed', action='store_true', default=False, help='Detailed Account Analysis')
    args = parser.parse_args()

    analyse_data(args.clients, args.detailed)

if __name__ == '__main__':
    main()
