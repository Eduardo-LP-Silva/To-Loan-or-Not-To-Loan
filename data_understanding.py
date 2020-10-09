import csv
import operator
import datetime
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data to be passed to preparation
attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'missing_loans': 0, 'missing_districts': 0,
    'missing_dispositions': 0, 'missing_accounts': 0, 'frequency_monthly': 0, 'frequency_transactional': 0,
    'frequency_weekly': 0, 'cards_classic': 0, 'cards_junior': 0, 'cards_gold': 0, 'disposition_owner': 0,
    'disposition_disponent': 0}

# Analyses csv's data and produces respective statistics
def analyse_data(clients=False):
    calc_missing_values()
    analyse_loans()
    analyse_accounts(False)
    analyse_cards()
    analyse_dispositions()
    analyse_districts()
    analyse_transactions()
    
    if clients:
        analyse_clients()

    return attr_data

# Analyses the transactions csv and produces related statistics and metrics
def analyse_transactions():
    with open('./files/trans_train.csv') as transactions:
        trans_reader = csv.reader(transactions, delimiter=';')
        trans_types = {}
        trans_operations = {}
        trans_ks = {}
        trans_attrs = {'amount': [], 'balance': []}

        next(trans_reader)

        for trans in trans_reader:
            if len(trans) == 10: 
                if trans[3] in trans_types:
                    trans_types[trans[3]] += 1
                else:
                    trans_types[trans[3]] = 0

                trans_op = trans[4] if trans[4] != '' else 'Missing'

                if trans_op in trans_operations:
                    trans_operations[trans_op] += 1
                else:
                    trans_operations[trans_op] = 0

                trans_k = trans[7] if trans[7] != '' else 'Missing'

                if trans_k in trans_ks:
                    trans_ks[trans_k] += 1
                else:
                    trans_ks[trans_k] = 0

                trans_attrs['amount'].append(float(trans[5]))
                trans_attrs['balance'].append(float(trans[6]))
                
        plot_pie(trans_types.values(), trans_types.keys(), 'Transaction Types')
        plot_pie(trans_operations.values(), trans_operations.keys(), 'Transaction Operations')
        plot_pie(trans_ks.values(), trans_ks.keys(), 'Transaction K Symbols')
        plot_box(trans_attrs, 'Transaction')
        
# Analyses the clients csv and produces the number of accounts per client plot
def analyse_clients():
    with open('./files/client.csv') as clients, open('./files/disp.csv') as dispositions, open('./files/account.csv') as accounts:
        clients_reader = csv.reader(clients, delimiter=';')
        disp_reader = csv.reader(dispositions, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        client_account_no = {}

        next(clients_reader)

        for client in clients_reader:
            if len(client) == 3:
                client_accs = len(get_client_accounts(int(client[0]), dispositions, disp_reader, accounts, acc_reader))

                if client_accs in client_account_no:
                    client_account_no[client_accs] += 1
                else:
                    client_account_no[client_accs] = 1

        plot_pie(client_account_no.values(), client_account_no.keys(), 'Accounts per Client')


# Analyses districts csv and produces box charts for each relevant attribute
def analyse_districts():
    with open('./files/district.csv') as districts:
        attrs = {'inh': [], 'mun_lt_499': [], 'mun_500_1999': [], 'mun_2000_9999': [], 'mun_gt_10000': [],
            'cities': [], 'urban_inh_r': [], 'avg_salary': [], 'unemp_96': [], 'enterp_p_1000': [], 'crimes_96': []}
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
                attrs['unemp_96'].append(float(row[12]))
                attrs['enterp_p_1000'].append(int(row[13]))
                attrs['crimes_96'].append(int(row[15]))

        plot_box(attrs, 'Districts')

# Analyses dispositions csv and produces disposition type pie chart
def analyse_dispositions():
    with open('./files/disp.csv') as dispositions:
        disp_reader = csv.reader(dispositions, delimiter=';')
        next(disp_reader)

        for disp in disp_reader:
            if len(disp) == 4:
                attr_data['disposition_' + disp[3].lower()] += 1

        plot_pie([attr_data['disposition_owner'], attr_data['disposition_disponent']], ['Owner', 'Disponent'],
            'Disposition')

# Analyses training cards csv and produces card type pie chart
def analyse_cards():
    with open('./files/card_train.csv') as cards:
        card_reader = csv.reader(cards, delimiter=';')
        next(card_reader)

        for card in card_reader:
            if len(card) == 4:
                attr_data['cards_' + card[2]] += 1

        plot_pie([attr_data['cards_classic'], attr_data['cards_junior'], attr_data['cards_gold']], ['Classic', 'Junior',
            'Gold'], 'Card Type')

# Analyses accounts csv and produces various statistics regarding them
def analyse_accounts(detailed):
    with open('./files/account.csv') as accounts, open('./files/disp.csv') as dispositions_file, open('./files/card_train.csv') as cards, open('./files/loan_train.csv') as loans:
        acc_reader = csv.reader(accounts, delimiter=';')
        disp_reader = csv.reader(dispositions_file, delimiter=';')

        if detailed:
            loans_reader = csv.reader(loans, delimiter=';')
            cards_reader = csv.reader(cards, delimiter=';')
            disp_nos = {}
            acc_owner_card = {'none': 0, 'junior': 0, 'classic': 0, 'gold': 0}
            acc_cards_no = {}
            acc_loan_no = {}

        next(acc_reader)

        for account in acc_reader:
            if len(account) == 4:
                acc_id = int(account[0])

                if account[2] == 'monthly issuance':
                    attr_data['frequency_monthly'] += 1
                elif account[2] == 'weekly issuance':
                    attr_data['frequency_weekly'] += 1
                else:
                    attr_data['frequency_transactional'] += 1

                if detailed:
                    dispositions = get_dispositions(dispositions_file, disp_reader, acc_id)
                    key = str(len(dispositions))

                    if key in disp_nos:
                        disp_nos[key] += 1
                    else:
                        disp_nos[key] = 1

                    acc_owner_card[get_owner_card(cards, cards_reader, dispositions)] += 1
                    card_no = sum(get_card_types_no(cards, cards_reader, dispositions))

                    if card_no in acc_cards_no:
                        acc_cards_no[card_no] += 1
                    else:
                        acc_cards_no[card_no] = 1

                    acc_loans = len(get_account_loans(loans, loans_reader, acc_id))

                    if acc_loans in acc_loan_no:
                        acc_loan_no[acc_loans] += 1
                    else:
                        acc_loan_no[acc_loans] = 1

        plot_pie([attr_data['frequency_monthly'], attr_data['frequency_transactional'], attr_data['frequency_weekly']],
            ['Monthly', 'After Transaction', 'Weekly'], 'Account Issuance Frequency')

        if detailed:
            plot_pie(disp_nos.values(), disp_nos.keys(), 'Account Dispositions No')
            plot_pie(acc_owner_card.values(), acc_owner_card.keys(), 'Account Owner Card Type')
            plot_pie([acc_owner_card['none'],
                acc_owner_card['junior'] + acc_owner_card['classic'] + acc_owner_card['gold']], ['No', 'Yes'],
                'Account Owner Has Card')
            plot_pie(acc_cards_no.values(), acc_cards_no.keys(), 'Account Card Number')
            plot_pie(acc_loan_no.values(), acc_loan_no.keys(), 'Loans Per Account')

# Analyses training loans csv and produces relevant attributes box chart and loan status pie chart
def analyse_loans():
    attrs = {'amount': [], 'duration': [], 'payments': []}
    status_disp = pd.DataFrame([[0, 0], [0, 0]], columns=['1', '2'], index=['Unsuccessful', 'Successful'])
    loans = pd.read_csv('./files/loan_train.csv', sep=';', header=0, index_col=False)
    dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)

    for i, loan in loans.iterrows():
        if len(loan) == 7:
            status = loan['status']
            row = -1

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

    axis = status_disp[['1', '2']].plot(kind='bar', stacked=True, rot=0)
    fig = axis.get_figure()
    fig.savefig('./figures/Disposition No. & Status.png')
    plt.close()
    
    plot_pie([attr_data['loan_status_appr'], attr_data['loan_status_rej']], ['approved', 'rejected'], 'Loan Status')
    plot_box(attrs, 'Loans')

# Calculates (necessary) missing and / or not loan linked values
def calc_missing_values():
    with open('./files/loan_train.csv') as loans, open('./files/account.csv') as accounts, open('./files/card_train.csv') as cards, open('./files/disp.csv') as dispositions, open('./files/district.csv') as districts:
        loans_reader = csv.reader(loans, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        cards_reader = csv.reader(cards, delimiter=';')
        disp_reader = csv.reader(dispositions, delimiter=';')
        dist_reader = csv.reader(districts, delimiter=';')
        next(loans_reader)

        for row in loans_reader:
            if len(row) == 7:
                acc_id = int(row[1])
                missing_vals = {'missing_districts': True, 'missing_dispositions': True, 'missing_accounts': True}
                account = get_account(accounts, acc_reader, acc_id)

                if len(account) > 0:
                    missing_vals['missing_accounts'] = False
                    dist_id = int(account[1])
                    district = get_district(districts, dist_reader, dist_id)
                    dispositions_list = get_dispositions(dispositions, disp_reader, acc_id)

                    if len(district) > 0:
                        missing_vals['missing_districts'] = False

                    if len(dispositions_list) > 0:
                        missing_vals['missing_dispositions'] = False
            else:
                attr_data['missing_loans'] += 1

        for key in missing_vals.keys():
            if missing_vals[key]:
                attr_data[key] += 1

def get_avg_balance(transactions):
    return transactions['balance'].mean()

# Parses a YYMMDD date to tuple format
def parse_date(date):
    year = '19' + date[:2]
    month = date[2:4]
    day = date[4:]

    return (int(year), int(month), int(day))

# Returns the last transaction of an account before a given date
def get_acc_last_transaction(transactions, date):
    d1 = datetime.datetime(*date)
    prior_trans = []

    for index, trans in transactions.iterrows():
        d2 = datetime.datetime(*parse_date(str(trans['date'])))

        if d2 < d1:
            prior_trans.append(trans)

    return max(prior_trans, key=operator.itemgetter('date'), default=[])
        

# Returns the transactions associated with an account
def get_acc_transactions(transactions_df, acc_id):
    return transactions_df.loc[transactions_df['account_id'] == acc_id]    

# Returns the accounts associated with a given client
def get_client_accounts(client_id, disp_file, disp_reader, acc_file, acc_reader):
    accs = []

    disp_file.seek(0)
    next(disp_reader)

    for disp in disp_reader:
        if len(disp) == 4 and int(disp[1]) == client_id:
            acc_id = int(disp[2])
            acc_file.seek(0)
            next(acc_reader)

            for acc in acc_reader:
                if len(acc) == 4 and int(acc[0]) == acc_id:
                    accs.append(acc)
                    break

    return accs

# Returns the loans associated with a given account
def get_account_loans(loans_file, loans_reader, acc_id):
    loans_file.seek(0)
    next(loans_reader)

    loans = []

    for loan in loans_reader:
        if len(loan) == 7 and int(loan[1]) == acc_id:
            loans.append(loan)

    return loans

# Returns the (junior_card_no, classic_card_no, gold_card_no) associated with an account given its associated dispositions
def get_card_types_no(cards_file, cards_reader, dispositions):
    classic_no, junior_no, gold_no = 0, 0, 0

    for disposition in dispositions:
        disp_id = int(disposition[0])
        cards_file.seek(0)
        next(cards_reader)

        for card in cards_reader:
            if len(card) == 4 and int(card[1]) == disp_id:
                card_type = card[2]

                if card_type == 'classic':
                    classic_no += 1
                elif card_type == 'junior':
                    junior_no += 1
                elif card_type == 'gold':
                    gold_no += 1

    return (junior_no, classic_no, gold_no)

# Returns the (client_id, disposition_id) of the owner of an account given the ASSOCIATED dispositions
def get_account_owner_info(dispositions):
    for disposition in dispositions:
        if disposition[3] == 'OWNER' or int(disposition[3]) == 1:
            return (int(disposition[1]), int(disposition[0]))

    return ()

# Returns the card type (or none) of the owner of an account given the ASSOCIATED dispositions
def get_owner_card(cards_file, cards_reader, acc_dispositions):
    owner = get_account_owner_info(acc_dispositions)

    cards_file.seek(0)
    next(cards_reader)

    for card in cards_reader:
        if len(owner) == 2 and int(card[1]) == owner[1]:
            return card[2]

    return 'none'

# Returns the dispositions associated with an account given an account id
def get_dispositions(dispositions_file, disp_reader, acc_id):
    dispositions = []
    dispositions_file.seek(0)
    next(disp_reader)

    for disposition in disp_reader:
        if len(disposition) == 4 and int(disposition[2]) == acc_id:
            dispositions.append(disposition)

    return dispositions

# Returns a district given a district id
def get_district(districts_file, dist_reader, dist_id):
    districts_file.seek(0)
    next(dist_reader)

    for district in dist_reader:
        if len(district) == 16 and int(district[0]) == dist_id:
            return district

    return []

# Returns an account given an account id
def get_account(accounts_file, acc_reader, acc_id):
    accounts_file.seek(0)
    next(acc_reader)

    for account in acc_reader:
        if len(account) == 4 and int(account[0]) == acc_id:
            return account

    return []

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
    #plt.show()
    plt.savefig('./figures/' + title + '_confusion_matrix.png')
    plt.close()

# Draws a box chart based on a set of numerical attributes
def plot_box(attrs, title):
    for attr in attrs.keys():
        attr_array = attrs[attr]
        r = plt.boxplot(attr_array, vert=False)
        minThresh = r['whiskers'][0].get_xdata()[1]
        maxThresh = r['whiskers'][1].get_xdata()[1]

        attr_data[attr] = (minThresh, maxThresh)

        print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)),
            'Avg: ' + str(sum(attr_array) // len(attr_array)), 'Min.Thresh: ' +  str(attr_data[attr][0]),
            'Max.Thresh ' + str(attr_data[attr][1]), sep=' | ')

        plt.title(title + ' - ' + attr)
        #plt.show()
        plt.savefig('./figures/' + title + '_' + attr + '_box.png')
        plt.close()

# Draws a pie chart based on a set of sizes / numerical data and respective labels
def plot_pie(sizes, labels, title):
    _, loan_chart = plt.subplots()
    loan_chart.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    loan_chart.axis('equal')
    #plt.show()
    plt.title(title)
    plt.savefig('./figures/' +  title + '.png')
    plt.close()

def main():
    analyse_data()

if __name__ == '__main__':
    main()
