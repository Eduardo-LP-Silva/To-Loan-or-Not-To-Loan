import csv
import operator
import datetime
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import date

plt.rcParams['font.size'] = 8.0

# Data to be passed to preparation
attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'trans_op_mode': '', 'trans_k_mode': ''}

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

                trans_op = 'Missing' if not trans[4] or trans[4].isspace() else trans[4]

                if trans_op in trans_operations:
                    trans_operations[trans_op] += 1
                else:
                    trans_operations[trans_op] = 0

                trans_k = 'Missing' if not trans[7] or trans[7].isspace() else trans[7]

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

        trans_operations.pop('Missing')
        trans_ks.pop('Missing')
        attr_data['trans_op_mode'] = max(trans_operations.items(), key=operator.itemgetter(1))[0]
        attr_data['trans_k_mode'] = max(trans_ks.items(), key=operator.itemgetter(1))[0]

# Analyses the clients csv and produces the number of accounts per client plot
def analyse_clients():
    with open('./files/client.csv') as clients, open('./files/disp.csv') as dispositions, open('./files/account.csv') as accounts:
        clients_reader = csv.reader(clients, delimiter=';')
        disp_reader = csv.reader(dispositions, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        client_account_no = {}
        gender_list = {'male': 0, 'female': 0}

        next(clients_reader)

        for client in clients_reader:
            if len(client) == 3:
                client_accs = len(get_client_accounts(int(client[0]), dispositions, disp_reader, accounts, acc_reader))
                # TODO
                # client_age = get_client_age(client[1])
                client_gender = get_client_gender(client[1])
                if client_gender == 0:
                    gender_list['male'] += 1
                else:
                    gender_list['female'] += 1

                if client_accs in client_account_no:
                    client_account_no[client_accs] += 1
                else:
                    client_account_no[client_accs] = 1

        plot_pie(client_account_no.values(), client_account_no.keys(), 'Accounts per Client')
        plot_pie(gender_list.values(), gender_list.keys(), 'Clients Gender')


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
    with open('./files/disp.csv') as dispositions:
        disp_reader = csv.reader(dispositions, delimiter=';')
        disp_type = {'owner': 0, 'disponent': 0}
        next(disp_reader)

        for disp in disp_reader:
            if len(disp) == 4:
                disp_type[disp[3].lower()] += 1

        plot_pie(disp_type.values(), disp_type.keys(), 'Disposition')

# Analyses training cards csv and produces card type pie chart
def analyse_cards():
    with open('./files/card_train.csv') as cards:
        card_reader = csv.reader(cards, delimiter=';')
        card_types = {'classic': 0, 'junior': 0, 'gold': 0}
        next(card_reader)

        for card in card_reader:
            if len(card) == 4:
                card_types[card[2]] += 1

        plot_pie(card_types.values(), card_types.keys(), 'Card Type')

# Analyses accounts csv and produces various statistics regarding them
def analyse_accounts(detailed):
    with open('./files/account.csv') as accounts, open('./files/card_train.csv') as cards, open('./files/loan_train.csv') as loans:
        acc_reader = csv.reader(accounts, delimiter=';')
        dispositions = pd.read_csv('./files/disp.csv', sep=';', header=0, index_col=False)
        freqs = {'Monthly': 0, 'Weekly': 0, 'After Transaction': 0}

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
                    freqs['Monthly'] += 1
                elif account[2] == 'weekly issuance':
                    freqs['Weekly'] += 1
                else:
                    freqs['After Transaction'] += 1

                if detailed:
                    acc_dispositions = get_dispositions(dispositions, acc_id)
                    key = str(len(acc_dispositions))

                    if key in disp_nos:
                        disp_nos[key] += 1
                    else:
                        disp_nos[key] = 1

                    acc_owner_card[get_owner_card(cards, cards_reader, acc_dispositions)] += 1
                    card_no = sum(get_card_types_no(cards, cards_reader, acc_dispositions))

                    if card_no in acc_cards_no:
                        acc_cards_no[card_no] += 1
                    else:
                        acc_cards_no[card_no] = 1

                    acc_loans = len(get_account_loans(loans, loans_reader, acc_id))

                    if acc_loans in acc_loan_no:
                        acc_loan_no[acc_loans] += 1
                    else:
                        acc_loan_no[acc_loans] = 1

        plot_pie(freqs.values(), freqs.keys(), 'Account Issuance Frequency')

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
    clients = pd.read_csv('./files/client.csv', sep=';', header=0, index_col=False)
    age_dist = {'0-19':0, '20-29':0,'30-39':0, '40-49':0, '50-59':0, '60-69':0, '70+':0}

    for i, loan in loans.iterrows():
        if len(loan) == 7:
            status = loan['status']
            row = -1

            acc_id = loan['account_id']
            owner = get_acc_owner(acc_id, dispositions, clients)
            owner_loan_age = calculate_loan_client_age(str(owner['birth_number']), str(loan['date']))

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

    axis = status_disp[['1', '2']].plot(kind='bar', stacked=True, rot=0)
    fig = axis.get_figure()
    fig.savefig('./figures/Disposition No. & Status.png')
    plt.close()

    plot_pie(age_dist.values(), age_dist.keys(), 'Clients Age at Loan Request')
    plot_pie([attr_data['loan_status_appr'], attr_data['loan_status_rej']], ['approved', 'rejected'], 'Loan Status')
    plot_box(attrs, 'Loans')

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

'''
def get_income(acc_transactions):
    acc_transactions.sort(key=operator.attrgetter('date'))
    monthly_incomes = [0]
    i = 0
    previous_date = parse_date(str(acc_transactions[0]['date'])) if len(acc_transactions) > 0 else None

    for trans in acc_transactions:
        if trans['type'] == 'credit':
            date = parse_date(str(trans['date']))

            if date[0] != previous_date[0]:
                monthly_incomes.append(trans['amount'])
                i += 1
            else:
                monthly_incomes[i] += trans['amount']

            previous_date = date

    return sum(monthly_incomes) / len(monthly_incomes)
'''

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

    for _, disposition in dispositions.iterrows():
        disp_id = disposition['disp_id']
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
    acc_owner = dispositions[dispositions['type'] == 'OWNER' or dispositions['type'] == 1].iloc[0]

    return (acc_owner.at['client_id'], acc_owner.at['disp_id'])

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
def get_dispositions(dispositions, acc_id):
    return dispositions[dispositions['account_id'] == acc_id]

# Returns a district given a district id
def get_district(districts, dist_id):
    return districts[districts['code '] == dist_id].iloc[0]

# Returns an account given an account id
def get_account(accounts, acc_id):
    return accounts[accounts['account_id'] == acc_id].iloc[0]

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

        thresholds = (minThresh, maxThresh)

        print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)),
            'Avg: ' + str(sum(attr_array) // len(attr_array)), 'Min.Thresh: ' +  str(thresholds[0]),
            'Max.Thresh ' + str(thresholds[1]), sep=' | ')

        plt.title(title + ' - ' + attr)
        #plt.show()
        plt.savefig('./figures/' + title + '_' + attr + '_box.png')
        plt.close()

# Draws a pie chart based on a set of sizes / numerical data and respective labels
def plot_pie(sizes, labels, title):
    _, loan_chart = plt.subplots()
    loan_chart.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.tight_layout(pad=6.0)
    loan_chart.axis('equal')
    #plt.show()
    plt.title(title)
    plt.savefig('./figures/' +  title + '.png')
    plt.close()

def main():
    analyse_data()

if __name__ == '__main__':
    main()
