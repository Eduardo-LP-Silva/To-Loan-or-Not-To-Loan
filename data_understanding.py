import csv
import matplotlib.pyplot as plt

# Data to be passed to preparation
attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'missing_loans': 0, 'missing_districts': 0, 
    'missing_dispositions': 0, 'missing_accounts': 0, 'frequency_monthly': 0, 'frequency_transactional': 0, 
    'frequency_weekly': 0, 'cards_classic': 0, 'cards_junior': 0, 'cards_gold': 0, 'disposition_owner': 0, 
    'disposition_disponent': 0}

# Analyses csv's data and produces respective statistics
def analyse_data():
    calc_missing_values()
    analyse_loans()
    analyse_accounts()
    analyse_cards()
    analyse_dispositions()
    analyse_districts()

    for key in attr_data.keys():
        print(key, str(attr_data[key]), sep=': ')

    return attr_data

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

# Analyses accounts csv and produces statement issuance frequency pie chart
def analyse_accounts():
    with open('./files/account.csv') as accounts:
        acc_reader = csv.reader(accounts, delimiter=';')
        next(acc_reader)

        for account in acc_reader:
            if len(account) == 4:
                if account[2] == 'monthly issuance':   
                    attr_data['frequency_monthly'] += 1
                elif account[2] == 'weekly issuance':
                    attr_data['frequency_weekly'] += 1
                else:
                    attr_data['frequency_transactional'] += 1

        plot_pie([attr_data['frequency_monthly'], attr_data['frequency_transactional'], attr_data['frequency_weekly']], 
            ['Monthly', 'After Transaction', 'Weekly'], 'Account Issuance Frequency')

# Analyses training loans csv and produces relevant attributes box chart and loan status pie chart
def analyse_loans():
    with open('./files/loan_train.csv') as loans:
        attrs = {'amount': [], 'duration': [], 'payments': []}
        loans_reader = csv.reader(loans, delimiter=';')
        next(loans_reader)

        for row in loans_reader:
            if len(row) == 7:
                status = int(row[6])
                attrs['amount'].append(int(row[3]))
                attrs['duration'].append(int(row[4]))
                attrs['payments'].append(int(row[5]))
            
                if status == 1:
                    attr_data['loan_status_appr'] += 1
                else:
                    attr_data['loan_status_rej'] += 1

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

# Returns the (client_id, disposition_id) of the owner of an account given the associated dispositions 
def get_account_owner_info(dispositions):
    for disposition in dispositions:
        if disposition[3] == 'OWNER':
            return (int(disposition[1]), int(disposition[0]))

    return ()

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