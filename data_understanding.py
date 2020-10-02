import csv
import matplotlib.pyplot as plt

attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'missing_loans': 0, 'missing_districts': 0,
    'missing_dispositions': 0, 'missing_cards': 0, 'missing_accounts': 0, 'frequency_monthly': 0,
    'frequency_transactional': 0, 'cards_classic': 0, 'cards_junior': 0, 'cards_gold': 0, 'disposition_owner': 0,
    'disposition_disponent': 0}

def analyse_data():
    calc_missing_values()
    analyse_loans()
    analyse_accounts()
    analyse_cards()
    analyse_dispositions()
    analyse_districts()
    return attr_data

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

def analyse_dispositions():
    with open('./files/disp.csv') as dispositions:
        disp_reader = csv.reader(dispositions, delimiter=';')
        next(disp_reader)

        for disp in disp_reader:
            if len(disp) == 4:
                attr_data['disposition_' + disp[3].lower()] += 1

        plot_pie([attr_data['disposition_owner'], attr_data['disposition_disponent']], ['Owner', 'Disponent'],
            'Disposition')

def analyse_cards():
    with open('./files/card_train.csv') as cards:
        card_reader = csv.reader(cards, delimiter=';')
        next(card_reader)

        for card in card_reader:
            if len(card) == 4:
                attr_data['cards_' + card[2]] += 1

        plot_pie([attr_data['cards_classic'], attr_data['cards_junior'], attr_data['cards_gold']], ['Classic', 'Junior',
            'Gold'], 'Card Type')


def analyse_accounts():
    with open('./files/account.csv') as accounts:
        acc_reader = csv.reader(accounts, delimiter=';')
        next(acc_reader)

        for account in acc_reader:
            if len(account) == 4:
                if account[2] == 'monthly issuance':
                    attr_data['frequency_monthly'] += 1
                else:
                    attr_data['frequency_transactional'] += 1

        plot_pie([attr_data['frequency_monthly'], attr_data['frequency_transactional']], ['Monthly', 'After Transaction'],
            'Account Issuance Frequency')


def analyse_loans():
    with open('./files/loan_train.csv') as loans:
        attrs = {'amount': [], 'duration': [], 'payments': []}
        loans_reader = csv.reader(loans, delimiter=';')
        next(loans_reader)

        for row in loans_reader:
            if len(row) == 7:
                attrs['amount'].append(int(row[3]))
                attrs['duration'].append(int(row[4]))
                attrs['payments'].append(int(row[5]))

        plot_pie([attr_data['loan_status_appr'], attr_data['loan_status_rej']], ['approved', 'rejected'], 'Loan Status')

        for key in attr_data.keys():
            print(key, str(attr_data[key]), sep=': ')

        plot_box(attrs, 'Loans')

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
                status = int(row[6])
                missing_vals = {'missing_districts': True, 'missing_dispositions': True, 'missing_cards': True,
                    'missing_accounts': True}

                if status == 1:
                    attr_data['loan_status_appr'] += 1
                else:
                    attr_data['loan_status_rej'] += 1

                accounts.seek(0)
                next(acc_reader)

                for account in acc_reader:
                    if int(account[0]) == acc_id and len(account) == 4:
                        dist_id = int(account[1])

                        districts.seek(0)
                        next(dist_reader)

                        for district in dist_reader:
                            if int(district[0]) == dist_id and len(district) == 16:
                                missing_vals['missing_districts'] = False
                                break

                        dispositions.seek(0)
                        next(disp_reader)

                        for disposition in disp_reader:
                            if int(disposition[2]) == acc_id and len(disposition) == 4:
                                disp_id = int(disposition[0])

                                cards.seek(0)
                                next(cards_reader)

                                for card in cards_reader:
                                    if int(card[1]) == disp_id and len(card) == 4:
                                        missing_vals['missing_cards'] = False
                                        break

                                missing_vals['missing_dispositions'] = False
                                break


                        missing_vals['missing_accounts'] = False
                        break
            else:
                attr_data['missing_loans'] += 1

        for key in missing_vals.keys():
            if missing_vals[key]:
                attr_data[key] += 1

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
        # plt.savefig('./figures/' + title + '_' + attr + '_box.png')
        plt.close()

def plot_pie(sizes, labels, title):
    _, loan_chart = plt.subplots()
    loan_chart.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    loan_chart.axis('equal')
    #plt.show()
    plt.title(title)
    # plt.savefig('./figures/' +  title + '.png')
    plt.close()

# analyse_data()
