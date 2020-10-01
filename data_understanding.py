import csv
import matplotlib.pyplot as plt

attr_data = {'loan_status_appr': 0, 'loan_status_rej': 0, 'missing_loans': 0, 'missing_districts': 0, 
    'missing_dispositions': 0, 'missing_cards': 0, 'missing_accounts': 0}

def analyse_data():
    calc_missing_values()
    analyse_loans()
    return attr_data

def analyse_loans():
    with open('./files/loan_train.csv') as loans:
        attrs = {'amount': [], 'duration': [], 'payments': []}
        loans_reader = csv.reader(loans, delimiter=';')
        next(loans_reader)

        for row in loans_reader:
            attrs['amount'].append(int(row[3]))
            attrs['duration'].append(int(row[4]))
            attrs['payments'].append(int(row[5]))

        draw_loan_pie_chart()

        for key in attr_data.keys():
            print(key, str(attr_data[key]), sep=': ')

        for attr in attrs.keys():
            attr_array = attrs[attr]
            r = plt.boxplot(attr_array, vert=False)
            minThresh = r['whiskers'][0].get_xdata()[1]
            maxThresh = r['whiskers'][1].get_xdata()[1]

            attr_data[attr] = (minThresh, maxThresh)

            print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)), 
                'Avg: ' + str(sum(attr_array) // len(attr_array)), 'Min.Thresh: ' +  str(attr_data[attr][0]), 
                'Max.Thresh ' + str(attr_data[attr][1]), sep=' | ')
            
            plt.title('Loan - ' + attr)
            #plt.show()
            plt.savefig('loan_' + attr + '_box.png')

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

def draw_loan_pie_chart():
    total_loans = attr_data['loan_status_appr'] + attr_data['loan_status_rej']
    loan_labels = ['approved', 'rejected']
    loan_sizes = [(attr_data['loan_status_appr'] / total_loans) * 100, 
        (attr_data['loan_status_rej'] / total_loans) * 100]

    _, loan_chart = plt.subplots()
    loan_chart.pie(loan_sizes, labels=loan_labels, autopct='%1.1f%%', startangle=90)
    loan_chart.axis('equal')
    #plt.show()
    plt.title('Loan Status')
    plt.savefig('loan_status.png')

analyse_data()