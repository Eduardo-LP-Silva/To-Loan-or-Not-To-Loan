
import csv
import data_understanding

# Splits development csv data in two sets: training (2/3, equal number of cases) and testing (1/3)
def arrange_train_test_data(attr_data):
    with open('./files/development.csv') as dev_file, open('./files/train.csv', 'w', newline='') as train_file, open('./files/test.csv', 'w', newline='') as test_file:
        dev_reader = csv.reader(dev_file, delimiter=';')
        train_writer = csv.writer(train_file, delimiter=';')
        test_writer = csv.writer(test_file, delimiter=';')

        total_loans = attr_data['loan_status_appr'] + attr_data['loan_status_rej']
        train_no = (total_loans * 2) // 3
        test_no = total_loans - train_no
        train_data = []
        train_data_status = [0, 0] # Approved / rejected loan status count
        i = 1

        next(dev_reader)

        for dev_row in dev_reader:
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


# Generates new development csv with all relevant data from most csv's
def arrange_dev_data():
    attr_data = data_understanding.analyse_data()
    clean_data(attr_data)
    
    with open('./files/loan_train_clean.csv') as loans, open('./files/development.csv', 'w', newline='') as dev_file, open('./files/account.csv') as accounts, open('./files/card_train.csv') as cards, open('./files/district.csv') as districts, open('./files/disp_clean.csv') as dispositions:
        loan_reader = csv.reader(loans, delimiter=';')
        acc_reader = csv.reader(accounts, delimiter=';')
        dist_reader = csv.reader(districts, delimiter=';')
        disp_reader = csv.reader(dispositions, delimiter=';')
        dev_writer = csv.writer(dev_file, delimiter=';')
        next(loan_reader)

        dev_writer.writerow(['loan_id', 'amount', 'duration', 'dist. no. of inhabitants', 
            'dist. no. of municipalities with inhabitants < 499', 'dist. no. of municipalities with inhabitants 500-1999',
            'dist. no. of municipalities with inhabitants 2000-9999', 'dist. no. of municipalities with inhabitants >10000',
            'dist. no. of cities', 'dist. ratio of urban inhabitants', 'dist. average salary', 
            'dist. unemploymant rate 96', 'dist. no. of enterpreneurs per 1000 inhabitants', 
            'dist. no. of commited crimes 96', 'disposition_type', 'status'])

        for loan in loan_reader:
            status = int(loan[6])
            acc_id = int(loan[1])

            accounts.seek(0)
            next(acc_reader)

            for account in acc_reader:
                if len(account) == 4 and int(account[0]) == acc_id:
                    dist_id = int(account[1])
                    dist_data = []
                    disp_type = -1

                    districts.seek(0)
                    next(dist_reader)

                    for district in dist_reader:
                        if len(district) == 16 and int(district[0]) == dist_id:
                            dist_data = [district[3], district[4], district[5],
                                district[6], district[7], district[8], district[9], district[10], district[12],
                                district[13], district[15]]

                            '''
                            train_writer.writerow([loan[0], loan[3], loan[4], district[3], district[4], district[5],
                                district[6], district[7], district[8], district[9], district[10], district[12],
                                district[13], district[15], loan[6]])
                            '''
                            break

                    dispositions.seek(0)
                    next(disp_reader)

                    for disp in disp_reader:
                        if len(disp) == 4 and int(disp[2]) == acc_id:
                            disp_type = disp[3]
                            break
                            
                    # Account - Add frequency (categorical)? Seems irrelevant
                    # Transactions - Add avg monthly balance and avg transaction value for account
                    # Card - Add Type (Add extra type for None) (categorical)
                    # Add binary if account has already taken a previous loan (only one loan per account) (must parse dates)
                    if len(dist_data) > 0 and disp_type != -1: # if every data point is available add row
                        data_row = [loan[0], loan[3], loan[4]]
                        data_row.extend(dist_data)
                        data_row.append(disp_type)
                        data_row.append(loan[6])
                        dev_writer.writerow(data_row)
        
        arrange_train_test_data(attr_data)
                        
# Copies and changes some csv data to new, 'cleaned' files
def clean_data(attr_data):
    clean_loans(attr_data)
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
                
'''
def clean_accounts(attr_data):
    with open('./files/account.csv') as accounts, open('./files/account_clean.csv', 'w', newline='') as accounts_clean:
        acc_reader = csv.reader(accounts, delimiter=';')
        acc_writer = csv.writer(accounts_clean, delimiter=';')
        acc_writer.writerow(next(acc_reader))

        for account in acc_reader:
            if len(account) == 4:
                frequency = account[2]

                acc_writer.writerow(account)
'''

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

arrange_dev_data()