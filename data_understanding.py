import csv
import numpy as np
import matplotlib.pyplot as plt

with open('./files/loan_train.csv') as csv_file:
    attrs = {'loan_id': [], 'amount': [], 'duration': []}
    #payments = amount / duration => Correlated, discard
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)

    for row in csv_reader:
        if len(row) == 7:
            attrs['loan_id'].append(int(row[0]))
            attrs['amount'].append(int(row[3]))
            attrs['duration'].append(int(row[4]))
        else:
            print('Loan ID ' + row[0] + ' has missing values')

    for attr in attrs.keys():
        attr_array = attrs[attr]
        print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)), 
            'Avg: ' + str(sum(attr_array) // len(attr_array)), sep=' | ')
    '''
    plt.scatter(ids, amounts, s=1, alpha=0.5)
    plt.title('Loan Amounts - Scatter')
    plt.xlabel('Loan ID')
    plt.ylabel('Amount')

    plt.show()
    plt.savefig('loan_amounts_scatter.png')
    '''