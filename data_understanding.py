import csv
import numpy as np
import matplotlib.pyplot as plt

with open('./files/loan_train.csv') as csv_file:
    attrs = {'amount': [], 'duration': []}
    #payments = amount / duration => Correlated, discard
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)

    for row in csv_reader:
        if len(row) == 7:
            attrs['amount'].append(int(row[3]))
            attrs['duration'].append(int(row[4]))
        else:
            print('Loan ID ' + row[0] + ' has missing values')

    for attr in attrs.keys():
        attr_array = attrs[attr]
        r = plt.boxplot(attr_array, vert=False)

        print(attr, 'Max: ' + str(max(attr_array)), 'Min: ' + str(min(attr_array)), 
            'Avg: ' + str(sum(attr_array) // len(attr_array)), 'Min.Thresh: ' +  str(r['whiskers'][0].get_xdata()[1]), 
            'Max.Thresh ' + str(r['whiskers'][1].get_xdata()[1]), sep=' | ')
        plt.title('Loan Amounts - ' + attr)
        plt.show()
        plt.savefig('loan_' + attr + '_box.png')
    '''
    plt.scatter(ids, amounts, s=1, alpha=0.5)
    plt.title('Loan Amounts - Scatter')
    plt.xlabel('Loan ID')
    plt.ylabel('Amount')

    plt.show()
    plt.savefig('loan_amounts_scatter.png')
    '''