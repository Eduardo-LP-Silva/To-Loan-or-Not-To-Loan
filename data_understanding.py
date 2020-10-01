import csv
import matplotlib.pyplot as plt

def analyse_data():
    with open('./files/loan_train.csv') as csv_file:
        attrs = {'amount': [], 'duration': [], 'payments': []}
        attr_data = {'status_appr': 0, 'status_rej': 0}
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)

        for row in csv_reader:
            if len(row) == 7:
                attrs['amount'].append(int(row[3]))
                attrs['duration'].append(int(row[4]))
                attrs['payments'].append(int(row[5]))
                status = int(row[6])

                if status == 1:
                    attr_data['status_appr'] += 1
                else:
                    attr_data['status_rej'] += 1 
            else:
                print('Loan ID ' + row[0] + ' has missing values')

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
            
            plt.title('Loan Amounts - ' + attr)
            #plt.show()
            plt.savefig('loan_' + attr + '_box.png')
            
        return attr_data

analyse_data()