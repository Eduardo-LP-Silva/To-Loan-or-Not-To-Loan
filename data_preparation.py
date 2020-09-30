
import csv
import data_understanding

def clean_data():
    with open('./files/loan_train.csv') as dirty_file, open('./files/loan_train_clean.csv', 'w', newline='') as clean_file:
        csv_reader = csv.reader(dirty_file, delimiter=';')
        csv_writer = csv.writer(clean_file, delimiter=';')
        attr_data = data_understanding.analyse_data()
        next(csv_reader)

        for row in csv_reader:
            if len(row) == 7:
                id = int(row[0])
                amount = int(row[3])
                duration = int(row[4])
                status = int(row[6])
                if(amount >= attr_data['amount'][0] and amount <= attr_data['amount'][1]
                    and duration >= attr_data['duration'][0] and duration <= attr_data['duration'][1]):
                    csv_writer.writerow([row[0], row[3], row[4], row[6]])

clean_data()