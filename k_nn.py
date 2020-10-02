import csv
import data_preparation

def create_prediction():
    with open('./files/loan_train_clean.csv') as loans, open('./files/prediction.csv', 'w', newline='') as prediction:
        loan_reader = csv.reader(loans, delimiter=';')
        prediction_writer = csv.writer(prediction, delimiter=';')

        prediction_writer.writerow(['Id', 'Prediction'])
        for row in loan_reader:
            if row[0] != 'loan_id':
                prediction_writer.writerow([row[0],''])
        print('Created Prediction File')

create_prediction()
data_preparation.arrange_data()
data_preparation.normalize_data()
