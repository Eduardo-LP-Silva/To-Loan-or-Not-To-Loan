import csv

with open('./files/district.csv') as districts, open('./files/district_names.csv', 'w', newline='') as dist_names:
    dist_reader = csv.reader(districts, delimiter=';')
    dist_writer = csv.writer(dist_names, delimiter='_')

    next(dist_reader)

    for district in dist_reader:
        dist_writer.writerow([district[1]])