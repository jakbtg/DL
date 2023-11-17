import csv
import random
import os


# Divide the data into training and dev sets
def build_datasets(force=False):
    if not os.path.exists('../data/sent140.dev.csv') or force:
        # Open the source file
        with open('../data/original.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Shuffle the data
        random.shuffle(data)

        # Open the destination files
        with open('../data/sent140.dev.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data[:10000])

        with open('../data/sent140.test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data[10000:20000])

        with open('../data/sent140.train.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data[20000:])


def clean_data():
    for file in os.listdir('../data'):
        if file.endswith('.csv') and file != 'original.csv':
            with open('../data/' + file, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                # Check if it is already cleaned by checking number of columns, if it is 2, then it is already cleaned
                if len(data[0]) == 2:
                    print(f'{file} is already cleaned')
                    continue
                # Convert the labels from 0,4 to 0,1
                for row in data:
                    if row[0] == '0':
                        row[0] = '0'
                    elif row[0] == '4':
                        row[0] = '1'
                # Remove all columns except the first and last
                for row in data:
                    del row[1:-1]
                # Write the cleaned data to the file
                with open('../data/' + file, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(data)



if __name__ == '__main__':
    build_datasets()
    clean_data()

