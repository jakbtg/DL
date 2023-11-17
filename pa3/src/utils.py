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


if __name__ == '__main__':
    build_datasets()
    

