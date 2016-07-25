
import csv
import numpy as np

def load_data(input_filepath):
    data = []
    with open(input_filepath, 'rb') as infile:
        # remove header
        infile.readline()
        # read each line 
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            data.append(int(row[-1]))
    data = np.asarray(data, dtype=np.int64)
    return data

if __name__ == '__main__':
    input_filepath = '../data/old_faithful.csv'
    data = load_data(input_filepath)
    print data