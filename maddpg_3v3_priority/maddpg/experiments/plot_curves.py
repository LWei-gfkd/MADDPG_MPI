import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def plot_curves(file_path, column, interval = 10, **kwargs):
    plt.xlabel(kwargs['x_name'])
    plt.ylabel(kwargs['y_name'])
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        tmp = []
        result = []
        for row in reader:
            tmp.append(eval(row[column]))
            if len(tmp) == interval:
                result.append(np.mean(tmp))
                tmp = []
        plt.plot(np.array(result), label=kwargs['label'])


def plot_figure(file_list, y_name, column_num):
    plt.figure()

    for file in file_list:
        args = {}
        args['x_name'] = 'Episodes'
        args['y_name'] = y_name
        label = file.replace('1mydata_', '').split('.')[0]
        args['label'] = label
        plot_curves(file, column_num, **args)
    plt.legend()
    plt.show()

def main():
    plot_figure(sys.argv[1:], 'Reward', 1)
    plot_figure(sys.argv[1:], "Win rate", 5)

if __name__ == '__main__':
    main()