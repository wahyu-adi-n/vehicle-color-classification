from config import *
import matplotlib.pyplot as plt
import csv
import argparse
import time


def readCSV(filepath):
    filename = filepath[7:-4]
    acc, loss, val_acc, val_loss = [], [], [], []

    with open(filepath, 'r') as f:
        files = csv.reader(f)
        next(files, None)
        for file in files:
            loss.append(float(file[0]))
            acc.append(float(file[1]))
            val_loss.append(float(file[2]))
            val_acc.append(float(file[3]))

    return loss, acc, val_loss, val_acc, filename


def plot(train, val=None, label='accuracy'):
    plt.title(f'model {label}')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.plot(train)
    plt.plot(val)
    plt.legend([f'train {label}', f'val {label}'],
               loc='lower right' if label[-1] == 'y' else 'upper right')
    plt.savefig(
        f'{SAVE_PLOT_PATH}{time.strftime("%Y%m%d-%H%M%S")}-{label}')
    plt.close()


def main(args):
    loss, acc, val_loss, val_acc, filename = readCSV(args.file)
    plot(acc, val_acc, f'{filename} accuracy')
    plot(loss, val_loss, f'{filename} loss')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Color Visualize')

    parser.add_argument(
        '--file',
        type=str,
        default="output/cnn_50.csv",
        metavar='FP',
        help='File path of file'
    )

    args = parser.parse_args()

    main(args)
