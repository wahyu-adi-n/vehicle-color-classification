# Setting Up the Labels
labels = ['black', 'blue', 'cyan', 'gray', 'green', 'red', 'white', 'yellow']


def decode_label(index):
    return labels[index]


def encode_label_from_path(path):
    for index, value in enumerate(labels):
        if value in path:
            return index
