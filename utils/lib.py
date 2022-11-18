# Setting Up the Labels
labels = ['black', 'blue', 'gray',  'red', 'white']


def decode_label(index):
    return labels[index]


def encode_label_from_path(path):
    for index, value in enumerate(labels):
        if value in path:
            return index
