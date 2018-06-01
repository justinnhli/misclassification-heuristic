#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from collections import defaultdict
from random import sample
from os import mkdir
from os.path import basename, dirname, isdir, realpath, join as join_path, exists as file_exists

import numpy as np
from keras.optimizers import rmsprop
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

from classifiers import DomainUtils, Classifier, Dataset, RegretTrial

FILE_DIR = dirname(realpath(__file__))

class ImageUtils(DomainUtils):

    def __init__(self, dataset_str):
        assert dataset_str in ['cifar10', 'cifar100'], 'unrecognized dataset "{}"'.format(dataset_str)
        # set data files
        if dataset_str == 'cifar10':
            uri_file = join_path(FILE_DIR, 'umbel/cifar10.uris')
            distance_file = join_path(FILE_DIR, 'umbel/cifar10.distances')
        elif dataset_str == 'cifar100':
            uri_file = join_path(FILE_DIR, 'umbel/cifar100.uris')
            distance_file = join_path(FILE_DIR, 'umbel/cifar100.distances')
        # initialize data
        self.dataset_str = dataset_str
        self.labels = []
        self.concept_label_map = {}
        self.umbel_distances = defaultdict(dict)
        # read in label-URI data
        with open(uri_file) as fd:
            for line in fd.readlines():
                label, uri = line.strip().split('\t')
        # read in distance data
        with open(distance_file) as fd:
            for line in fd.read().splitlines():
                match = re.match(r'^(umbel-rc:[^ ]*) -- (umbel-rc:[^ ]*) \(([^)]*)\)$', line)
                if not match:
                    continue
                label1 = self.concept_label_map[match.group(1)]
                label2 = self.concept_label_map[match.group(2)]
                distance = int(match.group(3))
                self.umbel_distances[label1][label2] = distance
                self.umbel_distances[label2][label1] = distance
            for label in list(self.umbel_distances.keys()):
                self.umbel_distances[label][label] = 0

    def class_to_label(self, y):
        return self.labels[y]

    def label_to_class(self, label):
        return self.labels.index(label)

    def class_distance(self, y1, y2):
        return self.label_distance(self.class_to_label(y1), self.class_to_label(y2))

    def label_distance(self, label1, label2):
        return self.umbel_distances[label1][label2]


class NeuralNetwork(Classifier):

    def __init__(self, filepath, load_network=True):
        if load_network:
            self.model = load_model(filepath)
        else:
            self.model = None
        match = re.search('([^-]+)-l([0-9]+)-b([0-9]+)-e([0-9]+).hdf5$', basename(filepath))
        assert match, 'Cannot parse filename "{}"'.format(filepath)
        self.dataset_str = match.group(1)
        self.int_labels = binary_to_ints(int(match.group(2)))
        self.batch_size = int(match.group(3))
        self.num_epochs = int(match.group(4))

    def get_persistent_id(self):
        """Get a persistent (string) id

        Returns:
            str: an id for this classifier
        """
        return '{}-l{}-b{:02d}-e{:03d}'.format(
            self.dataset_str,
            str(ints_to_binary(self.int_labels)),
            self.batch_size,
            self.num_epochs,
        )

    def get_ys(self):
        """Get the labels

        Returns:
            [int]: the integer labels
        """
        return self.int_labels

    def classify(self, xs):
        """Classify the data

        Arguments:
            xs: the data to be classified
                this can be whatever Dataset.get_x() returns

        Returns:
            [int]: the classifications
        """
        y_predict_raw = self.model.predict(xs)
        y_predict = np.vectorize(lambda l: self.int_labels[l])(np.argmax(y_predict_raw, axis=1)).tolist()
        return y_predict

    def to_file(self, filepath):
        """Save the classifier to file

        Arguments:
            filepath (str): The file to save to
        """
        self.model.save(filepath)

    @classmethod
    def from_file(cls, filepath):
        """Create a classifier from file

        Arguments:
            filepath (str): The file to load from
        """
        return cls(filepath)


class ImageDataset(Dataset):

    def __init__(self, dataset_str):
        self.dataset_str = dataset_str

    def get_persistent_id(self):
        return self.dataset_str

    def get_x(self):
        if self.dataset_str == 'cifar10':
            (_, _), (x_test, _) = cifar10.load_data()
        elif self.dataset_str == 'cifar100':
            (_, _), (x_test, _) = cifar100.load_data()
        else:
            assert False
        return x_test

    def get_y(self):
        if self.dataset_str == 'cifar10':
            (_, _), (_, y_test) = cifar10.load_data()
        elif self.dataset_str == 'cifar100':
            (_, _), (_, y_test) = cifar100.load_data()
        else:
            assert False
        return y_test.transpose().tolist()[0]


def train_neural_network(int_labels, batch_size, num_epochs, dataset_str, verbose=False):
    OUTPUT_PATH = 'images'
    if file_exists(OUTPUT_PATH):
        assert isdir(OUTPUT_PATH), '"{}" exists but is not a directory'.format(OUTPUT_PATH)
    else:
        mkdir(OUTPUT_PATH)

    # load the data
    if dataset_str == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # remove unused labels
    label_filter = np.in1d(y_train, int_labels)
    x_train = x_train[label_filter]
    y_train = y_train[label_filter]
    label_filter = np.in1d(y_test, int_labels)
    x_test = x_test[label_filter]
    y_test = y_test[label_filter]

    # map original labels to range(num_labels)
    mapping = {label: i for i, label in enumerate(int_labels)}
    mapping_fn = np.vectorize(lambda old: mapping[old])
    y_train = mapping_fn(y_train)
    y_test = mapping_fn(y_test)

    # normalize dataset to [0, 1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, len(int_labels))
    y_test = to_categorical(y_test, len(int_labels))

    # this will do preprocessing and real time data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False, # randomly flip images
    )

    # compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # create the neural network
    model = Sequential()
    # add a 2D convolution layer (with special arguments for the input)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    # add a 2D convolution layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # add a max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # add a 2D convolution layer
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # add a 2D convolution layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # add a max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # add the output voting layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(int_labels)))
    model.add(Activation('softmax'))
    # initiate RMSprop optimizer
    optimizer = rmsprop(lr=0.0001, decay=1e-6)
    # train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        verbose=(1 if verbose else 0),
    )

    filepath = join_path(
        OUTPUT_PATH,
        '{}-l{}-b{:02d}-e{:03d}.hdf5'.format(
            dataset_str,
            str(ints_to_binary(int_labels)),
            batch_size,
            num_epochs,
        ),
    )
    model.save(filepath)

    return NeuralNetwork(filepath)


def labels_to_str(int_labels):
    return [LABELS_10[i] for i in int_labels]

def labels_to_int(str_labels):
    return sorted(set(LABELS_10.index(label) for label in str_labels))

def ints_to_binary(int_labels):
    """
    Arguments:
        int_labels ([int]): The index of the 1 bits

    Returns:
        int: The binary as an integer
    """
    return sum(2**i for i in int_labels)

def binary_to_ints(binary):
    """
    Arguments:
        binary (int): The binary as an integer

    Returns:
        [int]: The index of the 1 bits
    """
    binary_str = '{:b}'.format(binary)
    return sorted(i for i, b in enumerate(reversed(binary_str)) if b == '1')

def sample_classes(p, k):
    int_labels = sample(range(p), k)
    while p == 100 and 6 in int_labels:
        int_labels = sample(range(p), k)
    return sorted(int_labels)

def regret_trial_test():
    nn = NeuralNetwork('image10l544b32e003.hdf5')
    (_, _), (dataset, _) = cifar10.load_data()
    RegretTrial(
        nn,
        ImageUtils('cifar10'),
        ImageDataset('cifar10'),
        path_prefix='images',
    )

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'], help='the dataset to use')
    arg_parser.add_argument('--num-labels', type=int, default=5, help='number of labels to use')
    arg_parser.add_argument('--batch-size', type=int, default=32, help='size of each batch')
    arg_parser.add_argument('--num-epochs', type=int, default=200, help='number of epochs to run')
    arg_parser.add_argument('--verbose', action='store_true', help='show animated progress bar')
    args = arg_parser.parse_args()
    if args.dataset == 'cifar10':
        int_labels = sample_classes(10, args.num_labels)
    elif args.dataset == 'cifar100':
        int_labels = sample_classes(100, args.num_labels)
    train_neural_network(int_labels, args.batch_size, args.num_epochs, args.dataset, verbose=args.verbose)


if __name__ == '__main__':
    main()
