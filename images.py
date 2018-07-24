#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from os import mkdir
from os.path import basename, dirname, isdir, realpath, join as join_path, exists as file_exists, splitext, expanduser
from random import sample

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10, cifar100
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.metrics import categorical_accuracy
from keras.models import load_model, Sequential
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from classifiers import DomainUtils, Classifier, Dataset, RegretTrial

FILE_DIR = dirname(realpath(__file__))

AQUARIUM_FISH_Y = 5
AQUARIUM_FISH = 'aquarium fish'


def prepare_data(x_train, y_train, x_test, y_test, int_labels):
    # remove unused labels
    if int_labels is not None:
        label_filter = np.isin(y_train, int_labels).T[0]
        x_train = x_train[label_filter]
        y_train = y_train[label_filter]
        label_filter = np.isin(y_test, int_labels).T[0]
        x_test = x_test[label_filter]
        y_test = y_test[label_filter]
    # normalize RGB to [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return ((x_train, y_train), (x_test, y_test))


def load_cifar10(int_labels=None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data(x_train, y_train, x_test, y_test, int_labels)
    return ((x_train, y_train), (x_test, y_test))


def load_cifar100(int_labels=None):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    # filter out aquarium fish
    if int_labels is None:
        int_labels = list(range(100))
    int_labels = list(i for i in range(100) if i != AQUARIUM_FISH_Y)
    (x_train, y_train), (x_test, y_test) = prepare_data(x_train, y_train, x_test, y_test, int_labels)
    return ((x_train, y_train), (x_test, y_test))


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
                # HACK deal with AQUARIUM_FISH, which doesn't have a clear concept
                if label != AQUARIUM_FISH:
                    self.labels.append(label)
                    self.concept_label_map[uri] = label
                else:
                    self.labels.append(None)
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
        result = self.labels[y]
        assert result is not None, 'Looking up "{}", which is "aquarium fish"'.format(y)
        return result

    def label_to_class(self, label):
        return self.labels.index(label)

    def class_distance(self, y1, y2):
        return self.label_distance(self.class_to_label(y1), self.class_to_label(y2))

    def label_distance(self, label1, label2):
        return self.umbel_distances[label1][label2]


class NeuralNetwork(Classifier):

    def __init__(self, filepath):
        match = re.search('([^-]+)-l([0-9]+)-b([0-9]+)-e([0-9]+).hdf5$', basename(filepath))
        assert match, 'Cannot parse filename "{}"'.format(filepath)
        self.filepath = filepath
        self._model = None
        self.dataset_str = match.group(1)
        self.int_labels = binary_to_ints(int(match.group(2)))
        self.batch_size = int(match.group(3))
        self.num_epochs = int(match.group(4))
        self._test_predictions = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.filepath)
        return self._model

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

    def label_true_positive_init(self, old_label):
        if self.dataset_str == 'cifar10':
            (_, _), (x_test, y_test) = load_cifar10()
        else:
            (_, _), (x_test, y_test) = load_cifar100()
        if self._test_predictions is None:
            self._test_predictions = self.classify(x_test)
        y_predict = self._test_predictions
        num_correct = 0
        num_positive = 0
        for actual, predicted in zip(y_test.T[0], y_predict):
            if actual == old_label:
                if actual == predicted:
                    num_correct += 1
                num_positive += 1
        return num_correct / num_positive

    def true_positive_init(self):
        if self.dataset_str == 'cifar10':
            (_, _), (x_test, y_test) = load_cifar10()
        else:
            (_, _), (x_test, y_test) = load_cifar100()
        if self._test_predictions is None:
            self._test_predictions = self.classify(x_test)
        y_predict = self._test_predictions
        num_correct = 0
        num_positive = 0
        for actual, predicted in zip(y_test.T[0], y_predict):
            if actual in self.get_ys():
                if actual == predicted:
                    num_correct += 1
                num_positive += 1
        return num_correct / num_positive

    def error_rate_init(self):
        return 1 - self.true_positive_init()

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
            (_, _), (x_test, _) = load_cifar10()
        elif self.dataset_str == 'cifar100':
            (_, _), (x_test, _) = load_cifar100()
        return x_test

    def get_y(self):
        if self.dataset_str == 'cifar10':
            (_, _), (_, y_test) = load_cifar10()
        elif self.dataset_str == 'cifar100':
            (_, _), (_, y_test) = load_cifar100()
        return y_test.transpose().tolist()[0]


def from_file(filepath):
    filepath = realpath(expanduser(filepath))
    path_prefix = dirname(filepath)
    filename = splitext(basename(filepath))[0]
    classifier_id, dataset_id = filename.split('_', maxsplit=1)
    hdf5_file = join_path(dirname(filepath), classifier_id + '.hdf5')
    assert file_exists(hdf5_file), 'Cannot find neural network file for {}'.format(classifier_id)
    classifier = NeuralNetwork(hdf5_file)
    utils = ImageUtils(dataset_id)
    dataset = ImageDataset(dataset_id)
    return RegretTrial(classifier, utils, dataset, path_prefix=path_prefix)

def train_neural_network(int_labels, batch_size, num_epochs, dataset_str, verbose=False, output_path='images', checkpoint=0):
    if file_exists(output_path):
        assert isdir(output_path), '"{}" exists but is not a directory'.format(output_path)
    else:
        mkdir(output_path)

    # load the data
    if dataset_str == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        num_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = load_cifar100()
        num_classes = 100

    # convert class vectors to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

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
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    # initiate RMSprop optimizer
    optimizer = rmsprop(lr=0.0001, decay=1e-6)
    # train the model using RMSprop
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=[categorical_accuracy],
    )
    # other arguments
    kwargs = {}
    # verbosity (show progress bar)
    if verbose:
        kwargs['verbose'] = 1
    else:
        kwargs['verbose'] = 0
    # save intermediate models at every epoch
    if checkpoint > 0:
        kwargs['callbacks'] = [
            ModelCheckpoint(
                join_path(
                    output_path,
                    '{}-l{}-b{:02d}-e{{epoch:03d}}.hdf5'.format(
                        dataset_str,
                        str(ints_to_binary(int_labels)),
                        batch_size,
                    ),
                ),
                period=checkpoint,
            ),
        ]

    # fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        **kwargs,
    )

    filepath = join_path(
        output_path,
        '{}-l{}-b{:02d}-e{:03d}.hdf5'.format(
            dataset_str,
            str(ints_to_binary(int_labels)),
            batch_size,
            num_epochs,
        ),
    )
    model.save(filepath)

    # list of paths saved
    if checkpoint > 0:
        filepaths = [filepath]
    else:
        filepaths = [
            join_path(
                output_path,
                '{}-l{}-b{:02d}-e{:03d}.hdf5'.format(
                    dataset_str,
                    str(ints_to_binary(int_labels)),
                    batch_size,
                    epoch,
                ),
            )
            for epoch in range(0, num_epochs, checkpoint)
        ]

    return filepaths


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
    while p == 100 and AQUARIUM_FISH_Y in int_labels:
        int_labels = sample(range(p), k)
    return sorted(int_labels)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'], help='the dataset to use')
    arg_parser.add_argument('--labels', action='append', help='labels to use')
    arg_parser.add_argument('--batch-size', type=int, default=32, help='size of each batch')
    arg_parser.add_argument('--num-epochs', type=int, default=200, help='number of epochs to run')
    arg_parser.add_argument('--verbose', action='store_true', help='show animated progress bar')
    arg_parser.add_argument('--checkpoint', action='store_true', help='store snapshots of the network at every epoch')
    arg_parser.add_argument('--directory', action='store', default=datetime.now().isoformat(), help='output directory')
    args = arg_parser.parse_args()
    args.int_labels = sorted(set(args.int_labels))
    if args.dataset == 'cifar10':
        assert all(0 <= k < 10 for k in args.int_labels)
    elif args.dataset == 'cifar100':
        assert all(0 <= k < 100 for k in args.int_labels)
    train_neural_network(
        args.int_labels,
        args.batch_size,
        args.num_epochs,
        args.dataset,
        output_path=args.directory,
        verbose=args.verbose,
        checkpoint=args.checkpoint,
    )


if __name__ == '__main__':
    main()
