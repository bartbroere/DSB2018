import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize


class BatchGenerator(object):

    def __init__(self, height, width, channels, data_dir_train, data_dir_test, submission_run):

        self.height = height
        self.width = width
        self.channels = channels

        self.x_train, self.y_train = self.read_train_data(data_dir_train)
        self.shuffle()

        if submission_run:
           self.x_val, self.y_val = self.x_train, self.y_train
        else:
            self.x_val, self.y_val = self.x_train[:100], self.y_train[:100]
            self.x_train, self.y_train = self.x_train[100:], self.y_train[100:]

        self.x_test = self.read_test_data(data_dir_test)

        self.cursor = 0


    def read_train_data(self, data_dir):

        train_ids = next(os.walk(data_dir))[1]
        images = np.zeros((len(train_ids), self.height, self.width, self.channels), dtype=np.uint8)
        labels = np.zeros((len(train_ids), self.height, self.width, 1), dtype=np.bool)
        sys.stdout.flush()

        print('Getting and resizing train images ... ')
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = data_dir + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.channels]
            img = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
            images[n] = img
            mask = np.zeros((self.height, self.width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.height, self.width), mode='constant',
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            labels[n] = mask

        x_train = images
        y_train = labels

        return x_train, y_train


    def read_test_data(self, data_dir):

        test_ids = next(os.walk(data_dir))[1]
        x_test = np.zeros((len(test_ids), self.height, self.width, self.channels), dtype=np.uint8)
        sizes_test = []
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = data_dir + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :self.channels]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (self.height, self.width), mode='constant', preserve_range=True)
            x_test[n] = img

        return x_test, test_ids, sizes_test

        
    def shuffle(self):

        indices = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        print("Every day I'm shuffling.")


    def generate_batch(self, batch_size):


        x_batch = []
        y_batch = []

        for _ in range(batch_size):

            x = self.x_train[self.cursor]
            y = self.y_train[self.cursor]
            self.cursor += 1
            if self.cursor == len(self.x_train):
                self.shuffle()
                self.cursor = 0

            x_batch.append(x)
            y_batch.append(y)

        return np.array(x_batch).reshape([batch_size, self.height, self.width, self.channels]), np.array(y_batch).reshape([batch_size, self.height, self.width, 1])


    def generate_val_data(self):

        return self.x_val, self.y_val


    def generate_test_data(self):

        return self.x_test