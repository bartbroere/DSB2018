from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model
from keras import optimizers
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image, ImageOps

import numpy as np
import os
import generator


def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples


# def create_mask(path, width, height):
#     samples = find_all_samples(path)
#     for sample in samples:
#         sample_path = os.path.join(path, sample)
#         sample_path_masks = os.path.join(sample_path, 'masks')
#         masks = os.listdir(sample_path_masks)
#         complete_mask = np.zeros((width, height), dtype=int)
#         for mask in masks:
#             with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
#                 _mask = _mask.resize((width, height))
#                 _mask = np.array(_mask)
#                 complete_mask = np.maximum(complete_mask, _mask)
#         os.mkdir(os.path.join(sample_path, 'mask'))
#         mask_image = Image.fromarray(complete_mask.astype('uint8'), 'L')
#         mask_image.save(os.path.join(sample_path, 'mask', '{}.png'.format(sample)))


def sample_x_y(path, x_shape=(256,256), y_shape=(256,256), size=670):
    samples = os.listdir(path)
    # TODO: Shuffle
    X = np.empty((size, *x_shape, 1))
    Y = np.empty((size, *y_shape, 1))

    for i in range(size):
        with Image.open(os.path.join(path, samples[i], 'images', '{}.png'.format(samples[i]))) as x_img:
            x_img = x_img.resize(x_shape)
            x_img = x_img.convert(mode='L')
            x_img = ImageOps.autocontrast(x_img)
            x_arr = np.array(x_img) /255
            x_arr = np.expand_dims(x_arr,axis=2)
            X[i,] = x_arr
        with Image.open(os.path.join(path, samples[i], 'mask', '{}.png'.format(samples[i]))) as y_img:
            y_img = y_img.resize(y_shape)
            y_arr = np.array(y_img) /255
            y_arr = np.expand_dims(y_arr,2)
            Y[i,] = y_arr
    return X, Y


def create_model():
    filter_size = 8
    drop_rate = .5

    # img_input = Input(shape=(256,256,1))
    img_input = Input(shape=(260,260,1))

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    # Upconvolutional layers
    uconv4 = Conv2DTranspose(filters=filter_size * 8, kernel_size=2, strides=2, activation='relu', padding='same')(drop5)
    uconc4 = concatenate([drop4, uconv4], axis=3)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconc4)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconv4)

    uconv3 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='same')(uconv4)
    uconc3 = concatenate([conv3, uconv3], axis=3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconc3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconv3)

    uconv2 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='same')(uconv3)
    uconc2 = concatenate([conv2, uconv2], axis=3)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconc2)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv2)

    uconv1 = Conv2DTranspose(filters=filter_size, kernel_size=2, strides=2, activation='relu', padding='same')(uconv2)
    uconc1 = concatenate([conv1, uconv1], axis=3)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconc1)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)

    pred = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(uconv1)

    model = Model(inputs=img_input, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

if __name__ == '__main__':
    # XX, YY = sample_x_y('img')
    model_x = create_model()
    # model_x.fit(XX,YY, epochs=2, batch_size=10)
    training = os.listdir('img')[0:64]
    training_generator = keras_generator.DataGenerator(training, 'img', rotation=True, flipping=True, mirror_edges=4)
    model_x.fit_generator(generator=training_generator, epochs=4)

