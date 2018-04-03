from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, AveragePooling2D, Cropping2D
from keras.models import Model
from keras import optimizers
import keras.backend as K
import tensorflow as tf

from PIL import Image, ImageOps

import numpy as np
import os
import generator
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples


def compute_option(starts):
    outs = dict.fromkeys(starts)
    for start in starts:
        out = ((((((((start * 2) - 4 ) * 2 ) - 4) * 2 ) - 4) * 2) - 4)
        outs[start] = out
    return outs


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




def mean_iou(y_true, y_pred):
    y_true = K.round(y_true)
    print(y_true)
    y_pred = K.round(y_pred)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
    return score


def create_model(filter_size = 12, drop_rate=.25):
    img_input = Input(shape=(256,256,1))

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', mean_iou])
    return model


def create_inception_model():
    filter_size = 8
    drop_rate = .5

    img_input = Input(shape=(256,256,1))

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    # Conv 2 INCEPTION
    conv2_1x1 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(pool1)

    conv2_3x3 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(pool1)
    conv2_3x3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(conv2_3x3)

    conv2_5x5 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(pool1)
    conv2_5x5 = Conv2D(filters=filter_size * 2, kernel_size=5, strides=1, activation='relu', padding='same')(conv2_5x5)

    conv2_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pool1)

    conv2 = concatenate([conv2_1x1, conv2_3x3, conv2_5x5, conv2_bas], axis=3)
    pool2 = MaxPooling2D(2, 2)(conv2)

    # Conv 3 INCEPTION
    conv3_1x1 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(pool2)

    conv3_3x3 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(pool2)
    conv3_3x3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(conv3_3x3)

    conv3_5x5 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(pool2)
    conv3_5x5 = Conv2D(filters=filter_size * 4, kernel_size=5, strides=1, activation='relu', padding='same')(conv3_5x5)

    conv3_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pool2)

    conv3 = concatenate([conv3_1x1, conv3_3x3, conv3_5x5, conv3_bas], axis=3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    # Conv 4 INCPETION
    conv4_1x1 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(pool3)

    conv4_3x3 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(pool3)
    conv4_3x3 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(conv4_3x3)

    conv4_5x5 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(pool3)
    conv4_5x5 = Conv2D(filters=filter_size * 8, kernel_size=5, strides=1, activation='relu', padding='same')(conv4_5x5)

    conv4_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pool3)

    conv4 = concatenate([conv4_1x1, conv4_3x3, conv4_5x5, conv4_bas], axis=3)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    # Conv 5 INCEPTION
    conv5_1x1 = Conv2D(filters=filter_size * 16, kernel_size=1, strides=1, activation='relu', padding='same')(pool4)

    conv5_3x3 = Conv2D(filters=filter_size * 16, kernel_size=1, strides=1, activation='relu', padding='same')(pool4)
    conv5_3x3 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(conv5_3x3)

    conv5_5x5 = Conv2D(filters=filter_size * 16, kernel_size=1, strides=1, activation='relu', padding='same')(pool4)
    conv5_5x5 = Conv2D(filters=filter_size * 16, kernel_size=5, strides=1, activation='relu', padding='same')(conv5_5x5)

    conv5_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pool4)

    conv5 = concatenate([conv5_1x1, conv5_3x3, conv5_5x5, conv5_bas], axis=3)
    drop5 = Dropout(drop_rate)(conv5)

    ### Upconvolutional layers
    # UConv 4 INCEPTION
    uconv4 = Conv2DTranspose(filters=filter_size * 8, kernel_size=2, strides=2, activation='relu', padding='same')(drop5)
    uconc4 = concatenate([drop4, uconv4], axis=3)

    uconv4_1x1 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(uconc4)

    uconv4_3x3 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(uconc4)
    uconv4_3x3 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconv4_3x3)

    uconv4_5x5 = Conv2D(filters=filter_size * 8, kernel_size=1, strides=1, activation='relu', padding='same')(uconc4)
    uconv4_5x5 = Conv2D(filters=filter_size * 8, kernel_size=5, strides=1, activation='relu', padding='same')(uconv4_5x5)

    uconv4_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(uconc4)

    uconv4_fin = concatenate([uconv4_1x1, uconv4_3x3, uconv4_5x5, uconv4_bas], axis=3)

    # UConv 3 INCEPTION
    uconv3 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='same')(uconv4_fin)
    uconc3 = concatenate([conv3, uconv3], axis=3)

    uconv3_1x1 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(uconc3)

    uconv3_3x3 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(uconc3)
    uconv3_3x3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconv3_3x3)

    uconv3_5x5 = Conv2D(filters=filter_size * 4, kernel_size=1, strides=1, activation='relu', padding='same')(uconc3)
    uconv3_5x5 = Conv2D(filters=filter_size * 4, kernel_size=5, strides=1, activation='relu', padding='same')(uconv3_5x5)

    uconv3_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(uconc3)

    uconv3_fin = concatenate([uconv3_1x1, uconv3_3x3, uconv3_5x5, uconv3_bas], axis=3)

    # UConv 2 INCEPTION
    uconv2 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='same')(uconv3_fin)
    uconc2 = concatenate([conv2, uconv2], axis=3)

    uconv2_1x1 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(uconc2)

    uconv2_3x3 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(uconc2)
    uconv2_3x3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv2_3x3)

    uconv2_5x5 = Conv2D(filters=filter_size * 2, kernel_size=1, strides=1, activation='relu', padding='same')(uconc2)
    uconv2_5x5 = Conv2D(filters=filter_size * 2, kernel_size=5, strides=1, activation='relu', padding='same')(uconv2_5x5)

    uconv2_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(uconc2)

    uconv2_fin = concatenate([uconv2_1x1, uconv2_3x3, uconv2_5x5, uconv2_bas], axis=3)

    # UConv 1 INCEPTION
    uconv1 = Conv2DTranspose(filters=filter_size * 1, kernel_size=2, strides=2, activation='relu', padding='same')(uconv2_fin)
    uconc1 = concatenate([conv1, uconv1], axis=3)

    uconv1_1x1 = Conv2D(filters=filter_size * 1, kernel_size=1, strides=1, activation='relu', padding='same')(uconc1)

    uconv1_3x3 = Conv2D(filters=filter_size * 1, kernel_size=1, strides=1, activation='relu', padding='same')(uconc1)
    uconv1_3x3 = Conv2D(filters=filter_size * 1, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1_3x3)

    uconv1_5x5 = Conv2D(filters=filter_size * 1, kernel_size=1, strides=1, activation='relu', padding='same')(uconc1)
    uconv1_5x5 = Conv2D(filters=filter_size * 1, kernel_size=5, strides=1, activation='relu', padding='same')(uconv1_5x5)

    uconv1_bas = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(uconc1)

    uconv1_fin = concatenate([uconv1_1x1, uconv1_3x3, uconv1_5x5, uconv1_bas], axis=3)

    ####
    uconv1_pred = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1_fin)

    pred = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(uconv1_pred)

    model = Model(inputs=img_input, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', mean_iou])
    return model


def create_model_valid():
    filter_size = 8
    drop_rate = .5

    img_input = Input(shape=(444, 444, 1))

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = Conv2D(filters=filter_size * 1, kernel_size=3, strides=1, activation='relu', padding='valid')(pool1)
    conv2 = Conv2D(filters=filter_size * 1, kernel_size=3, strides=1, activation='relu', padding='valid')(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='valid')(pool2)
    conv3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='valid')(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='valid')(pool3)
    conv4 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='valid')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    conv5 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='valid')(pool4)
    conv5 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='valid')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    # Upconvolutional layers
    uconv4 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='valid')(drop5)
    cdrop4 = Cropping2D(((4,4),(4,4)))(drop4)
    uconc4 = concatenate([cdrop4, uconv4], axis=3)
    uconv4 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='valid')(uconc4)
    uconv4 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='valid')(uconv4)

    uconv3 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='valid')(uconv4)
    cconv3 = Cropping2D(((16,16),(16,16)))(conv3)
    uconc3 = concatenate([cconv3, uconv3], axis=3)
    uconv3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='valid')(uconc3)
    uconv3 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='valid')(uconv3)

    uconv2 = Conv2DTranspose(filters=filter_size * 1, kernel_size=2, strides=2, activation='relu', padding='valid')(uconv3)
    cconv2 = Cropping2D(((40,40),(40,40)))(conv2)
    uconc2 = concatenate([cconv2, uconv2], axis=3)
    uconv2 = Conv2D(filters=filter_size * 1, kernel_size=3, strides=1, activation='relu', padding='valid')(uconc2)
    uconv2 = Conv2D(filters=filter_size * 1, kernel_size=3, strides=1, activation='relu', padding='valid')(uconv2)

    uconv1 = Conv2DTranspose(filters=filter_size, kernel_size=2, strides=2, activation='relu', padding='valid')(uconv2)
    cconv1 = Cropping2D(((88,88),(88,88)))(conv1)
    uconc1 = concatenate([cconv1, uconv1], axis=3)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(uconc1)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='valid')(uconv1)
    uconv1 = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)

    pred = Conv2D(filters=1, kernel_size=1, strides=1, padding='valid', activation='sigmoid')(uconv1)

    model = Model(inputs=img_input, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', mean_iou])
    return model


if __name__ == '__main__':
    path_img = 'C:/Users/huubh/Dropbox/DSB'
    model_x2 = create_model()
    # model_x2 = create_inception_model()
    # model_x2 = create_model_valid()
    # labels = os.listdir('../img')
    labels = os.listdir(path_img)
    training = labels[:608]
    validation = labels[608:]
    print(len(training))
    print(len(validation))
    training_generator = generator.DataGenerator(training, path_img,
                                                 rotation=True, flipping=True, zoom=2, batch_size = 36, dim=(256,256))
    validation_generator = generator.DataGenerator(validation, path_img,
                                                 rotation=True, flipping=True, zoom=2, batch_size = 2, dim=(256,256))
    model_x2.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=36)

    # prediction_ids = ['01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf']
    #
    # prediction_generator = generator.PredictDataGenerator(prediction_ids, path_img)
    # zz = model_x2.predict_generator(prediction_generator)
    #
    # out = generator.post_process_concat(prediction_ids, zz, threshold=1)
    # one_y_pred = out['01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf']
    #
    # plt.imsave('filename2.png', one_y_pred[0,:,:,0], cmap=cm.gray)
