import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import label
from skimage.transform import resize
import pandas as pd

from neural_net import NeuralNet
from batch_generator import BatchGenerator
import tensorflow as tf

batchgen = BatchGenerator(128, 128, 3, 'stage1_train/', 'stage1_test/')

x_train = batchgen.x_train
x_val = batchgen.x_val
x_test, test_ids, sizes_test = batchgen.x_test

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

model = NeuralNet(128, 128, batchgen)

#model.load_weights('/home/sander/kaggle/models/neural_net400.ckpt')

loss_list, val_loss_list = model.train(num_steps=1000,
             batch_size=32,
             dropout_rate=0,
             lr=.0001,
             decay=1,
             checkpoint='models/neural_net')

plt.plot(loss_list)
plt.plot(val_loss_list)
plt.legend(['train_loss', 'val_loss'])
plt.show()

###########################
# Validatie
###########################

x_val, y_val = batchgen.generate_val_data()
val_preds = model.predict(x_val)
n=1

plt.imshow(x_val[n].reshape(128, 128, 3))
plt.imshow(y_val[n].reshape(128, 128))
plt.imshow(val_preds[n].reshape(128, 128), cmap='gray')
plt.imshow(np.round(val_preds[n].reshape(128, 128)), cmap='gray')


def jaccard_coef(y_true, y_pred, smooth = 1e-12):

    intersection = np.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = np.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return np.mean(jac)


def jaccard_coef_int(y_true, y_pred, smooth = 1e-12):

    y_pred_pos = tf.round(tf.clip(y_pred, 0, 1))

    intersection = tf.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = tf.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return tf.mean(jac)

###########################
# Submission
###########################

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

preds = model.predict(x_test)
preds = (preds > 0.5).astype(np.uint8)

preds_test_upsampled = []
for i in range(len(preds)):
    preds_test_upsampled.append(resize(np.squeeze(preds[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []

for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)


plt.imshow(x_test[5].reshape(128,128, 3))
plt.imshow(preds[5].reshape(128,128), cmap='gray')
