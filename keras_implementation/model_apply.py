# from keras_implementation import generator
# from keras_implementation import pipeline
import generator, pipeline
import keras.metrics
keras.metrics.mean_iou = pipeline.mean_iou
from keras import models
import os


# def predict_plot_label(ids, path, model):
#     for i in range(len(ids)):
#         if i%10 == 0:
#             prediction_generator = generator.PredictDataGenerator([ids[i]], path)
#             pred = model.predict_generator(prediction_generator)
#             out = generator.post_process_concat([ids[i]], pred, threshold=4)
#             generator.plot_image_true_mask([ids[i]][0], out, path)
#     return

if __name__ == '__main__':
    path_img = '/Users/HuCa/Dropbox/DSB'
    path_img = '/Users/Huca/Documents/DSB2018/Test/stage1_test'
    labels = os.listdir(path_img)[1:]
    print(labels)
    validation = labels[:]

    model_x5 = models.load_model('model_x5.h5')

    prediction_ids = validation[:]


    prediction_generator = generator.PredictDataGenerator(prediction_ids[:], path_img)
    zz = model_x5.predict_generator(prediction_generator)

    out = generator.post_process_concat(prediction_ids[:], zz, threshold=1)

    out_true = generator.post_process_original_size(out, path_img)

    for ids, out_arra in out_true.items():
        if ids == '9ab2d381f90b485a68b82bc07f94397a0373e3215ad20935a958738e55f3cfc2':
            generator.plot_image_true_mask(ids, out_arra, path_img)

    # out_true = post_process_original_size(out, path_img)
