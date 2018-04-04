# from keras_implementation import generator
# from keras_implementation import pipeline
import generator, pipeline
import keras.metrics
keras.metrics.mean_iou = pipeline.mean_iou
from keras import models
import os


if __name__ == '__main__':
    path_img = './Test/stage1_test'
    labels = os.listdir(path_img)[1:]
    print(labels)
    prediction_ids = labels[:]

    model_x5 = models.load_model('model_x88.h5')


    prediction_generator = generator.PredictDataGenerator(prediction_ids[:], path_img)
    predictions = model_x5.predict_generator(prediction_generator)

    out_square = generator.post_process_concat(prediction_ids[:], predictions, threshold=4)

    out_true = generator.post_process_original_size(out_square, path_img)

    for ids, out_arra in out_true.items():
        generator.plot_image_true_mask(ids, out_arra, path_img)

    # out_true = post_process_original_size(out, path_img)
