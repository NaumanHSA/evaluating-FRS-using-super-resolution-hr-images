import cv2
from tensorflow.keras.preprocessing import image
import numpy as np


def preprocess_face(img, target_size=(224, 224)):

    # TODO: resize causes transformation on base image, you should add black pixels to rezie it to target_size    
    img = cv2.resize(img, target_size)
	
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)

    #normalize input in [0, 1]
    img_pixels /= 255
    return img_pixels


def print_train_results(results):
    best_acc = None
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, params in zip(means, stds, params):
        # print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
        if params == results.best_params_:
            best_acc = round(mean, 3)
    print('\nBEST PARAMS: {} | acc: {} (+/-{})\n'.format(results.best_params_,
                                                         best_acc, round(std * 2, 3)))
