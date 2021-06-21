from scipy.spatial.distance import cosine
import cv2

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization

from tensorflow.keras.preprocessing import image
import numpy as np


def compare_two_faces(known_embedding, candidate_embedding, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return True if cosine(known_embedding, candidate_embedding) <= tolerance else False


def compare_faces(known_embeddings, candidate_embedding, tolerance=0.4):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    # calculate distance between embeddings
    distances = [cosine(known_embedding, candidate_embedding)
                        for known_embedding in known_embeddings]
    min_dist = min(distances)
    min_dist_index = distances.index(min_dist)
    return min_dist_index if min_dist <= tolerance else None

def preprocess_face(img, target_size=(224, 224), align=True):
        
    img = cv2.resize(img, target_size)
	# TODO: resize causes transformation on base image, you should add black pixels to rezie it to target_size
    
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    return img_pixels


def find_input_shape(model):

	# face recognition models have different size of inputs
	# my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape

def Classifier(input_dim, outputs):
    clmodel = Sequential()
    clmodel.add(Dense(units=200, input_dim=input_dim, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.5))
    clmodel.add(Dense(units=100, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.4))
    clmodel.add(Dense(units=10, kernel_initializer='glorot_uniform'))
    clmodel.add(BatchNormalization())
    clmodel.add(Activation('tanh'))
    clmodel.add(Dropout(0.2))
    clmodel.add(Dense(units=outputs, kernel_initializer='he_uniform'))
    clmodel.add(Activation('softmax'))

    clmodel.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        optimizer='adam', 
        metrics=['accuracy']
        )

    return clmodel



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
