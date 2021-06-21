import numpy as np
import os
from tqdm import tqdm
import cv2

from net import Facenet
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import joblib
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
from utils import find_input_shape, preprocess_face, print_train_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_TO_DATASET_DIR = "datasets/FERET_80x80_EDSR_hr"
PATH_TO_CLASSIFIER = "models/FERET_80x80_EDSR_hr.h5"

PATH_TO_DATASET_FILE = "datasets/FERET_80x80_EDSR_hr.npz"
FACENET_WEIGHTS = "weights/facenet_weights.h5 "


def process_dataset(dataset_path):
    X, y = [], []
    facenet = Facenet.loadModel(weights=FACENET_WEIGHTS)
    input_shape_x, input_shape_y = find_input_shape(facenet)

    for person in tqdm(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        label = float(person.split("-")[1])

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            # read image. Convert to numpy array
            img = cv2.imread(image_path)

            img = preprocess_face(img=img, target_size=(input_shape_y, input_shape_x))
            embeddings = facenet.predict(img)[0].tolist()

            X.append(embeddings)
            y.append(label)

    X, y = np.array(X), np.array(y)
    np.savez(PATH_TO_DATASET_FILE, features=X, labels=y)
    return X, y

   
def train(X, y):

    print("starting training ....")
     # training SVC classifier
    svc = SVC()
    # defining parameter range
    parameters = {
        'C': [0.01, 1, 5, 100, 500],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['linear', 'poly'],
        'probability': [True]
    }

    # --------------------------------------------------------
    svccv = GridSearchCV(svc, parameters, cv=5, n_jobs=-1)
    svccv.fit(X, y)
    print_train_results(svccv)

    joblib.dump(svccv.best_estimator_, PATH_TO_CLASSIFIER)
    print('model saved successfully...')
    # --------------------------------------------------------

def test(X, y):
    clf = joblib.load(PATH_TO_CLASSIFIER)
    predictions = clf.predict(X)
    accuracy = accuracy_score(y, predictions)

    correct = np.where(y == predictions)[0]
    print(f"total correct classification: {correct.shape[0]}")
    print(f"total wrong classification: {len(predictions) - correct.shape[0]}")
    print(f"accuracy on test set: {accuracy}")

    
if __name__ == "__main__":

    if os.path.exists(PATH_TO_DATASET_FILE):
        print("dataset found...")
        data = np.load(PATH_TO_DATASET_FILE)
        X, y = data["features"], data["labels"]
    else:
        print("processing images...")
        X, y = process_dataset(PATH_TO_DATASET_DIR)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Total Samples shape: {X.shape[0]}, # of classes: {len(np.unique(y))}")
    print(f"training samples: {X_train.shape[0]}")
    print(f"testing samples: {X_test.shape[0]}")

    if not os.path.exists(PATH_TO_CLASSIFIER):
        train(X_train, y_train)
    else:
        print("trained model found...")

    print("evaluating model now...")    
    test(X_test, y_test)