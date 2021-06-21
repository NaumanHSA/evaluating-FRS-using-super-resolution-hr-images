import numpy as np
import os, shutil
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import random

from net import Facenet, RatinaFaceWrapper
from utils import find_input_shape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH_TO_DATASET_IMAGES = "FERED_Dataset/Images/dvd1/data/images"
PATH_TO_DATASET_LABELS = "FERED_Dataset/Images/dvd1/data/ground_truths/name_value"
PATH_TO_CROPPED_FACES = "FERED_Dataset/Images/dvd1/data/faces"
PATH_TO_DATASET_FILE = "FERED_DATASET.npz"

DATASET_TO_VISUALIZE = "datasets/FERET_80x80"


def process_dataset():
    X, y = [], []
    facenet = Facenet.loadModel()
    face_detector = RatinaFaceWrapper.build_model()
    input_shape_x, input_shape_y = find_input_shape(facenet)

    counter = 0
    for person in tqdm(os.listdir(PATH_TO_DATASET_IMAGES)):

        person_path = os.path.join(PATH_TO_DATASET_IMAGES, person)
        label_path = os.path.join(PATH_TO_DATASET_LABELS, person, person + ".txt")
        person_face_path = os.path.join(PATH_TO_CROPPED_FACES, person)

        if not os.path.exists(person_face_path):
            os.makedirs(person_face_path)

        gt = dict()
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                attr, val = line.split("=")
                gt[attr] = val.replace("\n", "")

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            # read the image. Convert into numpy array
            img = cv2.imread(image_path)

            # detect a face in the image
            detected_face, box = RatinaFaceWrapper.detect_face(face_detector, img, align=True)

            if isinstance(detected_face, np.ndarray):

                face_array = cv2.resize(detected_face, (input_shape_y, input_shape_x))
                face_array = image.img_to_array(face_array)
                face_array = np.expand_dims(face_array, axis = 0)
                face_array /= 255   #normalize input in [0, 1]
                embeddings = facenet.predict(face_array)[0].tolist()
                
                X.append(embeddings)
                y.append(gt)

                # save the cropped face                        
                detected_face = cv2.resize(detected_face, (224, 224))
                face_path = os.path.join(person_face_path, image_name)
                cv2.imwrite(face_path, detected_face)

                if counter % 50 == 0:
                    X_, y_ = np.array(X), np.array(y)
                    np.savez(PATH_TO_DATASET_FILE, features=X_, labels=y_)

                counter += 1

    X_, y_ = np.array(X), np.array(y)
    np.savez(PATH_TO_DATASET_FILE, features=X_, labels=y_)
    print("dataset file saved successfully...")

def extract_faces_with_n_samples(n=5):
    if os.path.exists(PATH_TO_DATASET_FILE):
        data = np.load(PATH_TO_DATASET_FILE, allow_pickle=True)
        X, y = data["features"], data["labels"]

        labels = np.array([int(v["id"].replace("cfrS", "")) for v in y])
        (unique, counts) = np.unique(labels, return_counts=True)

        ids_n = unique[np.where(counts >= n)]

        for id_ in ids_n:
            id_ = str(id_).zfill(5)

            print(id_)
            src = os.path.join(PATH_TO_CROPPED_FACES, id_)
            dst = os.path.join("FERED_Dataset/Images/dvd1/data/faces_n", id_)
            shutil.copytree(src, dst)

            for image_name in os.listdir(dst):
                
                image_path = os.path.join(dst, image_name)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)

                cv2.imwrite(image_path, img)


def visualize():
    rows, cols = 3, 5
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 12))

    all_images = [os.path.join(DATASET_TO_VISUALIZE, person) for person in os.listdir(DATASET_TO_VISUALIZE)]

    for row in range(rows):
        for col in range(cols):
            index = random.randint(0, len(all_images))
            image_path = os.path.join(all_images[index], os.listdir(all_images[index])[0])
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[row][col].imshow(img)

    fig.tight_layout(pad=2.0)
    fig.suptitle('Original Dataset (80x80)', fontsize=16)
    plt.savefig('gallery/FERET_80x80.png')
    # plt.show()


if __name__ == "__main__":
    # process_dataset()
    # extract_faces_with_n_samples(n=4)
    visualize()