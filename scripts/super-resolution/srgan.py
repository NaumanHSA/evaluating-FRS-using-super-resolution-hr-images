from model.srgan import generator
from utils import load_image
from model import resolve_single

import os, sys
import cv2
from tqdm import tqdm
import numpy as np


model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

DATASET_PATH = "datasets/FERET_80x80_lr"
OUTPUT_PATH = "datasets/FERET_80x80_EDSR_hr"

if __name__ == "__main__":

    for person in tqdm(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)

        # create dirs for output images
        if not os.path.exists(os.path.join(OUTPUT_PATH, person)):
            os.makedirs(os.path.join(OUTPUT_PATH, person))

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            new_image_path = os.path.join(OUTPUT_PATH, person, image_name)

            lr = load_image(image_path)
            sr = resolve_single(model, lr)

            img_hr = np.array(sr)
            img_hr = cv2.resize(img_hr, (80, 80))
            cv2.imwrite(new_image_path, img_hr)

            # plot_sample(lr, sr)



