from model import resolve_single
from model.wdsr import wdsr_b
from utils import load_image

import os
import cv2
from tqdm import tqdm
import numpy as np


model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('weights/wdsr-b-32-x4/weights.h5')

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