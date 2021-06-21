import os 
import cv2
from tqdm import tqdm


DATASET_PATH = "datasets/FERET_80x80"
OUTPUT_PATH = "datasets/FERET_80x80_lr"

if __name__ == "__main__":

    for person in tqdm(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)

        # create dirs for output images
        if not os.path.exists(os.path.join(OUTPUT_PATH, person)):
            os.makedirs(os.path.join(OUTPUT_PATH, person))

        for i, image_name in enumerate(os.listdir(person_path)):
                
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (52, 52), interpolation=cv2.INTER_AREA)

            new_image_path = os.path.join(OUTPUT_PATH, person, image_name)
            cv2.imwrite(new_image_path, img)

