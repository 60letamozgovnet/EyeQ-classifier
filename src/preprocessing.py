import pandas as pd
import cv2 as cv
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import gc
import logging

from get_path import get_config

config = get_config()
# относительно корневой директории
DATA_DIR = config['DATA_DIR'] 
SRC_DIR = config["SRC_DIR"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def augmentation():
    rotate_transform = transforms.Compose([
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Resize((224, 224))

    ])
    
    clahe_transform = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    return rotate_transform, clahe_transform

def crop_img(img_np):
    gray = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    coords = cv.findNonZero(thresh)

    if coords is None:
        return img_np

    x, y, w, h = cv.boundingRect(coords)
    cropped = img_np[y:y+h, x:x+w]
    return cropped

def preprocessing(img_path, output_dir):
    rotate_transform, clahe_transform = augmentation()

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = crop_img(np.array(img))  # обрезаем чёрные края

        # CLAHE обработка
        channels = cv.split(img_np)
        clahe_applied = [clahe_transform.apply(c) for c in channels]
        img_clahe = cv.merge(clahe_applied)

        img_pil = Image.fromarray(img_clahe).resize((224, 224), Image.BILINEAR)
        img_np = np.array(img_pil)

        # Преобразование в тензор с аугментацией
        img_tensor = rotate_transform(img_pil)
        img_tensor_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        img_pil = Image.fromarray((img_tensor_np * 255).astype(np.uint8))
        img_pil.save(output_path)
        del img, img_tensor, img_np, img_pil, img_tensor_np, channels, clahe_applied, img_clahe
        gc.collect()
    
    except Exception as e:
        logger.info(f"Error: {e}")
    
def dataset_preprocess(input_dir, output_dir, csv_paths = None):
    if csv_paths:
        path_to_test_csv = csv_paths[0]
        path_to_train_csv = csv_paths[1]

    else:
        path_to_test_csv = os.path.join(DATA_DIR, "Label_EyeQ_test.csv")
        path_to_train_csv = os.path.join(DATA_DIR, "Label_EyeQ_train.csv")
        
        test_df = pd.read_csv(path_to_test_csv)
        train_df = pd.read_csv(path_to_train_csv)

    df = pd.concat([test_df.drop(test_df.columns[0], axis=1), train_df.drop(train_df.columns[0], axis=1)], ignore_index=True, axis=0)
    img_names = df['image'].values
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        # img_path = f"{img_path}.jpeg"
        if os.path.exists(img_path):
            preprocessing(img_path, output_dir)
            logger.info(f"Image saved. PATH: {img_path}")
        else:
            logger.info(f"Image not found. PATH: {img_path}")
    


if __name__ == "__main__":
    input_dir = os.path.join(DATA_DIR, 'train')# r"D:\code\python\ML_PIR\main\data\train"
    output_dir = os.path.join(DATA_DIR, "preprocessed") #"D:/code/python/ML_PIR/main/data/preprocessed"

    dataset_preprocess(input_dir, output_dir)