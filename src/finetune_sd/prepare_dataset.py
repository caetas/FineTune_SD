import os
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import matplotlib.pyplot as plt

def open_image(bytes_string):
    image = Image.open(BytesIO(bytes_string))
    image = np.array(image)
    return image

def create_sketch(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # edges
    edges = cv2.Canny(blur, 100, 200)
    # threshold
    ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    # dilate
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    return thresh

# open parquet file
df = pd.read_parquet('./../../data/raw/pokemons.parquet')

if not os.path.exists('./../../data/processed/pokemons/'):
    os.makedirs('./../../data/processed/pokemons/')
if not os.path.exists('./../../data/processed/pokemons/train/'):
    os.makedirs('./../../data/processed/pokemons/train/')

save_dir = './../../data/processed/pokemons/train'

# for each image, open it
for i in range(len(df)):
    image = open_image(df['image'][i]['bytes'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prompt = df['text'][i]
    sketch = create_sketch(image)
    # save the images
    # img names are 0000.png, 0001.png, etc.
    img_name = str(i).zfill(4) + '.png'
    cv2.imwrite(os.path.join(save_dir, img_name), image)
    # sketch names follow the same pattern
    sketch_name = str(i).zfill(4) + '_mask.png'
    cv2.imwrite(os.path.join(save_dir, sketch_name), sketch)
    # save img_name, sketch_name and prompt in a metadata.jsonl file
    with open(os.path.join(save_dir, 'metadata.jsonl'), 'a') as f:
        f.write('{"image": "' + img_name + '", "conditioning_image": "' + sketch_name + '", "text": "' + prompt + '"}\n')