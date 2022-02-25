import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import cv2
from albumentations import *

train_path = '/opt/ml/input/data/train'
img_path = os.path.join(train_path, 'images')
traininfo = pd.read_csv(os.path.join(train_path, 'train.csv'))
img_paths = img_path + '/' + traininfo['path']
file_names = []
for path in img_paths:
    print(path)
    names = os.listdir(path)
    names = [name for name in names if name[0] != '.']
    for name in names:
        if name[0] == 'i' or name[0] == 'n':
            decompose = name.split('.')
            image = Image.open(os.path.join(path, name))
            image = np.array(image)
            flip = Compose([
                HorizontalFlip(p=1.0),
            ])
            rotation = Compose([
                ShiftScaleRotate(p=1.0),
            ])
            bright_contrast = Compose([
                RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0)
            ])
            composition = Compose([
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=1.0),
                RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0)
            ])
            result = Image.fromarray(flip(image=image)['image'])
            result.save(os.path.join(path, decompose[0]+'1.'+decompose[1]))
            result = Image.fromarray(rotation(image=image)['image'])
            result.save(os.path.join(path, decompose[0]+'2.'+decompose[1]))
            result = Image.fromarray(bright_contrast(image=image)['image'])
            result.save(os.path.join(path, decompose[0]+'3.'+decompose[1]))
            result = Image.fromarray(composition(image=image)['image'])
            result.save(os.path.join(path, decompose[0]+'4.'+decompose[1]))
print('augmentation complete!')