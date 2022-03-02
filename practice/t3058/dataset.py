import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from random import randint
import albumentations as A
# import albumentations as A

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.7,0.4,0.1,0),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD

from glob import glob
class TrainDataset(Dataset):    #Dataset만 받아야 한다.
    def __init__(self, transform='', val_ratio=0.2, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.transform = transform
        XX = glob('/opt/ml/input/data/train/images/*/*')
        self.X = []
        self.Y = []
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.num_classes = 18
        self.resize = [512,384]
        self.cnt = 0
        self.transform = transforms.Compose([
            Resize(self.resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std),
        ])

        self.transform2 = [transforms.Grayscale(3),
            transforms.Pad(randint(4,50)),
            transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.4,p=1.0),
            RandomRotation(degrees=40), 
            transforms.ColorJitter(0.7,0.4,0.1,0),
        ]
        
        for i,img_dir in enumerate(XX):
            tmp = img_dir.split("/")
            if tmp[-1][0]=='.': continue
            self.X.append(img_dir)
            mask,gender_age = tmp[-1],tmp[-2].split("_")
            gender,age = gender_age[1],int(gender_age[-1])

            if age<30:
                age = 0
            elif 30<=age<60:
                age = 1
            else:
                age = 2

            if gender[0]=='m':
                gender = 0
            else:
                gender = 1

            if mask[0] == 'i':
                mask = 1
            elif mask[0] == 'n':
                mask = 2
            else:
                mask = 0
            
            label = 6*mask+3*gender+age
            self.Y.append(label)

            
            # if label in [2,7,13]:
            #     for j in range(5):
            #         self.X.append(img_dir)
            #         self.Y.append(label)
            # elif label in [5,6,12]:
            #     for j in range(4):
            #         self.X.append(img_dir)
            #         self.Y.append(label)
            # elif label in [4,8,11,17]:
            #     for j in range(20):
            #         self.X.append(img_dir)
            #         self.Y.append(label)
            # elif label in [15,16]:
            #     for j in range(2):
            #         self.X.append(img_dir)
            #         self.Y.append(label)
                    
                


    def __getitem__(self, index):
        image = Image.open(self.X[index])
        if self.transform:
            tmp = self.randAugment(randint(index%3,len(self.transform2)))
            image = tmp(image)
            image = self.transform(image)
            
        if randint(0,10)<5:
            image = AddGaussianNoise()(image)    
        return image, self.Y[index]

    def __len__(self):
        return len(self.X)

    def split_dataset(self):
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """

        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        # train_set, val_set = self[:n_train], self
        return train_set, val_set
    
    def set_transform(self, transform):
        self.transform = transform
    def set_resize(self, resize):
        self.resize = resize
        
    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def randAugment(self,N):
        sample = list(np.random.choice(self.transform2, size = N))
        random.shuffle(sample)
        c= transforms.Compose(sample)
        return c

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = A.Compose([
                A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), always_apply=True),
                ])
        self.setup()
        self.calc_statistics()

        self.transform2 = [
            A.CLAHE(clip_limit = 2,always_apply=True),
            A.ColorJitter(0.2,0.4,0.1,0,always_apply=True),
            A.FancyPCA(alpha = 0.2,always_apply=True),
            A.GaussNoise(var_limit = 100,always_apply=True),
            A.RandomFog(fog_coef_lower=0.01,fog_coef_upper=0.5,always_apply=True),
            A.ShiftScaleRotate(shift_limit= 0.02, scale_limit= 0, rotate_limit=6,always_apply=True),
            A.RGBShift(r_shift_limit=3,always_apply=True),
            A.Sharpen(alpha = (1.0,1.0),always_apply=True)
            
        ]

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image = A.Compose([
            A.CenterCrop(512,164,always_apply=True),
            A.Resize(512,328,always_apply=True)]
        )(image = np.array(image))['image']
        tmp = self.randAugment(randint(index%3,len(self.transform2)))
        image = tmp(image = np.array(image))['image']
        image_transform = transforms.ToTensor()(self.transform(image = np.array(image))['image'])
            
        # if randint(0,10)<5:
        #     image_transform = AddGaussianNoise()(image_transform)
        return image_transform, [mask_label,gender_label,age_label]

    def randAugment(self,N):
        sample = list(np.random.choice(self.transform2, size = N))
        random.shuffle(sample)
        c= A.Compose(sample)
        return c
        
    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio= 5):
        # self.indices = defaultdict(list)
        # self.indices = []
        self.X = []
        self.Y = []
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    # def appending(self, img_path,mask_label,gender_label,age_label,phase):
    def appending(self, img_path,mask_label, gender_label, age_label):
        self.image_paths.append(img_path)
        self.mask_labels.append(mask_label)
        self.gender_labels.append(gender_label)
        self.age_labels.append(age_label)
        # self.indices[phase].append(self.cnt)
        self.personX.append(self.cnt)
        self.l[self.encode_multi_class(mask_label,gender_label,age_label)]+=1

    def setup(self):
        self.l = [0 for _ in range(18)]
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        # split_profiles = self._split_profile(profiles, self.val_ratio)

        self.cnt = 0
        # for phase, indices in split_profiles.items():
            # for _idx in indices:
        for _idx in range(len(profiles)):
            profile = profiles[_idx]
            img_folder = os.path.join(self.data_dir, profile)
            self.personX = []
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                # self.appending(img_path,mask_label,gender_label,age_label,phase)
                label = self.encode_multi_class(mask_label,gender_label,age_label)
                self.appending(img_path,mask_label, gender_label, age_label)
                
                if label in [0]:
                    self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [1]:
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [2,7,13]:
                    for j in range(9):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [5]:
                    for j in range(8):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [6,12]:
                    for j in range(7):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [11]:
                    for j in range(45):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [8,14]:
                    for j in range(50):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [17]:
                    for j in range(44):
                        self.appending(img_path,mask_label, gender_label, age_label)
                elif label in [9,10,15,16]:
                    for j in range(5):
                        self.appending(img_path,mask_label, gender_label, age_label)
                self.cnt+=1
            self.X.append(self.personX)
        self.cnt = len(self.X)//self.val_ratio
        before = 0
        tmpX = []
        while True:
            if before+self.cnt>len(self.X):
                tmpX.append(self.X[before:])
                break
            else:
                tmpX.append(self.X[before:before+self.cnt])
                before+=self.cnt
        self.X = tmpX
        print(self.l[:9])
        print(self.l[9:])

    def split_dataset(self,idx) -> List[Subset]:
        val_indices = []
        for i in self.X[idx]:
            val_indices+=i
        train_indices = []
        for i in range(len(self.X)):
            if i!=idx:
                for j in self.X[i]:
                    train_indices+=j
        train_set, val_set = Subset(self,train_indices), Subset(self, val_indices)
        return [train_set,val_set]
        # return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
