import pandas as pd
import numpy as np

import os
import shutil
from tqdm import tqdm
import argparse


def mask_labels(masked:str) -> int:
    mask, incorrect, normal = 0, 1, 2
    
    if masked.startswith('mask'):
        return mask
    elif masked.startswith('in'):
        return incorrect
    else:
        return normal

def age_labels(age:str) -> int:
    young, middle, old = 0, 1, 2
    age = int(age)
    
    if age < 30:
        return young
    elif age < 60:
        return middle
    else:
        return old
    
def gender_labels(gender:str) -> int:
    male, female = 0, 1
    
    if gender == 'male':
        return male
    else:
        return female
    
def augment_list_from_profiles(data_dir:str, n_augmentation:int = 1000) -> pd.Series:
    ''' 
    (나이*성별)의 분포를 균등하게 증가시킬 프로필을 모두 담은 pd.Series를 반환합니다.  
    '''
    
    profiles = os.listdir(data_dir)
    indices = list()
    for profile in profiles:
        if profile.startswith('.'):
            continue
        
        id, gender, race, age = profile.split('_')
        gender_num, age_num = gender_labels(gender), age_labels(age)
        indices.append((profile, gender, age, gender_num * 3 + age_num))

    profiles = pd.DataFrame(indices, columns=['path', 'gender', 'age', 'class_age_gender'])

    counted_values = profiles.class_age_gender.value_counts()
    assert counted_values.max() < n_augmentation

    augmented_profiles = profiles.copy()
    for ag_class in range(len(counted_values)):
        augment = pd.DataFrame()
        for i in range((n_augmentation - counted_values[ag_class]) // counted_values[ag_class]):
            augment = pd.concat([augment, profiles[profiles.class_age_gender == ag_class].copy()])
        
        sample_augment = profiles[profiles.class_age_gender == ag_class].sample(n = n_augmentation - len(augment) - counted_values[ag_class])
        augment = pd.concat([augment, sample_augment])
        augmented_profiles = pd.concat([augmented_profiles, augment])

    return augmented_profiles.path

def augment_profiles(augmented_profiles:pd.Series, data_dir:str = None, augment_dir:str = None) -> None:
    '''  
    입력으로 들어오는 pd.Series 정보를 기준으로 
    프로필(나이*성별)의 분포를 균등하게 증가(복사)시킵니다. 
    <주의> 환경에 따라 시간이 오래걸릴 수 있습니다.
    '''
    
    if data_dir == None:
        data_dir = './train/images'
    if augment_dir == None:
        augment_dir = './train/images_augmented'    
    
    if os.path.exists(augment_dir):
        print('augment_dir is already exists !! ')
        return
    
    else:
        print('augment_profiles...')
        for idx, profile in enumerate(tqdm(augmented_profiles)):
            data_dir_path = os.path.join(data_dir, profile)
            data_copy_path = os.path.join(augment_dir, str(idx) + profile)
            shutil.copytree(data_dir_path, data_copy_path)        

        print('augment_profiles done !')
        print()
    
def augment_non_mask(augment_dir:str = None):
    '''
    마스크 착용 여부(mask, incorrect, normal)를 기준으로 
    데이터의 분포를 귱등하기 증가(복사) 시킵니다.
    <주의> 환경에 따라 시간이 오래걸릴 수 있습니다.
    '''
    
    if augment_dir == None:
        augment_dir = './train/images_augmented'    
    
    profiles = os.listdir(augment_dir)
    if 'normal2.jpg' in os.listdir(augment_dir + '/' + profiles[0]):
        print('augment_non_mask is already done !! ')
        return
    else:        
        print('augment_non_mask...')
        for profile in tqdm(profiles):
            img_folder = augment_dir + '/' + profile
            for file_name in os.listdir(img_folder):
                if file_name.startswith('.'):
                    continue
                
                if file_name.split('.')[0].startswith('incorrect') or file_name.split('.')[0].startswith('normal'):
                    for i in range(4):
                        file = img_folder + '/' + file_name
                        file, ext = os.path.splitext(file)
                        shutil.copy(file + ext, file + str(i + 2) + ext)            
        
        print('augment_non_mask is done !')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_augmentation', type=int, default=1000, help='n_augmentation에 맞게 프로필(나이*성별)을 증강시킵니다 (기본값: 1000)')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/images', help='증강시킬(복사시킬) 이미지의 경로를 지정합니다.')
    parser.add_argument('--augment_dir', type=str, default='/opt/ml/input/data/train/images_augmented_1000', help='증강시킨(복사된) 이미지의 경로를 지정합니다.')
    
    args = parser.parse_args()
    print(args)
    
    n_augmentation = args.n_augmentation
    data_dir = args.data_dir
    augment_dir  = args.augment_dir
    
    augmented_profiles = augment_list_from_profiles(data_dir=data_dir, n_augmentation=n_augmentation)
    augment_profiles(augmented_profiles=augmented_profiles, data_dir=data_dir, augment_dir=augment_dir)
    augment_non_mask(augment_dir=augment_dir)