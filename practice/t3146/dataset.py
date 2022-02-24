from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class TrainDataset_01(Dataset):
    def __init__(self, img_paths, labels, transform = None):
        self.img_paths = img_paths
        self.labels = labels
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 384), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}
    
    def __len__(self):
        return len(self.img_paths)

# Mission

# 2. 강의때 보여드렸던 torchvision에 내장된 여러 Augmentation 함수와 albumentation 라이브러리의 여러 transform 기법을 적용해보세요. 
# 적용해 보신 뒤에 실제로 어떻게 변환되어 나오는지 확인해보세요. 
# 아마 plot형태로 그려서 확인해야 할거에요. 
# 그리고 이러한 Transforms를  추가한 Dataset이 과연 어느 정도의 성능을 가지는지 체크해보세요. 
# 혹여 너무 무거운 프로세스라면 생각보다 느리게 동작하겠죠? 