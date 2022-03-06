import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class exp0_resnet18Adamax(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        



class exp1_resnet34Adamax(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.model.layer4.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x



class exp2_resnet152AdamW(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet152(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x






class exp3_resnet152Adamax_data(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet152(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x






class exp4_resnet152Adamax_Dropout(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.model.fc.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        


        



class exp5_resnet34Adamax_Dropout(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.model.fc.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        




class exp6_mnasnet_a1(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('mnasnet_a1')
        self.model.classifier = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        




        
class exp7_tf_efficientnetv2_b0(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('tf_efficientnetv2_b0',pretrained=True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        



        
class exp8_tf_efficientnetv2_b1(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('tf_efficientnetv2_b1',pretrained=True)
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        





class exp9_tf_efficientnetv2_b2(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('tf_efficientnetv2_b2',pretrained=True)
        self.model.classifier = nn.Linear(in_features=1408, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        







        
class exp10_tf_efficientnetv2_b3(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('tf_efficientnetv2_b3',pretrained=True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        


        
class exp11_efficientnet_b3_pruned(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = timm.create_model('efficientnet_b3_pruned',pretrained=True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        self.model.act2.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.classifier.weight)


    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        







# Custom Model Template
class exp12_resnet101Adamax(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.model.layer4.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        






# Custom Model Template
class exp13_resnet50Adamax(nn.Module):
    def __init__(self, num_classes: int = 18):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # models = torchvision.models.resnet152
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.model.layer4.register_forward_hook(lambda m, inp, out : F.dropout(out, p=0.5, training=False))
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x        
