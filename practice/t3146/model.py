import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ModFcModel_NonFreeze_Res18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet18(pretrained=True)
        self.imagenet_resnet.fc = nn.Linear(in_features=self.imagenet_resnet.fc.in_features, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.imagenet_resnet(x)
        return x

class ModFcModel_NonFreeze_Res152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet152(pretrained=True)
        self.imagenet_resnet.fc = nn.Linear(in_features=self.imagenet_resnet.fc.in_features, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.imagenet_resnet(x)
        return x

# Custom Model Template
class ModFcModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet152(pretrained=True)
        for param in self.imagenet_resnet.parameters():
            param.requires_grad = False
        self.imagenet_resnet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.imagenet_resnet(x)
        return x


class AddOtherFCModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet152(pretrained=True)
        for param in self.imagenet_resnet.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.imagenet_resnet(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x

class ModFcModel_Freeze_Effib7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_efficientnet_b7 = torchvision.models.efficientnet_b7(pretrained=True)
        for param in self.imagenet_efficientnet_b7.parameters():
            param.requires_grad = False
        
        self.imagenet_efficientnet_b7.classifier = nn.Sequential( nn.Dropout(p=0.1, inplace=True),
                                                        nn.Linear(in_features=self.imagenet_efficientnet_b7.classifier[1].in_features, out_features=num_classes, bias=True))
        nn.init.xavier_uniform_(self.imagenet_efficientnet_b7.classifier[1].weight)
        self.imagenet_efficientnet_b7.classifier[1].bias.data.uniform_(-0.01, 0.01)


    def forward(self, x):
        x = self.imagenet_efficientnet_b7(x)
        return x


class ModFcModel_NonFreeze_Res18_Out8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet18(pretrained=True)
        self.imagenet_resnet.fc = nn.Linear(in_features=self.imagenet_resnet.fc.in_features, out_features=8, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.imagenet_resnet(x)
        return x

class ModFcModel_Freeze_Res18_Out8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.imagenet_resnet.parameters():
            param.requires_grad = False
        self.imagenet_resnet.fc = nn.Linear(in_features=self.imagenet_resnet.fc.in_features, out_features=8, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.imagenet_resnet(x)
        return x

class ModFcModel_NonFreeze_Res152_Out8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet152(pretrained=True)

        self.imagenet_resnet.fc = nn.Linear(in_features=self.imagenet_resnet.fc.in_features, out_features=8, bias=True)
        nn.init.xavier_uniform_(self.imagenet_resnet.fc.weight)
        self.imagenet_resnet.fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x = self.imagenet_resnet(x)
        return x

class CatFc_NonFreeze_Res18_Effib0_Out8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet18(pretrained=True)
        self.imagenet_efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
        
        resnet_out_features = self.imagenet_resnet.fc.in_features
        efficientnet_out_features = self.imagenet_efficientnet.classifier[1].in_features

        self.imagenet_resnet.fc = nn.Sequential( nn.Dropout(p=0.1, inplace=True) )
        self.imagenet_efficientnet.classifier = nn.Sequential( nn.Dropout(p=0.1, inplace=True) )

        self.concat_fc = nn.Linear(in_features=resnet_out_features + efficientnet_out_features, out_features=8, bias=True)
        nn.init.xavier_uniform_(self.concat_fc.weight)
        self.concat_fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x_res = self.imagenet_resnet(x)
        x_eff = self.imagenet_efficientnet(x)

        x = torch.cat((x_res, x_eff), dim=1)
        x = self.concat_fc(x)

        return x

class CatFc_Freeze_Res18_Effib0_Out8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.imagenet_resnet = torchvision.models.resnet18(pretrained=True)
        self.imagenet_efficientnet = torchvision.models.efficientnet_b0(pretrained=True)

        for param in self.imagenet_resnet.parameters():
            param.requires_grad = False
        for param in self.imagenet_efficientnet.parameters():
            param.requires_grad = False
        
        resnet_out_features = self.imagenet_resnet.fc.in_features
        efficientnet_out_features = self.imagenet_efficientnet.classifier[1].in_features

        self.imagenet_resnet.fc = nn.Sequential( nn.Dropout(p=0.1, inplace=True) )
        self.imagenet_efficientnet.classifier = nn.Sequential( nn.Dropout(p=0.1, inplace=True) )

        self.concat_fc = nn.Linear(in_features=resnet_out_features + efficientnet_out_features, out_features=8, bias=True)
        nn.init.xavier_uniform_(self.concat_fc.weight)
        self.concat_fc.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        x_res = self.imagenet_resnet(x)
        x_eff = self.imagenet_efficientnet(x)

        x = torch.cat((x_res, x_eff), dim=1)
        x = self.concat_fc(x)

        return x