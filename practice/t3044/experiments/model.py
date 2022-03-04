import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
class MyModelResnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.resnet152.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # # xavier uniform
        # torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
        # stdv = 1. / (self.resnet152.fc.weight.shape[1] ** 0.5)
        # self.resnet152.fc.bias.data.uniform_(-stdv, stdv)

        # he-initialization
        nn.init.kaiming_normal_(self.resnet152.fc.weight)
        self.resnet152.fc.bias.data.fill_(0.)

    def forward(self, x):
        x = self.resnet152(x)
        return x

class MyModelResnet152Freezed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.resnet152.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # xavier uniform
        #torch.nn.init.xavier_uniform_(self.resnet152.fc.weight)
        #stdv = 1. / (self.resnet152.fc.weight.shape[1] ** 0.5)
        #self.resnet152.fc.bias.data.uniform_(-stdv, stdv)                
        
    def forward(self, x):
        # freeze
        for param in self.resnet152.parameters():
            param.requires_grad_(True)
        for param in self.resnet152.fc.parameters():
            param.requires_grad_(True)
        x = self.resnet152(x)
        
        return x


class MyModelResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # he-initialization
        nn.init.kaiming_normal_(self.resnet18.fc.weight)
        self.resnet18.fc.bias.data.fill_(0.)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class MyModelResnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = torchvision.models.resnet34(pretrained=True)
        self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # he-initialization
        nn.init.kaiming_normal_(self.resnet18.fc.weight)
        self.resnet18.fc.bias.data.fill_(0.)

    def forward(self, x):
        x = self.resnet18(x)
        return x

