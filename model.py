import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models

import settings

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ResBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.pre_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.features = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.features(x)
        return x


class Delf_classification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(1024, num_classes)
        self.attention_layers = nn.Sequential(BasicBlock(1024, 512, 1), nn.Conv2d(512, 1, 1))
        self.softplus = nn.Softplus()

    def forward(self, x):  
        attention_score = self.attention_layers(x)
        attention_prob = self.softplus(attention_score)
        if settings.attention_type == 'use_l2_normalized_feature':
            attention_feature_map = F.normalize(x, p=2, dim=1)  # l2 normalize per channel
        elif settings.attention_type == 'use_default_input_feature':
            attention_feature_map = x

        attention_feat = torch.mean(torch.mean(attention_prob * attention_feature_map, dim=2, keepdim=True), dim=3, keepdim=True)  # or called prelogits


        fc = self.fc(attention_feat.view(attention_feat.shape[0], -1))
        return fc, attention_prob

class Resnet_classification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        self.res_layer4 = nn.Sequential(
            resnet.layer4,
            resnet.avgpool
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.res_layer4(x) 
        fc = self.fc(x.view(x.shape[0], -1))
        return fc


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 224, 224)
    vr = Variable(ts)
    net = DELF()
    #class_model = Delf_classification(settings.num_classes)
    class_model = Resnet_classification(settings.num_classes)
    #print(net)
    attention_feat, attention_prob, attention_score, x = net(vr)
    #oups = class_model(attention_feat)
    oups = class_model(x)
    for oup in oups:
        print(oup.size())

