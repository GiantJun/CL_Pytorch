import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Normalization(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Normalization, self).__init__()
        self.n_channels=n_channels
        if mean is None:
            mean = [.0] * n_channels
        if std is None:
            std = [.1] * n_channels
        self.mean = torch.tensor(list(mean)).reshape((1, self.n_channels, 1, 1))
        self.std = torch.tensor(list(std)).reshape((1, self.n_channels, 1, 1))
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
    
    def forward(self, x):
        y = (x - self.mean / self.std)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, normed=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.norm = Normalization([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        self.normed = normed

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.out_dim = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normed:
            x = self.norm(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        dim = x_4.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        pooled = pool(x_4)
        # features = pooled.view(x.size(0), -1)
        features = torch.flatten(pooled, 1)
        # return {
        #     'fmaps': [x_1, x_2, x_3, x_4],
        #     'features': features
        # }
        return self.fc(features)

    def is_fc(self, name):
        if "fc" in name:
            return True
        else:
            return False

    def is_bn(self, name):
        if "bn" in name or "downsample.1" in name:
            return True
        else:
            return False


def resnet18_cbam(pretrained=False, normed=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], normed=normed, **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


if __name__ == "__main__":
    # model = resnet18_cbam()
    # # pretrained_dict = torch.load("../saved_parameters/imagenet200_simsiam_pretrained_model.pth")

    # for name, param in model.named_parameters():
    #     if "bn" in name or "downsample.1" in name:
    #     #     print(name)
    #     # if "conv" not in name and "downsample.0" not in name and "norm" not in name:
    #     #     print(name)
    #     # if "conv" not in name and "downsample.0" not in name:
    #         print(name)

    pretrained_dict = torch.load("../saved_parameters/imagenet200_simsiam_pretrained_model.pth")
    dst_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if "conv" not in k and "downsample.0" not in k:
            dst_dict[k] = v
    torch.save(dst_dict, "../saved_parameters/imagenet200_simsiam_pretrained_model_bn.pth")
            

    # state_dict = model.state_dict()
    # for k, v in pretrained_dict.items():
    #     print(k)
    # print(pretrained_dict["conv1.weight"][0])
    # print(state_dict["layer4.1.bn2.num_batches_tracked"])
    # print(pretrained_dict["layer4.1.bn2.num_batches_tracked"])

    # state_dict.update(pretrained_dict)

    # model.load_state_dict(state_dict)
    # print(model.state_dict()["layer4.1.bn2.running_mean"])
    # for k, v in pretrained_dict.items():
    #     if "featureExactor" in k and "fc" not in k:
    #         changed_dict[".".join(k.split(".")[1:])] = v
    
    # torch.save(changed_dict, "/data/junjie/code/zhengjin/saved_parameters/imagenet200_simsiam_pretrained_model.pth")