import copy
from typing import Callable, Iterable

import timm.models as timm_models
import torch
import torchvision.models as torch_models
from torch import nn

from backbone.cifar_resnet import resnet32
from backbone.cifar_resnet import resnet18 as resnet18_cifar
from backbone.cifar_resnet_cbam import resnet18_cbam as resnet18_cbam
from backbone.linears import CosineLinear, SimpleLinear, SplitCosineLinear
from backbone.ucir_cifar_resnet import resnet32 as cosine_resnet32
from backbone.ucir_resnet import resnet18 as cosine_resnet18
from backbone.ucir_resnet import resnet34 as cosine_resnet34
from backbone.ucir_resnet import resnet50 as cosine_resnet50


def get_backbone(logger, backbone_type, pretrained=False, pretrain_path=None, normed=False) -> nn.Module:
    name = backbone_type.lower()
    net = None
    if name == 'resnet32':
        net = resnet32()
    elif name == 'resnet18_cifar':
        # ### resnet18 for cifar, version for dynamic_er
        # logger.info('getting model from torch...')
        # real_name = name.replace('_cifar', '')
        # weights = 'DEFAULT' if pretrained and pretrain_path is None else None
        # net = torch_models.__dict__[real_name](weights=weights)
        # net.maxpool = nn.Identity()
        # net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        assert pretrained is False, 'resnet18_cifar has no pretrain weights !'
        net = resnet18_cifar()
    elif name == 'cosine_resnet18':
        net = cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        net = cosine_resnet32()
    elif name == 'cosine_resnet34':
        net = cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        net = cosine_resnet50(pretrained=pretrained)
    elif name == 'resnet18_cbam':
        net = resnet18_cbam(normed=normed)
    elif name in torch_models.__dict__.keys():
        logger.info('getting model from torch...')
        # # for new version of pytorch (after 2022.7)
        # weights = 'DEFAULT' if pretrained and pretrain_path is None else None
        # net = torch_models.__dict__[name](weights=weights)
        
        # for older version of pytorch
        net = torch_models.__dict__[name](pretrained=pretrained)
    elif name in timm_models.__dict__.keys():
        logger.info('getting model from timm...')
        net = timm_models.create_model(backbone_type, pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(backbone_type))
    logger.info('Created {} !'.format(name))

    # 载入自定义预训练模型
    if pretrain_path is not None and pretrained:
        pretrained_dict = torch.load(pretrain_path)
        if 'state_dict' in pretrained_dict.keys():
            adjusted_dict = {}
            for key, value in pretrained_dict['state_dict'].items():
                if 'fc' not in key:
                    adjusted_dict[key.replace('feature_extractor.', '')] = value
            pretrained_dict = adjusted_dict
            # pretrained_dict = pretrained_dict['state_dict']
        elif 'model' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['model'].state_dict()

        state_dict = net.state_dict()
        logger.info('special keys not loaded in pretrain model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logger.info("loaded pretrained_dict_name: {}".format(pretrain_path))
    elif pretrained:
        logger.info("loaded pretrained weights from default")

    return net


class IncrementalNet(nn.Module):

    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names: Iterable[str]=[], MLP_projector=False):
        super(IncrementalNet, self).__init__()
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4']
        '''
        self._logger = logger
        self.feature_extractor = get_backbone(self._logger, backbone_type, pretrained, pretrain_path)
        self._layer_names = layer_names
        if 'resnet' in backbone_type:
            self._feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        elif 'efficientnet' in backbone_type:
            self._feature_dim = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier = nn.Dropout(p=0.4, inplace=True)
        elif 'vit' in backbone_type:
            self._feature_dim = self.feature_extractor.num_features
            self.feature_extractor.head = nn.Identity()
        elif 'mobilenet' in backbone_type:
            self._feature_dim = self.feature_extractor.classifier[-1].in_features
            self.feature_extractor.classifier = nn.Dropout(p=0.2, inplace=False)
        elif 'vgg16_bn' in backbone_type:
            self._feature_dim = self.feature_extractor.classifier[-1].in_features
            self.feature_extractor.classifier = self.feature_extractor.classifier[:-1]
        else:
            raise ValueError('{} did not support yet!'.format(backbone_type))
        self._logger.info("Removed original backbone--{}'s fc classifier !".format(backbone_type))
        self.fc = None
        if MLP_projector:
            self.MLP_projector = nn.Sequential(nn.Linear(self._feature_dim, self._feature_dim),
                                            nn.ReLU(), nn.Linear(self._feature_dim, self._feature_dim))
        else:
            self.MLP_projector = nn.Identity()
        self.seperate_fc = nn.ModuleList()
        self.output_features = {}
        self.task_sizes = []
        self.set_hooks()
        
    def set_hooks(self):
        if len(self._layer_names)>0:
            model_dict = dict([*self.feature_extractor.named_modules()]) 
            for layer_id in self._layer_names:
                layer = model_dict[layer_id]
                layer.register_forward_hook(self.save_output_features(layer_id, self.output_features))
                self._logger.info('Registered forward hook for {}'.format(layer_id))
        # return self

    def save_output_features(self, layer_id: str, save_dict: dict) -> Callable:
        def hook(module, input, output):
            save_dict[layer_id] = output
        return hook

    @property
    def feature_dim(self):
        return self._feature_dim

    def extract_features(self, x):
        features = self.feature_extractor(x)
        return features

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.MLP_projector(features)
        self.output_features['features'] = features
        if len(self.seperate_fc) == 0 and self.fc is not None:
            out = self.fc(features)
            if isinstance(out, tuple):
                # for special classifier head which return (Tensor, Dict), e.g. cosineLinear
                self.output_features.update(out[1])
                out = out[0]
        elif len(self.seperate_fc) > 0 and self.fc is None:
            out = []
            for task_head in self.seperate_fc:
                out.append(task_head(features))
            out = torch.cat(out, dim=-1) # b, total_class
        elif len(self.seperate_fc) == 0 and self.fc is None:
            out = features    # for methods that do not need fc
        else:
            raise ValueError('Seperate FC or FC should not appear at once')
        return out, self.output_features

    def update_fc(self, nb_classes):
        self.task_sizes.append(nb_classes - sum(self.task_sizes))
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc
    
    def update_seperate_fc(self, nb_cur_class):
        self.task_sizes.append(nb_cur_class)
        self.seperate_fc.append(self.generate_fc(self.feature_dim, nb_cur_class))

    def generate_fc(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)

    def copy(self):
        """ Warning: this method will reset output_features! """
        self.output_features = {}
        copy_net = copy.deepcopy(self)
        self._logger.info("Setting copied network's hooks...")
        copy_net.set_hooks()
        self._logger.info("Reseting current network's hooks...")
        self.set_hooks()
        return copy_net

    def freeze_FE(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self._logger.info('Freezing feature extractor(requires_grad=False) ...')
        return self
    
    def activate_FE(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.feature_extractor.train()
        self._logger.info('Activating feature extractor(requires_grad=True) ...')
        return self
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self._logger.info('Freezing the whole network(requires_grad=False) ...')
        return self
    
    def activate(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        self._logger.info('Activating the whole network(requires_grad=True) ...')
        return self
    
    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm = (torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:] *= gamma
    
    def reset_fc_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)


class CosineIncrementalNet(IncrementalNet):

    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names: Iterable[str]=[], nb_proxy=1):
        super().__init__(logger, backbone_type, pretrained, pretrain_path, layer_names)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_id):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_id == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, bias_correction=False):
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.fc(features)
        if self.bias_correction:
            logits = out
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
        self.output_features['features'] = features
        return logits, self.output_features

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def freeze_bias_layers(self):
        for param in self.bias_layers.parameters():
            param.requires_grad = False
    
    def activate_bias_layers(self):
        for param in self.bias_layers.parameters():
            param.requires_grad = True       

class SimpleCosineIncrementalNet(IncrementalNet):

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight,nextperiod_initialization])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
