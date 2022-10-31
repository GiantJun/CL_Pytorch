import copy
import random
from typing import Callable, Iterable

import timm.models as timm_models
import torch
import torchvision.models as torch_models
from torch import nn
from torch.nn import functional as F

from backbone.cifar_resnet import resnet32
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
        net = torch_models.__dict__[name](pretrained=pretrained)
    elif name in timm_models.__dict__.keys():
        net = timm_models.create_model(backbone_type, pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(backbone_type))
    logger.info('Created {} !'.format(name))

    # 载入自定义预训练模型
    if pretrain_path != None and pretrained:
        pretrained_dict = torch.load(pretrain_path)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = torch.load(pretrain_path)['state_dict']
        state_dict = net.state_dict()
        logger.info('special keys in load model state dict: {}'.format(pretrained_dict.keys()-state_dict.keys()))
        for key in (pretrained_dict.keys() & state_dict.keys()):
            state_dict[key] = pretrained_dict[key]
        net.load_state_dict(state_dict)

        logger.info("loaded pretrained_dict_name: {}".format(pretrain_path))

    return net


class IncrementalNet(nn.Module):

    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names: Iterable[str]=[], use_MLP=False):
        super(IncrementalNet, self).__init__()
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4']
        '''
        self._logger = logger
        self.feature_extractor = get_backbone(self._logger, backbone_type, pretrained, pretrain_path)
        self._use_MLP = use_MLP
        self._layer_names = layer_names
        if 'resnet' in backbone_type:
            self._feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        elif 'efficientnet' in backbone_type:
            self._feature_dim = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier = nn.Dropout(p=0.4, inplace=True)
        else:
            raise ValueError('{} did not support yet!'.format(backbone_type))
        self._logger.info("Removed original backbone-{}'s fc classifier !".format(backbone_type))
        self.fc = None
        self.fc_til = None
        self.output_features = {}
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
        out = self.fc(features)
        self.output_features['features'] = features
        if isinstance(out, tuple):
            self.output_features.update(out[1])
            out = out[0]
        return out, self.output_features
    
    def forward_til(self, x, task_id):
        features = self.feature_extractor(x)
        out = self.fc_til[task_id](features)
        self.output_features['features'] = features
        if isinstance(out, tuple):
            self.output_features.update(out[1])
            out = out[0]
        return out, self.output_features

    def update_fc(self, nb_classes):
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

    def update_til_fc(self, nb_classes):
        if self.fc_til == None:
            self.fc_til = nn.ModuleList([])
        self.fc_til.append(self.generate_fc(self.feature_dim, nb_classes))

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def copy(self):
        """ Warning: this method will reset output_features! """
        self.output_features = {}
        copy_net = copy.deepcopy(self)
        self._logger.info('Setting copy network hooks...')
        copy_net.set_hooks()
        self._logger.info('Reseting current network hooks...')
        self.set_hooks()
        return copy_net

    def freeze_FE(self):
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False
        self.eval()
        self._logger.info('Freezing feature extractor ...')
        return self
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.eval()
        self._logger.info('Freezing the whole network ...')
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

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, logger, backbone_type, pretrained):
        super(DERNet,self).__init__()
        self._logger = logger
        self.backbone_type = backbone_type
        self.feature_extractor = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.feature_extractor)

    def extract_features(self, x):
        features = [fe(x) for fe in self.feature_extractor]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [fe(x) for fe in self.feature_extractor]
        all_features = torch.cat(features, 1)

        out=self.fc(all_features) #{logics: self.fc(features)}

        aux_logits=self.aux_fc(features[-1])

        return out, {"aux_logits":aux_logits, "features":all_features}

    def update_fc(self, nb_classes):
        if len(self.feature_extractor)==0:
            self.feature_extractor.append(get_backbone(self._logger, self.backbone_type))
            self.out_dim = self.feature_extractor[-1].fc.in_features
            self.feature_extractor[-1].fc = nn.Identity()
        else:
            self.feature_extractor.append(get_backbone(self._logger, self.backbone_type))
            self.feature_extractor[-1].fc = nn.Identity()
            self.feature_extractor[-1].load_state_dict(self.feature_extractor[-2].state_dict())
            
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        # fc = SimpleLinear(in_dim, out_dim)
        fc = nn.Linear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def reset_fc_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)


class SimpleCosineIncrementalNet(IncrementalNet):

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
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
