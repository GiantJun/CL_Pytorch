from backbone.inc_net import IncrementalNet
from torch.nn import functional as F
from typing import Callable, Iterable
import random
from torch import nn
import torch
import copy

class Class_CNNPromptNet(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, gamma=1., layer_names:Iterable[str]=[], mode=None):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4', 'fc']
        '''
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        # self.gamma = gamma
        self.mode = mode
        self.layer_names = layer_names
        self.prompt_class_num = 0

        self._batch_target = None
        self._is_training = False

        self.mode = mode

        model_dict = dict([*self.feature_extractor.named_modules()]) 
        for layer_id in layer_names:
            layer = model_dict[layer_id]
            if 'add' in self.mode or 'minus' in self.mode:
                layer.register_forward_pre_hook(self.add_minus_prompt(layer_id))
            elif 'cat' in self.mode:
                layer.register_forward_pre_hook(self.cat_prompt(layer_id))
            elif 'conv1x1_only' in self.mode:
                layer.register_forward_pre_hook(self.apply_conv1x1_only(layer_id))
            else:
                raise ValueError('Unknown add prompt mode: {} !'.format(self.mode))

    def cat_prompt(self, layer_id: str) -> Callable:
        def hook(module, input):
            if layer_id == 'fc':
                b, dim = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((self.prompt_class_num, 1, dim)).cuda()))
                    self._logger.info('Created class-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
            else:
                b, c, h, w = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((self.prompt_class_num, 1, h, w)).cuda()))
                    self._logger.info('Created class-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                    conv_1x1_list = nn.ModuleList()
                    for i in range(self.prompt_class_num):
                        conv_1x1_list.append(nn.Conv2d(c+1, c, 1).cuda())
                        self._logger.info('Created class{}-specific {}-conv1 {} -> {}'.format(i, layer_id, c+1, c))
                    self.register_module(layer_id+'_conv_1x1', conv_1x1_list)
            
            # this part are same for fc and layerN    
            if self._is_training:
                correct_prompt_batch, wrong_prompt_batch = [], []
                for i in range(len(self._batch_target)):
                    correct_img_prompt = torch.cat((input[0][i], getattr(self, layer_id+'_prompt')[self._batch_target[i]]), dim=0)
                    wrong_prompt_id = random.randint(0, self.prompt_class_num-1)
                    while wrong_prompt_id == self._batch_target[i]:
                        wrong_prompt_id = random.randint(0, self.prompt_class_num-1)
                    wrong_img_prompt = torch.cat((input[0][i], getattr(self, layer_id+'_prompt')[wrong_prompt_id]), dim=0)
                    if layer_id != 'fc':
                        correct_prompt_batch.append(getattr(self, layer_id+'_conv_1x1')[self._batch_target[i]](correct_img_prompt.unsqueeze(0)))
                        wrong_prompt_batch.append(getattr(self, layer_id+'_conv_1x1')[wrong_prompt_id](wrong_img_prompt.unsqueeze(0)))
                
                correct_prompt_batch = torch.cat(correct_prompt_batch, dim=0)
                wrong_prompt_batch = torch.cat(wrong_prompt_batch, dim=0)
                return torch.cat((correct_prompt_batch, wrong_prompt_batch), dim=0)
            else: # testing
                result = []
                for sample_id in range(b):
                    for class_prompt_id in range(self.prompt_class_num):
                        img_prompt = torch.cat((input[0][sample_id], getattr(self, layer_id+'_prompt')[class_prompt_id]), dim=0).unsqueeze(0)
                        if layer_id != 'fc':
                            img_prompt = getattr(self, layer_id+'_conv_1x1')[class_prompt_id](img_prompt)
                        result.append(img_prompt)
                return (torch.cat(result, dim=0),)
        return hook
    
    def add_minus_prompt(self, layer_id: str) -> Callable:
        def hook(module, input):
            if layer_id == 'fc':
                b, dim = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((self.prompt_class_num, 1, dim)).cuda()))
                    self._logger.info('Created class-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))

            else:
                b, c, h, w = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((self.prompt_class_num, 1, h, w)).cuda()))
                    self._logger.info('Created class-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                
            # the following are same for fc or layerN
            if self._is_training:
                correct_prompt, wrong_prompt = [], []
                input_copy = copy.deepcopy(input[0])
                for i in range(len(self._batch_target)):
                    if layer_id != 'fc':
                        correct_prompt.append(getattr(self, layer_id+'_prompt')[self._batch_target[i]].expand(*input[0].shape[1:]).unsqueeze(0))
                    else:
                        correct_prompt.append(getattr(self, layer_id+'_prompt')[self._batch_target[i]])
                    rand_prompt_id = random.randint(0, self.prompt_class_num-1)
                    while rand_prompt_id == self._batch_target[i]:
                        rand_prompt_id = random.randint(0, self.prompt_class_num-1)
                    if layer_id != 'fc':
                        wrong_prompt.append(getattr(self, layer_id+'_prompt')[rand_prompt_id].expand(*input[0].shape[1:]).unsqueeze(0))
                    else:
                        wrong_prompt.append(getattr(self, layer_id+'_prompt')[rand_prompt_id])
                
                correct_prompt_batch = torch.cat(correct_prompt, dim=0)
                wrong_prompt_batch = torch.cat(wrong_prompt, dim=0)

                if 'add' in self.mode:
                    return torch.cat((input[0]+correct_prompt_batch, input_copy+wrong_prompt_batch), dim=0)
                elif 'minus' in self.mode:
                    return torch.cat((input[0]-correct_prompt_batch, input_copy-wrong_prompt_batch), dim=0)
                else:
                    raise ValueError('Unknown mode: {}'.format(self.mode))
            else: # testing
                result = []
                for batch_id in range(b):
                    for class_prompt_id in range(self.prompt_class_num):
                        if layer_id != 'fc':
                            epanded_prompt = getattr(self, layer_id+'_prompt')[class_prompt_id].expand(*input[0].shape[1:])
                        else:
                            epanded_prompt = getattr(self, layer_id+'_prompt')[class_prompt_id]
                        if 'add' in self.mode:
                            result.append((input[0][batch_id] + epanded_prompt).unsqueeze(0))
                        elif 'minus' in self.mode:
                            result.append((input[0][batch_id] - epanded_prompt).unsqueeze(0))
                        else:
                            raise ValueError('Unknown mode: {}'.format(self.mode))
                return (torch.cat(result, dim=0),)
        return hook
    
    def apply_conv1x1_only(self, layer_id: str) -> Callable:
        def hook(module, input):
            if layer_id == 'fc':
                raise ValueError('layer fc is not supported in conv1x1_only mode!')
            else:
                b, c, h, w = input[0].shape
                if not hasattr(self, layer_id+'_conv_1x1'):
                    conv_1x1_list = nn.ModuleList()
                    for i in range(self.prompt_class_num):
                        conv_1x1_list.append(nn.Conv2d(c, c, 1).cuda())
                        self._logger.info('Created class{}-specific {}-conv1 {} -> {}'.format(i, layer_id, c, c))
                    self.register_module(layer_id+'_conv_1x1', conv_1x1_list)
            
            if self._is_training:
                correct_prompt_batch, wrong_prompt_batch = [], []
                for i in range(len(self._batch_target)):
                    wrong_prompt_id = random.randint(0, self.prompt_class_num-1)
                    while wrong_prompt_id == self._batch_target[i]:
                        wrong_prompt_id = random.randint(0, self.prompt_class_num-1)
                    
                    correct_prompt_batch.append(getattr(self, layer_id+'_conv_1x1')[self._batch_target[i]](input[0][i].unsqueeze(0)))
                    wrong_prompt_batch.append(getattr(self, layer_id+'_conv_1x1')[wrong_prompt_id](input[0][i].unsqueeze(0)))
                
                correct_prompt_batch = torch.cat(correct_prompt_batch, dim=0)
                wrong_prompt_batch = torch.cat(wrong_prompt_batch, dim=0)
                return torch.cat((correct_prompt_batch, wrong_prompt_batch), dim=0)
            else: # testing
                result = []
                for sample_id in range(b):
                    for class_prompt_id in range(self.prompt_class_num):
                        img_prompt = getattr(self, layer_id+'_conv_1x1')[class_prompt_id](input[0][sample_id].unsqueeze(0))
                        result.append(img_prompt)
                return (torch.cat(result, dim=0),)
        return hook

    def forward(self, x):
        self._is_training = False
        features = self.feature_extractor(x)
        out = self.fc(features)
        self.output_features['features'] = features
        return out, self.output_features
    
    def train_forward(self, x, target):
        self._batch_target = target
        self._is_training = True
        features = self.feature_extractor(x)
        out = self.fc(features)
        self.output_features['features'] = features
        return out, self.output_features
    
    def update_fc(self, nb_classes):
        self.prompt_class_num += nb_classes
        if 'cat' in self.mode and 'fc' in self.layer_names:
            fc = self.generate_fc(self.feature_dim*2, nb_classes)
        elif 'class_plus_one' in self.mode:
            fc = self.generate_fc(self.feature_dim, nb_classes+1)
        else:
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


class Task_CNNPromptNet(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, gamma=1., layer_names:Iterable[str]=[], mode=None):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4']
        '''
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.gamma = gamma
        self.mode = mode
        self.layer_names = layer_names

        model_dict = dict([*self.feature_extractor.named_modules()]) 
        for layer_id in layer_names:
            layer = model_dict[layer_id]
            if 'cat' in mode:
                layer.register_forward_pre_hook(self.cat_prompt(layer_id))
            elif 'add' in mode or 'minus' in mode:
                layer.register_forward_pre_hook(self.add_minus_prompt(layer_id))
            elif 'conv1x1_only' in mode:
                layer.register_forward_pre_hook(self.apply_conv1x1_only(layer_id))
            else:
                raise ValueError('Unknown mode {} !'.format(mode))

    def cat_prompt(self, layer_id: str) -> Callable:
        def hook(module, input):
            if layer_id == 'fc':
                b, dim = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((1, dim)).cuda()))
                    self._logger.info('Created task-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                    if 'residual' in self.mode and 'alpha' in self.mode:
                        self.register_parameter(layer_id+'_alpha', nn.Parameter(torch.ones(1).cuda()))
                cated_input = torch.cat((input[0], getattr(self, layer_id+'_prompt').expand(b, -1)), dim=1)
                if 'residual' in self.mode:
                    if 'alpha' in self.mode:
                        return (getattr(self, layer_id+'_alpha') * cated_input + input[0],)
                    else:
                        return (cated_input + input[0],)
                else:
                    return (cated_input,)
            else:
                b, c, h, w = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((1, h, w)).cuda()))
                    self._logger.info('Created task-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                    self.register_module(layer_id+'_conv_1x1', nn.Conv2d(c+1, c, 1).cuda())
                    self._logger.info('Created task-specific {}-conv1 {} -> {}'.format(layer_id, c+1, c))
                    if 'residual' in self.mode and 'alpha' in self.mode:
                        self.register_parameter(layer_id+'_alpha', nn.Parameter(torch.ones(1).cuda()))
                cated_input = torch.cat((input[0], getattr(self, layer_id+'_prompt').expand(b, 1, -1, -1)), dim=1)
                if 'residual' in self.mode:
                    if 'alpha' in self.mode:
                        return (getattr(self, layer_id+'_alpha') * getattr(self, layer_id+'_conv_1x1')(cated_input)+input[0],)
                    else:
                        return (getattr(self, layer_id+'_conv_1x1')(cated_input)+input[0],)
                else:
                    return (getattr(self, layer_id+'_conv_1x1')(cated_input),)
        return hook
    
    def add_minus_prompt(self, layer_id: str) -> Callable:
        def hook(module, input):
            if 'residual' in self.mode:
                raise ValueError('add_minus_prompt do not support residual option yet!')
            if layer_id == 'fc':
                b, dim = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((1, dim)).cuda()))
                    self._logger.info('Created task-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                return (input[0] + self.gamma * getattr(self, layer_id+'_prompt').expand(b, -1),)
            else:
                b, c, h, w = input[0].shape
                if not hasattr(self, layer_id+'_prompt'):
                    self.register_parameter(layer_id+'_prompt', nn.Parameter(torch.rand((1, h, w)).cuda()))
                    self._logger.info('Created task-specific {}-prompt with shape {}'.format(layer_id, getattr(self, layer_id+'_prompt').shape))
                if 'add' in self.mode:
                    return (input[0] + self.gamma * getattr(self, layer_id+'_prompt').expand(b, c, -1, -1),)
                elif 'minus' in self.mode:
                    return (input[0] - self.gamma * getattr(self, layer_id+'_prompt').expand(b, c, -1, -1),)
                else:
                    raise ValueError('Unkown mode in add_minus prompt: {}'.format(self.mode))
        return hook
    
    def apply_conv1x1_only(self, layer_id: str) -> Callable:
        def hook(module, input):
            b, c, h, w = input[0].shape
            if not hasattr(self, layer_id+'_conv_1x1'):
                self.register_module(layer_id+'_conv_1x1', nn.Sequential(*[nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU()]).cuda())
                self._logger.info('Created task-specific {}-conv1 {} -> {}'.format(layer_id, c, c))
                if 'residual' in self.mode and 'alpha' in self.mode:
                    self.register_parameter(layer_id+'_alpha', nn.Parameter(torch.ones(1).cuda()))
            if 'residual' in self.mode:
                if 'alpha' in self.mode:
                    return (getattr(self, layer_id+'_alpha') * getattr(self, layer_id+'_conv_1x1')(input[0])+input[0],)
                else:
                    return (getattr(self, layer_id+'_conv_1x1')(input[0])+input[0],)
            else:
                return (getattr(self, layer_id+'_conv_1x1')(input[0]),)
        return hook

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.fc(features)
        self.output_features['features'] = features
        return out, self.output_features
    
    def update_fc(self, nb_classes):
        if self.mode == 'cat' and 'fc' in self.layer_names:
            fc = self.generate_fc(self.feature_dim*2, nb_classes)
        else:
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


class ProtoNet(IncrementalNet):
    
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, nb_proxy=1, dist_mode=None, use_MLP=False):
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.dist_mode = dist_mode
        self.nb_proxy = nb_proxy
        self._proto_type = None
        self._use_MLP = use_MLP
        self._MLP = nn.Identity()

    def update_fc(self, nb_classes):
        # do not support multi-steps yet
        if self._proto_type == None:
            self._proto_type = nn.Parameter(torch.randn(self.nb_proxy, nb_classes, self.feature_dim).cuda())
            self._logger.info('created proto_type with shape {}'.format((nb_classes, self.feature_dim)))
            if self._use_MLP:
                self._MLP = nn.Sequential(*[nn.Linear(self.feature_dim, self.feature_dim),
                        nn.ReLU(),
                        nn.Linear(self.feature_dim, self.feature_dim)
                            ])
    
    def forward(self, x, targets=None):
        features = self.feature_extractor(x)
        MLP_features = self._MLP(features)
        self.output_features['features'] = MLP_features
        all_dist, sub_dist = [], []
        for i in range(self.nb_proxy):
            if 'cosine' in self.dist_mode: # dist: [b, class_num]
                dist = F.linear(F.normalize(MLP_features, p=2, dim=1), F.normalize(self._proto_type[i], p=2, dim=1))
            elif 'euclidean' in self.dist_mode: # dist: [b, class_num]
                dist = torch.cdist(MLP_features, self._proto_type[i], p=2)
            all_dist.append(dist.unsqueeze(1))
            if targets is not None:
                sub_dist.append(dist.gather(1, targets.unsqueeze(1)).unsqueeze(1))
        if len(all_dist) > 1:
            cated_all_dist = torch.cat(all_dist, dim=1) # cat: [b, nb_proxy, class_num]
            if targets is None: # model is in testing
                if 'cosine' in self.dist_mode:
                    if 'close_cluster_test' in self.dist_mode: # cosine value -- choose the bigger one (more similar)
                        min_proxy_idx = torch.tensor([torch.argmax(cated_all_dist[i]).item()//cated_all_dist.shape[2] for i in range(cated_all_dist.shape[0])])
                        output = cated_all_dist.gather(1, min_proxy_idx.reshape(-1,1,1).repeat(1,1,cated_all_dist.shape[-1]).cuda()).squeeze()
                    elif 'mean_cluster_test' in self.dist_mode:
                        output = cated_all_dist.mean(dim=1)
                elif 'euclidean' in self.dist_mode:
                    if 'close_cluster_test' in self.dist_mode: # euclidean value -- choose the smaller one (more similar)
                        min_proxy_idx = torch.tensor([torch.argmin(cated_all_dist[i]).item()//cated_all_dist.shape[2] for i in range(cated_all_dist.shape[0])])
                        output = cated_all_dist.gather(1, min_proxy_idx.reshape(-1,1,1).repeat(1,1,cated_all_dist.shape[-1]).cuda()).squeeze()
                    elif 'mean_cluster_test' in self.dist_mode:
                        output = cated_all_dist.mean(dim=1)
            else: # model in training
                if 'cosine' in self.dist_mode:
                    if 'pull_far_cluster' in self.dist_mode: # cosine value -- the smaller the less similar
                        sub_output = torch.cat(sub_dist, dim=1) # b, nb_proxy, 1
                        min_idx = torch.argmin(sub_output, dim=1, keepdim=True) # b, 1
                        output = cated_all_dist.gather(1, min_idx.repeat(1,1,cated_all_dist.shape[-1])).squeeze()
                    elif 'pull_close_cluster' in self.dist_mode: # cosine value -- the bigger the more similar
                        sub_output = torch.cat(sub_dist, dim=1) # b, nb_proxy, 1
                        max_idx = torch.argmax(sub_output, dim=1, keepdim=True) # b, 1
                        output = cated_all_dist.gather(1, max_idx.repeat(1,1,cated_all_dist.shape[-1])).squeeze()
                    elif 'pull_mean_cluster' in self.dist_mode:
                        output = cated_all_dist.mean(dim=1) # cat: [b, nb_proxy, class_num]
                elif 'euclidean' in self.dist_mode:
                    if 'pull_far_cluster' in self.dist_mode: # euclidean value -- the bigger the less similar
                        sub_output = torch.cat(sub_dist, dim=1) # b, nb_proxy, 1
                        max_idx = torch.argmax(sub_output, dim=1, keepdim=True) # b, 1
                        output = cated_all_dist.gather(1, max_idx.repeat(1,1,cated_all_dist.shape[-1])).squeeze()
                    elif 'pull_close_cluster' in self.dist_mode: # euclidean value -- the smaller the more similar
                        sub_output = torch.cat(sub_dist, dim=1) # b, nb_proxy, 1
                        min_idx = torch.argmin(sub_output, dim=1, keepdim=True) # b, 1
                        output = cated_all_dist.gather(1, min_idx.repeat(1,1,cated_all_dist.shape[-1])).squeeze()
                    elif 'pull_mean_cluster' in self.dist_mode:
                        output = cated_all_dist.mean(dim=1) # cat: [b, nb_proxy, class_num]
        else:
            output = all_dist[-1].squeeze(1)
        return output, self.output_features

    def get_proxy_distinc_loss(self):
        class_proxy_dist = 0
        for i in range(self._proto_type.shape[1]):
            class_proxy = self._proto_type[:,i,:].contiguous()
            if 'euclidean' in self.dist_mode:
                proxy_dist = torch.cdist(class_proxy, class_proxy, p=2)
            elif 'cosine' in self.dist_mode:
                proxy_dist = F.linear(F.normalize(class_proxy, p=2, dim=1), F.normalize(class_proxy, p=2, dim=1))
            mask = torch.eye(self.nb_proxy, dtype=torch.bool).cuda()
            neg_proxy_dist = proxy_dist[~mask].view(proxy_dist.shape[0], -1)
            class_proxy_dist += -torch.log(torch.exp(neg_proxy_dist).sum())
            
        return class_proxy_dist / (self._proto_type.shape[1] * 2)


class ProtoNetV2(IncrementalNet):
    
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, dist_mode=None, use_MLP=False):
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.dist_mode = dist_mode
        self._prompt = None
        self._nb_proxy = 0
        self._class_num = 0
        self._use_MLP = use_MLP
        if use_MLP:
            self._MLP =  nn.Sequential(*[nn.Linear(self.feature_dim, self.feature_dim),
                        nn.ReLU(),
                        nn.Linear(self.feature_dim, self.feature_dim)
                        ])
        else:
            self._MLP = nn.Identity()
    
    def init_prompt(self, prompt):
        self._class_num = len(prompt)
        self._nb_proxy = prompt[-1].shape[0]
        prompt_cated = torch.cat(prompt, dim=0)
        self._prompt = torch.nn.Parameter(prompt_cated)

    def forward(self, x, targets=None):
        features = self.feature_extractor(x)
        MLP_features = self._MLP(features)
        prompt_features = self.feature_extractor(self._prompt).view(self._class_num, self._nb_proxy, -1)
        self.output_features['features'] = MLP_features
        all_dist, sub_dist = [], []
        for i in range(self._nb_proxy):
            if 'cosine' in self.dist_mode: # dist: [b, class_num]
                dist = F.linear(F.normalize(MLP_features, p=2, dim=1), F.normalize(prompt_features[:,i,:], p=2, dim=1))
            elif 'euclidean' in self.dist_mode: # dist: [b, class_num]
                dist = torch.cdist(MLP_features, prompt_features[:,i,:], p=2)
            all_dist.append(dist.unsqueeze(1))
            if targets is not None:
                sub_dist.append(dist.gather(1, targets.unsqueeze(1)).unsqueeze(1))
        if len(all_dist) > 1:
            if 'min_enhance' in self.dist_mode:
                cated_all_dist = torch.cat(all_dist, dim=1) # cat: [b, nb_proxy, class_num]
                sub_output = torch.cat(sub_dist, dim=1) # b, nb_proxy, 1
                max_idx = torch.argmin(sub_output, dim=1, keepdim=True) # b, 1
                output = cated_all_dist.gather(1, max_idx.repeat(1,1,cated_all_dist.shape[2])).squeeze()
            else:
                output = torch.cat(all_dist, dim=1).mean(dim=1) # cat: [b, nb_proxy, class_num]
        else:
            output = all_dist[-1].squeeze(1)
        return output, self.output_features

    def get_prompt(self):
        return self._prompt.cpu().detach().numpy() if self._prompt.is_cuda else self._prompt.detach().numpy()