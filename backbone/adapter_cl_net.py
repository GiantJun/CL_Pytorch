from backbone.inc_net import IncrementalNet
from typing import Callable, Iterable
from torch import nn
import torch
import copy

class Special_Adapter_v1(nn.Module):
    def __init__(self, in_planes:int, mid_planes:int, kernel_size:int, use_alpha=True, conv_group=1):
        super().__init__()
        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.conv = nn.Conv2d(in_planes, mid_planes, kernel_size=kernel_size, groups=conv_group)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)
        self.convTransposed = nn.ConvTranspose2d(mid_planes, in_planes, kernel_size=kernel_size, groups=conv_group)
        self.bn2 = nn.BatchNorm2d(in_planes)
        
        self.use_alpha = use_alpha
        if use_alpha:
            self.alpha = nn.Parameter(torch.ones(1)*0.02)
            print('Apply alpha!')
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        
        ### original: conv+bn+ReLU+convT+bn+ReLU ###
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.convTransposed(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.use_alpha:
            out = out * self.alpha

        return out


class CNN_Adapter_Net_CIL_V2(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None, layer_names:Iterable[str]=[], mode=None):
        '''
        layers_name can be ['conv1','layer1','layer2','layer3','layer4'] for resnet18
        '''
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        
        self.layer_names = [] if layer_names is None else layer_names
        self.task_sizes = []

        self._training_mode = 'test_mode'
        self.forward_batch_size = None

        model_dict = dict([*self.feature_extractor.named_modules()]) 
        for layer_id in self.layer_names:
            adapter_id = layer_id.replace('.', '_')+'_adapters'
            self.register_module(adapter_id, nn.ModuleList([]))
            layer = model_dict[layer_id]
            layer.register_forward_pre_hook(self.apply_adapters(adapter_id))
        
        self.seperate_fc = nn.ModuleList()
    
    def train_adapter_mode(self):
        self._training_mode = 'train_adapters'
        self._logger.info("Training mode 'train_adapters' is set !")

    def test_mode(self):
        self._training_mode = 'test_mode'
        self._logger.info("Training mode 'test_mode' is set !")
    
    def store_sample_mode(self):
        self._training_mode = 'store_sample_mode'
        self._logger.info("Training mode 'store_sample_mode' is set !")
    
    def skip_adapters_mode(self):
        self._training_mode = 'skip_adapters_mode'
        self._logger.info("Training mode 'skip_adapters_mode' is set !")

    def apply_adapters(self, adapter_id: str) -> Callable:
        def hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            b, c, h, w = input.shape
            
            if len(getattr(self, adapter_id)) < len(self.task_sizes):
                getattr(self, adapter_id).append(Special_Adapter_v1(c, c, 3).cuda())
                self._logger.info('Created task-specific adapter (special v1) before layer-{}: {}=>{}=>{}'.format(adapter_id, c, c, c))
                # init adapters' weight from previous
                if len(self.task_sizes) > 1:
                    getattr(self, adapter_id)[-1].load_state_dict(getattr(self, adapter_id)[-2].state_dict())
            
            if self._training_mode == 'skip_adapters_mode':
                return (input,)
            elif self._training_mode == 'train_adapters' or self._training_mode == 'store_sample_mode':
                return (getattr(self, adapter_id)[-1](input)+input,)
            else:
                # (self._training_mode == 'test_mode')
                if len(self.task_sizes) == 1:
                    return (getattr(self, adapter_id)[-1](input)+input,)
                else:
                    adapter_features = []
                    for i, adapter in enumerate(getattr(self, adapter_id)):
                        if b != self.forward_batch_size:
                            adapter_input = input[i*self.forward_batch_size : (i+1)*self.forward_batch_size]
                        else:
                            adapter_input = input
                        adapter_output = adapter(adapter_input)
                        adapter_features.append(adapter_output+adapter_input)
                    
                    return torch.cat(adapter_features, 0) # b * number of adapters, c, h, w
        return hook
    
    def forward(self, x):
        self.forward_batch_size = x.shape[0]

        features = self.feature_extractor(x)
        if self._training_mode == 'test_mode':
            # seperate fc
            if len(self.task_sizes) > 1:
                task_feature_list = []
                task_output_list = []
                for i in range(len(self.task_sizes)): # features: [b * number  of adapters, feature_dim]
                    task_feature = features[i*self.forward_batch_size : (i+1)*self.forward_batch_size]
                        
                    task_feature_list.append(task_feature) # features: [b, feature_dim]
                    task_output_list.append(self.seperate_fc[i](task_feature))
                
                features = torch.cat(task_feature_list, dim=1) # features: [b, feature_dim * number of adapters]
                out = torch.cat(task_output_list, dim=1) # seperate logits: [b, known_class_num]
            
            else:
                out = self.seperate_fc[0](features)
        else:
            # self._training_mode == 'train_adapters' or
            # self._training_mode == 'store_sample_mode' or
            # self._training_mode == 'skip_adapters_mode'
            out = self.aux_fc(features)

        self.output_features['features'] = features
        return out, self.output_features


    def freeze_adapters(self, mode:str):
        # freeze CNN adapters
        for layer_id in self.layer_names:
            adapter_id = layer_id.replace('.', '_') + '_adapters'
            self._logger.info('Freezing old adapters in {}'.format(layer_id))
            if 'all' in mode:
                loop_length = len(getattr(self, adapter_id))
            elif 'old' in mode:
                loop_length = len(getattr(self, adapter_id))-1
            else:
                raise ValueError('Unknow freeze_adapter mode: {}'.format(mode))
            for i in range(loop_length):
                for params in getattr(self, adapter_id)[i].parameters():
                    params.requires_grad = False
                getattr(self, adapter_id)[i].eval()
    
    def activate_new_adapters(self):
        # activate CNN adapters
        for layer_id in self.layer_names:
            adapter_id = layer_id.replace('.', '_') + '_adapters'
            self._logger.info('Activating new adapters in {}'.format(layer_id))
            for params in getattr(self, adapter_id)[-1].parameters():
                params.requires_grad = True
            getattr(self, adapter_id)[-1].train()

    def new_adapters_train(self):
        self.eval()
        for layer_id in self.layer_names:
            getattr(self, layer_id.replace('.', '_')+'_adapters')[-1].train()
    
    def update_fc(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.feature_dim, new_task_size+1)
        self._logger.info('Created aux others classifier head: {} => {}'.format(self.feature_dim, new_task_size+1))

        self.seperate_fc.append(self.generate_fc(self.feature_dim, new_task_size+1))
    
    def init_fc_from_aux(self):
        aux_weight = copy.deepcopy(self.aux_fc.weight.data)
        aux_bias = copy.deepcopy(self.aux_fc.bias.data)

        self.fc.weight.data[-aux_weight.shape[0]:, -aux_weight.shape[1]:] = aux_weight
        self.fc.bias.data[-aux_bias.shape[0]:] = aux_bias
