import torch
import copy
from torch import nn

from backbone.inc_net import get_backbone

class DERNet(nn.Module):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None):
        super(DERNet,self).__init__()
        self._logger = logger
        self.backbone_type = backbone_type
        self.feature_extractor = nn.ModuleList()
        self.pretrained = pretrained
        self.pretrain_path = pretrain_path
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []

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
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        ft = get_backbone(self._logger, self.backbone_type, pretrained=self.pretrained, pretrain_path=self.pretrain_path)
        if 'resnet' in self.backbone_type:
            feature_dim = ft.fc.in_features
            ft.fc = nn.Identity()
        elif 'efficientnet' in self.backbone_type:
            feature_dim = ft.classifier[1].in_features
            ft.classifier = nn.Dropout(p=0.4, inplace=True)
        elif 'mobilenet' in self.backbone_type:
            feature_dim = ft.classifier[-1].in_features
            ft.classifier = nn.Dropout(p=0.2, inplace=False)
        else:
            raise ValueError('{} did not support yet!'.format(self.backbone_type))

        if len(self.feature_extractor)==0:
            self.feature_extractor.append(ft)
            self._feature_dim = feature_dim
        else:
            self.feature_extractor.append(ft)
            self.feature_extractor[-1].load_state_dict(self.feature_extractor[-2].state_dict())
        
        self.aux_fc=self.generate_fc(self._feature_dim, new_task_size+1)
            
        fc = self.generate_fc(self._feature_dim*len(self.feature_extractor), nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:-self._feature_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc


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