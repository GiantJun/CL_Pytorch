import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache
from timm.models.layers import DropPath, trunc_normal_

from backbone.inc_net import IncrementalNet
from backbone.dytox_resnet import resnet18


class DytoxNet(IncrementalNet):
    def __init__(self, logger, backbone_type, pretrained, pretrain_path=None):
        assert backbone_type == 'resnet18', 'Only resnet18 is supported!'
        super().__init__(logger, backbone_type, pretrained, pretrain_path)
        self.feature_extractor = resnet18(pretrained=pretrained)
        self.feature_extractor.head = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 504, kernel_size=1),
                nn.BatchNorm2d(504),
                nn.ReLU(inplace=True)
            )
        self.feature_extractor.avgpool = nn.Identity()
        self.feature_extractor.layer4 = nn.Identity()
        self.feature_extractor.embed_dim = 504
        self.embed_dim = self.feature_extractor.embed_dim

        self.task_tokens = nn.ParameterList()
        self.tab = Block(
            dim=self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
            attention_type=ClassAttention)
        self.task_sizes = []
        self.aux_fc = None
    
    def update_fc(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        
        token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(token, std=.02)
        self.task_tokens.append(token)
        
        self.aux_fc = self.generate_fc(self.embed_dim, new_task_size+1)
        self.seperate_fc.append(self.generate_fc(self.embed_dim, new_task_size))
    
    def forward(self, x):
        B = x.shape[0]
        tokens = []
        logits = []

        features, feature_maps = self.feature_extractor.forward_tokens(x)
        self.output_features['features'] = features
        for i in range(len(self.task_sizes)):
            task_token, attn, v = self.tab(
                torch.cat((self.task_tokens[i].expand(B, -1, -1), features), dim=1))
            tokens.append(task_token[:, 0])

            task_logits = self.seperate_fc[i](task_token[:, 0])
            logits.append(task_logits)
        
        logits = torch.cat(logits, dim=1) if len(logits) > 1 else logits[0]
        aux_logits = self.aux_fc(tokens[-1])
        self.output_features['aux_logits'] = aux_logits
        self.output_features['tokens'] = tokens
        
        return logits, self.output_features
    
    def freeze_task_tokens(self, mode:str):
        if mode == 'old':
            loop_length = len(self.task_tokens) - 1
        elif mode == 'all':
            loop_length = len(self.task_tokens)
        else:
            raise ValueError('Unknown freeze_task_token mode: {}'.format(mode))
        for i in range(loop_length):
            self.task_tokens[i].requires_grad = False
            self._logger.info('Freezing task {} \'s task tokens (requires_grad=False) ...'.format(i))
    
    def activate_task_tokens(self, mode:str):
        if mode == 'all':
            loop_length = len(self.task_tokens)
        else:
            raise ValueError('Unknown freeze_task_token mode: {}'.format(mode))
        for i in range(loop_length):
            self.task_tokens[i].requires_grad = True
            self._logger.info('Activate task {} \'s task tokens (requires_grad=True) ...'.format(i))
    
    def freeze_task_heads(self, mode:str):
        if mode =='old':
            loop_length = len(self.seperate_fc) - 1
        elif mode == 'all':
            loop_length = len(self.seperate_fc)
        else:
            raise ValueError('Unknown freeze_task_token mode: {}'.format(mode))
        for i in range(loop_length):
            for param in self.seperate_fc[i].parameters():
                param.requires_grad = False
            self._logger.info('Freezeing task {} \'s head (requires_grad=False) ...'.format(i))
    
    def activate_task_heads(self, mode:str):
        if mode == 'all':
            loop_length = len(self.seperate_fc)
        else:
            raise ValueError('Unknown freeze_task_token mode: {}'.format(mode))
        for i in range(loop_length):
            for param in self.seperate_fc[i].parameters():
                param.requires_grad = True
            self._logger.info('Activating task {} \'s tokens (requires_grad=True) ...'.format(i))
    

class BatchEnsemble(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.out_features, self.in_features = out_features, in_features
        self.bias = bias

        self.r = nn.Parameter(torch.randn(self.out_features))
        self.s = nn.Parameter(torch.randn(self.in_features))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.in_features, self.out_features, self.bias)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        result.linear.weight = self.linear.weight
        return result

    def reset_parameters(self):
        device = self.linear.weight.device
        self.r = nn.Parameter(torch.randn(self.out_features).to(device))
        self.s = nn.Parameter(torch.randn(self.in_features).to(device))

        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x):
        w = torch.outer(self.r, self.s)
        w = w * self.linear.weight
        return F.linear(x, w, self.linear.bias)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., fc=nn.Linear):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = fc(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = fc(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BatchEnsemble):
            trunc_normal_(m.linear.weight, std=.02)
            if isinstance(m.linear, nn.Linear) and m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True, fc=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, v

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1,-1)
        pos_score = self.pos_proj(pos_score).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self, locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices   = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1,-1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size,img_size)
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        self.rel_indices = rel_indices.to(device)


class Block(nn.Module):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type=GPSA,
                 fc=nn.Linear, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, fc=fc, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, fc=fc)

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.reset_parameters()
        self.mlp.apply(self.mlp._init_weights)

    def forward(self, x, mask_heads=None, task_index=1, attn_mask=None):
        if isinstance(self.attn, ClassAttention) or isinstance(self.attn, JointCA):  # Like in CaiT
            cls_token = x[:, :task_index]

            xx = self.norm1(x)
            xx, attn, v = self.attn(
                xx,
                mask_heads=mask_heads,
                nb=task_index,
                attn_mask=attn_mask
            )

            cls_token = self.drop_path(xx[:, :task_index]) + cls_token
            cls_token = self.drop_path(self.mlp(self.norm2(cls_token))) + cls_token

            return cls_token, attn, v

        xx = self.norm1(x)
        xx, attn, v = self.attn(xx)

        x = self.drop_path(xx) + x
        x = self.drop_path(self.mlp(self.norm2(x))) + x

        return x, attn, v

class JointCA(nn.Module):
    """Forward all task tokens together.

    It uses a masked attention so that task tokens don't interact between them.
    It should have the same results as independent forward per task token but being
    much faster.

    HOWEVER, it works a bit worse (like ~2pts less in 'all top-1' CIFAR100 50 steps).
    So if anyone knows why, please tell me!
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @lru_cache(maxsize=1)
    def get_attention_mask(self, attn_shape, nb_task_tokens):
        """Mask so that task tokens don't interact together.

        Given two task tokens (t1, t2) and three patch tokens (p1, p2, p3), the
        attention matrix is:

        t1-t1 t1-t2 t1-p1 t1-p2 t1-p3
        t2-t1 t2-t2 t2-p1 t2-p2 t2-p3

        So that the mask (True values are deleted) should be:

        False True False False False
        True False False False False
        """
        mask = torch.zeros(attn_shape, dtype=torch.bool)
        for i in range(nb_task_tokens):
            mask[:, i, :i] = True
            mask[:, i, i+1:nb_task_tokens] = True
        return mask

    def forward(self, x, attn_mask=False, nb_task_tokens=1, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:,:nb_task_tokens]).reshape(B, nb_task_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        if attn_mask:
            mask = self.get_attention_mask(attn.shape, nb_task_tokens)
            attn[mask] = -float('inf')
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, nb_task_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn, v

class ClassAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fc=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = fc(dim, dim, bias=qkv_bias)
        self.k = fc(dim, dim, bias=qkv_bias)
        self.v = fc(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = fc(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_heads=None, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if mask_heads is not None:
            mask_heads = mask_heads.expand(B, self.num_heads, -1, N)
            attn = attn * mask_heads

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn, v


