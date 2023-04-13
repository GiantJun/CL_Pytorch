import torch
import torch.nn as nn
import copy

from backbone.inc_net import get_backbone

class L2PNet(nn.Module):

    def __init__(self, logger, backbone_type, prompt_length=5, embedding_key='cls', prompt_init='uniform',
            prompt_pool=True, prompt_key=True, pool_size=10, top_k=5, batchwise_prompt=True, prompt_key_init='uniform',
            head_type='prompt', use_prompt_mask=False, global_pool='token', class_token=True):
        super().__init__()
        assert 'vit' in backbone_type, 'l2p only support vit !'
        assert global_pool in ('', 'avg', 'token')

        self._logger = logger

        self.feature_extractor = get_backbone(logger, backbone_type, True)
        self.feature_extractor.pre_logits = nn.Identity()
        self.feature_extractor.head = nn.Identity()
        self.head = None

        self.feature_dim = self.feature_extractor.num_features
        self._head_type = head_type
        self._use_prompt_mask = use_prompt_mask

        self._embed_dim = self.feature_extractor.embed_dim
        self._class_token = class_token
        self._global_pool = global_pool

        self._prompt = Prompt(length=prompt_length, embed_dim=self._embed_dim, embedding_key=embedding_key, 
                    prompt_init=prompt_init, prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size,
                    top_k=top_k, batchwise_prompt=batchwise_prompt, prompt_key_init=prompt_key_init)

        self.num_prefix_tokens = 1 if class_token else 0 # class_token len
        self.num_prefix_tokens += prompt_length * top_k # the num of added prompt prefix token

        embed_len = self.feature_extractor.patch_embed.num_patches + self.num_prefix_tokens
        self.prompt_pos_embed = nn.Parameter(torch.randn(1, embed_len, self._embed_dim) * .02, requires_grad=True)
    
    def update_fc(self, nb_classes):
        head = nn.Linear(self.feature_dim, nb_classes)
        if self.head is not None:
            nb_output = self.head.out_features
            weight = copy.deepcopy(self.head.weight.data)
            bias = copy.deepcopy(self.head.bias.data)
            head.weight.data[:nb_output] = weight
            head.bias.data[:nb_output] = bias
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.head
        self.head = head

    def extract_origin_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x, task_id=-1, cls_features=None):
        x = self.feature_extractor.patch_embed(x)
        if self._use_prompt_mask:
            start = task_id * self._prompt.top_k
            end = (task_id + 1) * self._prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(x.device)
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
            if end > self._prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        res = self._prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
        self.total_prompt_len = res['total_prompt_len']
        x = res['prompted_embedding']

        if self.feature_extractor.cls_token is not None:
            x = torch.cat((self.feature_extractor.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.feature_extractor.pos_drop(x + self.prompt_pos_embed)

        x = self.feature_extractor.blocks(x)
        
        x = self.feature_extractor.norm(x)
        if self._class_token and self._head_type == 'token':
            x = x[:, 0]
        elif self._head_type == 'gap' and self._global_pool == 'avg':
            x = x.mean(dim=1)
        elif self._head_type == 'prompt' and self._prompt.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self._class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self._head_type == 'token+prompt' and self._prompt.prompt_pool and self._class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier head!')
        
        res['pre_logits'] = x
        return self.head(x), res

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()
        '''
        length (int): tokens in a single prompt
        embed_dim (int): embedding dimension, should be the same as token dimension in vit
        embeding_key (str): option for embeding key, after whitch img key will be computed similarity with prompt keys.
                            (option for L2P, e.t. 'mean', 'max', 'mean_max', 'cls')
        prompt_init (str): init option for prompt (e.t. 'uniform', 'zero')
        prompt_pool (bool): use prompt pool or not (option for L2P)
        prompt_key (bool): if using learnable prompt keys to query prompt or not (option for L2P)
        pool_size (int): the amount of prompt that pool can store
        '''
        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(pool_size, length, embed_dim))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(pool_size, length, embed_dim))
                nn.init.uniform_(self.prompt)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls' and cls_features is not None:
                x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way '{}' of calculating embedding keys!".format(self.embedding_key))

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt: # use the same topk prompts in the batch
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            
            batched_prompt_raw = torch.gather(self.prompt.expand(idx.shape[0], -1, -1, -1), 
                                                index=idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.length, x_embed.shape[-1]), dim=1) # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = torch.gather(prompt_norm.unsqueeze(0).expand(batch_size, -1, -1), 
                                            index=idx.unsqueeze(-1).expand(-1, -1, c), dim=1) # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out