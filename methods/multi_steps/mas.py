import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.multi_steps.finetune_il import Finetune_IL
from backbone.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

class MAS(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._network = IncrementalNet(config.backbone, config.pretrained)
        self._lamda = config.lamda
        self._alpha = config.alpha
        self._reg_params = {}
        self._reg_params['lambda'] = self._lamda
        if self._incre_type != 'til':
            raise ValueError('EWC is a task incremental method!')

    def _epoch_train(self, model, train_loader, optimizer, scheduler):
        losses = 0.
        losses_clf, losses_ewc = 0., 0.
        correct, total = 0, 0
        model.train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            logits, feature_outputs = model.forward_til(inputs, self._cur_task)
            
            loss_clf = F.cross_entropy(logits, targets - self._known_classes)
            losses_clf += loss_clf.item()
            if self._cur_task == 0:
                loss = loss_clf
            else:
                loss_ewc=self.compute_ewc()
                losses_ewc += loss_ewc.item()
                loss = loss_clf + self._lamda*loss_ewc

            preds = torch.max(logits, dim=1)[1] + self._known_classes
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_clf', losses_clf/len(train_loader), 'Loss_ewc', losses_ewc/len(train_loader)]
        return model, train_acc, train_loss

    def after_task(self):
        self.update_weights_params(self._network, norm='L2')
        super().after_task()

    # 更新重要程度的几种方式
    def update_weights_params(self, model_ft, norm='L2'):
        """update the importance weights based on the samples included in the reg_set. Assume starting from zero omega
        model_ft: the model trained on the previous task 
        """
        if self._cur_task == 0:
            #inialize the importance params,omega, to zero
            self.initialize_reg_params(model_ft)
        else:
            #store previous task param
            self.initialize_store_reg_params(model_ft)
        #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
        optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
        
        #compute the imporance params
        self.compute_importance(model_ft, optimizer_ft, norm)
        
        if self._cur_task > 0:
            self.accumulate_reg_params(model_ft)
        
        self._reg_params['lambda'] = self._lamda
        self.sanitycheck(model_ft)
    
    def initialize_reg_params(self, model):
        """initialize an omega for each parameter to zero"""
        reg_params={}
        for name, param in model.named_parameters():
            if param.requires_grad: # 参数可训练
                print('initializing param',name)
                omega=torch.FloatTensor(param.size()).zero_() # 
                init_val=param.data.clone()
                reg_param={}
                reg_param['omega'] = omega
                #initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[name]=reg_param
        self._reg_params = reg_params
    
    def initialize_store_reg_params(self, model):
        """set omega to zero but after storing its value in a temp omega in which later we can accumolate them both"""
        for name, param in model.named_parameters():
            #in case there some layers that are not trained
            if param.requires_grad:
                if name in self._reg_params:
                    reg_param=self._reg_params.get(name)
                    print('storing previous omega',name)
                    prev_omega=reg_param.get('omega')
                    new_omega=torch.FloatTensor(param.size()).zero_()
                    init_val=param.data.clone()
                    reg_param['prev_omega']=prev_omega   
                    reg_param['omega'] = new_omega
                    
                    #initialize the initial value to that before starting training
                    reg_param['init_val'] = init_val
                    self._reg_params[name]=reg_param
            else:
                if name in self._reg_params: 
                    reg_param=self._reg_params.get(name)
                    print('removing unused omega',name)
                    del reg_param['omega'] 
                    del self._reg_params[name]

    def accumulate_reg_params(self, model):
        """accumelate the newly computed omega with the previously stroed one from the old previous tasks"""
        reg_params=self._reg_params
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in reg_params:
                    reg_param=reg_params.get(name)
                    print('restoring previous omega',name)
                    prev_omega=reg_param.get('prev_omega')
                    prev_omega=prev_omega.cuda()
                    
                    new_omega=(reg_param.get('omega')).cuda()
                    acc_omega=torch.add(prev_omega,new_omega)
                    
                    del reg_param['prev_omega']
                    reg_param['omega'] = acc_omega
                
                    reg_params[name]=reg_param
                    del acc_omega
                    del new_omega
                    del prev_omega
            else:
                if name in reg_params: 
                    reg_param=reg_params.get(name)
                    print('removing unused omega',name)
                    del reg_param['omega'] 
                    del reg_params[name]             
        return reg_params

    def compute_importance(self, model, optimizer, norm='L2'):
        """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
        Uses the L2norm of the function output. This is what we MAS uses as default
        """
        model.eval()  # Set model to training mode so we get the gradient
        print('*'*10+' Computing MAS {} weight importance '.format(norm)+'*'*10)
        #nessecary index to keep the running average
        for index, (_, inputs, labels) in enumerate(self._train_loader):
            # get the inputs
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, feature_outputs = model.forward_til(inputs, self._cur_task)
        
            #compute the L2 norm of output 
            Target_zeros=torch.zeros(outputs.size())
            Target_zeros=Target_zeros.to(self._device)

            #note no avereging is happening here
            if norm == 'L2':
                loss = torch.nn.MSELoss(size_average=False)
            elif norm == 'L1':
                loss = torch.nn.L1Loss(size_average=False)
            else:
                self._logger.warning('Unsupport norm {} in updating weights importance!'.format(norm))
                exit(0)
            targets = loss(outputs,Target_zeros) 
            #compute the gradients
            targets.backward()

            #update the parameters importance
            optimizer.step(self._reg_params, self.param2name, index, labels.size(0))
            # print('batch number ',index)
        print('Done')
    
    def sanitycheck(self, model):
        self._logger.info('*'*10+'checking omega'+'*'*10)
        for name, param in model.named_parameters():
            if name in self._reg_params:
                self._logger.info('Checking '+name)
                reg_param=self._reg_params.get(name)
                omega=reg_param.get('omega')
                
                # print('omega max is',omega.max())
                # print('omega min is',omega.min())
                # print('omega mean is',omega.mean())

    def _get_optimizer(self, params, config, is_init:bool):
        optimizer = None
        if is_init:
            if config.opt_type == 'sgd':
                optimizer = Weight_Regularized_SGD(params, momentum=0.9, lr=config.init_lrate, weight_decay=config.init_weight_decay)
            elif config.opt_type == 'adam':
                optimizer = Weight_Regularized_Adam(params, lr=config.init_lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        else:
            if config.opt_type == 'sgd':
                optimizer = Weight_Regularized_SGD(params, momentum=0.9, lr=config.lrate, weight_decay=config.weight_decay)
            elif config.opt_type == 'adam':
                optimizer = Weight_Regularized_Adam(params, lr=config.lrate)
            else: 
                raise ValueError('No optimazer: {}'.format(config.opt_type))
        return optimizer

class Weight_Regularized_Adam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(Weight_Regularized_Adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(Weight_Regularized_Adam, self).__setstate__(state)
    
    def step(self, reg_params, param2name, closure=None):
        loss = None
        if closure is None:
            loss = closure
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            amsgrad=group['amsgrad']
            lr=group['lr']
            weight_decay=group['weight_decay']
            eps=group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                ### MAS part ###
                p_name = param2name[p]
                if p_name in reg_params:
                    reg_param=reg_params.get(p_name)
                    #get omega for this parameter
                    omega=reg_param.get('omega')
                    #initial value when the training start
                    init_val=reg_param.get('init_val')
                    
                    curr_wegiht_val=p.data
                    #move the tensors to cuda
                    init_val=init_val.cuda()
                    omega=omega.cuda()
                    
                    #get the difference
                    weight_dif=curr_wegiht_val.add(-1,init_val)
                    #compute the MAS penalty
                    regulizer=weight_dif.mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    d_p.add_(regulizer)
                    del regulizer
                ### MAS part End ###

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['step'] += 1
                
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)
                
                exp_avg.mul_(beta1).add_(d_p, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1-beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

class Weight_Regularized_SGD(optim.SGD): # 实现网络更新的同时，对重要权重特殊处理，进行带惩罚的更新
    r"""Implements SGD training with importance params regulization. IT inherents stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, orth_reg=False, L1_decay=False):
        
        super(Weight_Regularized_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        self.orth_reg = orth_reg
        self.L1_decay = L1_decay

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)
       
        
    def step(self, reg_params, param2name, closure=None):
        """Performs a single optimization step.
        Arguments:
            reg_params: omega of all the params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        reg_lambda=reg_params.get('lambda')
       
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
           
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
               
                #MAS PART CODE GOES HERE
                #if this param has an omega to use for regulization
                p_name = param2name[p]
                if p_name in reg_params:
                    reg_param=reg_params.get(p_name)
                    #get omega for this parameter
                    omega=reg_param.get('omega')
                    #initial value when the training start
                    init_val=reg_param.get('init_val')
                    
                    curr_wegiht_val=p.data
                    #move the tensors to cuda
                    init_val=init_val.cuda()
                    omega=omega.cuda()
                    
                    #get the difference
                    weight_dif=curr_wegiht_val.add(-1,init_val)
                    #compute the MAS penalty
                    regulizer=weight_dif.mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    d_p.add_(regulizer)
                    del regulizer
                #MAS PARAT CODE ENDS
                
                if weight_decay != 0:
                    if self.L1_decay:
                        d_p.add_(weight_decay, p.data.sign())
                    else:
                        d_p.add_(weight_decay, p.data)
                 
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                p.data.add_(-group['lr'], d_p)
                
        return loss#ELASTIC SGD

class MAS_Omega_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(MAS_Omega_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_update, self).__setstate__(state)
       
    def step(self, reg_params, param2name, batch_index,batch_size,closure=None):
        """
        Performs a single parameters importance update setp
        """
        #print('************************DOING A STEP************************')
        loss = None
        if closure is not None:
            loss = closure()
             
        for group in self.param_groups:
            #if the parameter has an omega to be updated
            for p in group['params']:
                #print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
            
                p_name = param2name[p]
                if p_name in reg_params:
                    d_p = p.grad.data
                  
                    #HERE MAS IMPOERANCE UPDATE GOES
                    #get the gradient
                    unreg_dp = p.grad.data.clone()
                    reg_param=reg_params.get(p_name)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    #get parameter omega
                    omega=reg_param.get('omega')
                    omega=omega.cuda()
                        
                    #sum up the magnitude of the gradient
                    prev_size=batch_index*batch_size
                    curr_size=(batch_index+1)*batch_size
                    omega=omega.mul(prev_size)
                    
                    omega=omega.add(unreg_dp.abs_())
                    #update omega value
                    omega=omega.div(curr_size)
                    if omega.equal(zero.cuda()):
                        print('omega after zero')

                    reg_param['omega']=omega
                   
                    reg_params[p_name]=reg_param
                    #HERE MAS IMPOERANCE UPDATE ENDS
        return loss#HAS NOTHING TO DO
