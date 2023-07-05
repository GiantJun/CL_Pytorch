import argparse
# from asyncio.log import logger
import copy
import yaml
import datetime
import os
import importlib

def load_yaml(settings_path):
    args = {}
    with open(settings_path) as data_file:
        param = yaml.load(data_file, Loader=yaml.FullLoader)
    args.update(param['basic'])

    dataset = args['dataset']
    backbone = args['backbone']
    if 'options' in param:
        args.update(param['options'][dataset][backbone])
    if 'special' in param:
        args.update(param['special'])
    return args

class Config:

    def __init__(self, config_mode=None):
        parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
        parser.add_argument('--config', type=str, default=None, help='yaml file of settings.')
        parser.add_argument('--checkpoint_dir', type=str, default=None, help="checkpoint's directory ready to load")

        self.project = 'IL_Framework'

        self.init_overwrite_names = []
        self.load_overwrite_names = []

        parser.add_argument('--logger_type', type=str, default=None, help='visualize logger type, e.g. tensorboard, wandb')

        parser.add_argument('--device', type=str, default=None, help='GPU ids, e.g. 0 (for single gpu) or 0,1,2 (for multi gpus)')
        # parser.add_argument('--local_rank', type=int, default=None, help='node rank for distributed training')

        parser.add_argument('--seed', nargs='+', type=int, default=None, help='random seed for the programe, 0 (for single seed) or 0 1 2 (run in seed 0 1 2 respectively)')
        parser.add_argument('--num_workers', type=int, default=None, help='CPUs for dataloader')
        
        parser.add_argument('--dataset', type=str, default=None, help='dataset to be used')
        parser.add_argument('--use_valid', type=bool, default=None, help='use valid dataset or not (default False)')
        parser.add_argument('--split_dataset', type=bool, default=None, help='split dataset into several step or not')
        parser.add_argument('--img_size', type=int, default=None, help='if not config will follow origin size of dataset')
        parser.add_argument('--shuffle', type=str, default=None, help='shuffle class order')
        parser.add_argument('--split_for_valid', type=bool, default=None, help='whether to split training set to true training set and valid set') # 赋初值为 None 相当于 False
        
        parser.add_argument('--backbone', type=str, default=None, help='backbone to train')
        parser.add_argument('--MLP_projector', type=bool, default=None, help='use MLP projector behind backbone or not')
        
        parser.add_argument('--method', type=str, default=None, help='methods to apply')
        parser.add_argument('--method_type', type=str, default=None, help='methods to apply')

        parser.add_argument('--apply_nme', type=bool, default=None, help='whether apply nme to classify')

        parser.add_argument('--incre_type', type=str, default=None, help='Incremental type e.t. cil or til')
        parser.add_argument('--pretrained', type=bool, default=None, help='whether use pretrained network weights to initial the network')
        parser.add_argument('--pretrain_path', type=str, default=None, help='prtrained network weights path to load')
        parser.add_argument('--freeze_fe', type=bool, default=None, help='freeze the feature extractor')
        parser.add_argument('--save_models', type=bool, default=None, help='save trained models weights')
        parser.add_argument('--save_pred_record', type=bool, default=None, help='save model predict record')
        parser.add_argument('--eval_metric', type=str, default=None, help='evaluate metrics option')

        parser.add_argument('--init_cls', type=str, default=None, help='init class num for the training')
        parser.add_argument('--increment', type=str, default=None, help='increment class num for the training')

        parser.add_argument('--note', type=str, default=None, help='addition notes added behind log folder to identify each run')

        parser.add_argument('--openset_test', type=bool, default=None, help='apply openset test or not.')

        # training config
        parser.add_argument('--init_epochs', type=int, default=None, help='init training epochs')
        parser.add_argument('--init_lrate', type=float, default=None, help='init training learning rate')
        parser.add_argument('--init_weight_decay', type=float, default=None, help='init weight decay for training')
        parser.add_argument('--init_opt_mom', type=float, default=None, help='init momenton for sgd')

        parser.add_argument('--init_scheduler', type=str, default=None, help='init learning rate decay method')
        parser.add_argument('--init_milestones', type=int, default=None, help='init milestones for training')
        parser.add_argument('--init_lrate_decay', type=float, default=None, help='init training learning rate decay')
        
        parser.add_argument('--epochs', type=int, default=None, help='training epochs')
        parser.add_argument('--test_epoch', type=int, default=None, help='test data per i epochs')
        parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
        parser.add_argument('--replay_batch_size', type=int, default=None, help='batch size for replay')

        parser.add_argument('--opt_type', type=str, default=None, help='optimizer for training')
        parser.add_argument('--lrate', type=float, default=None, help='training learning rate')
        parser.add_argument('--weight_decay', type=float, default=None, help='weight decay for sgd')
        parser.add_argument('--opt_mom', type=float, default=None, help='momenton for sgd')

        parser.add_argument('--scheduler', type=str, default=None, help='learning rate decay method')
        parser.add_argument('--milestones', nargs='+', type=int, default=None, help='for multi step learning rate decay scheduler')
        parser.add_argument('--lrate_decay', type=float, default=None, help='for multi step learning rate decay scheduler')

        parser.add_argument('--criterion', type=str, default=None, help='loss function, e.g. ce, focal')

        # memory bank config
        parser.add_argument('--memory_size', type=int, default=None, help='memory size for memory buffer')
        parser.add_argument('--fixed_memory', type=bool, default=None, help='fix memory size for each class')
        parser.add_argument('--memory_per_class', type=bool, default=None, help='memory size per class for fixed memory')
        parser.add_argument('--sampling_method', type=str, default=None, help='sampler methods option for memory buffer')

        parser.add_argument('--mode', type=str, default=None, help='mode of appling methods')

        # load basic configs, we just use a part of command args
        args_part = parser.parse_known_args()[0]
        for name, value in vars(args_part).items():
            setattr(self, name, value)

        if args_part.config is not None:
            init_config = load_yaml(self.config) # obtain configs from yaml

            # get specific configs from (the rest of) command args
            modellib = importlib.import_module('methods.' + init_config['method_type'] + '.' + init_config['method'])
            if hasattr(modellib, 'add_special_args'):
                modellib.add_special_args(parser)
            
            for name, value in vars(parser.parse_args()).items():
                setattr(self, name, value)
            
            for name, value in init_config.items():
                if getattr(self, name) == None:
                    setattr(self, name, value)
                    self.init_overwrite_names.append(name)
            print('Loaded config file: {}'.format(self.config))
        else:
            parser.parse_args()
        
        self.gpu_num = len(self.device.split(','))
        
        self.checkpoint_names = []
        
        # init logdir
        if self.checkpoint_dir:
            logdir = self.checkpoint_dir
        else:
            nowTime = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
            if self.method_type == 'multi_steps':
                logdir = 'logs/multi_step/{}_{}_replay_{}/{}/{}_b{}i{}'.format(self.method, self.incre_type, 
                    self.sampling_method, self.dataset, self.backbone, 
                    self.init_cls, self.increment)
            else:
                logdir = 'logs/{}/{}/{}/{}'.format(self.method_type, self.method,
                    self.dataset, self.backbone)
            if config_mode is not None and 'seach_param' in config_mode:
                logdir = logdir + '_param_search'
            
            if self.note is not None:
                logdir = os.path.join(logdir, self.note)
                if os.path.exists(logdir):
                    logdir = logdir + nowTime
            else:
                logdir = os.path.join(logdir, nowTime)
        self.logdir = logdir
    
    def load_saved_config(self, init_dict: dict) -> None:
        for key, value in init_dict.items():
            if (not hasattr(self,key)) or getattr(self, key) == None:
                setattr(self, key, value)
                self.load_overwrite_names.append(key)

    def update(self, update_dict: dict) -> None:
        for key, value in update_dict.items():
            setattr(self, key, value)
    
    def get_parameters_dict(self):
        parameter_dict = copy.deepcopy(vars(self))
        for name, value in vars(self).items():
            if name in ['init_overwrite_names', 'load_overwrite_names'] or value == None:
                parameter_dict.pop(name)
        return parameter_dict
    
    def print_config(self, logger) -> None:
        logger.info(30*"=")
        logger.info("log hyperparameters in seed {}".format(self.seed))
        logger.info(30*"-")
        for name, value in sorted(vars(self).items()):
            if not name in ['init_overwrite_names', 'load_overwrite_names'] and not getattr(self, name) == None:
                logger.info('{}: {}'.format(name, value))
        logger.info(30*"=")
        logger.info('Inital configs overwrited by yaml config file: {}'.format(self.init_overwrite_names))
        logger.info('Inital configs overwrited by loaded pkl configs: {}'.format(self.load_overwrite_names))
        