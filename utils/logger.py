import os
from datetime import datetime
import logging
import random
from tensorboardX import SummaryWriter
import wandb

from utils.toolkit import check_makedirs

class MyLogger:
    def __init__(self, config, file_log=True):
        self._logger_create_time = datetime.now().strftime('_%Y%m%d_%H%M%S')
        self._cur_seed = config.seed

        # initial logger
        gpid, pid = os.getpgid(os.getpid()), os.getpid()

        self._logger = logging.getLogger('GPID{}-PID{}'.format(gpid, pid))
        self._logger.setLevel(logging.INFO)
        
        # format = '%(asctime)s [GPID-{} PID-{}] [%(filename)s] => %(message)s'.format(gpid, pid)
        format = '%(asctime)s => %(message)s'
        formatter = logging.Formatter(format)
        
        # prepare console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        if file_log:
            # prepare file handler
            if config.checkpoint_dir is None:
                log_file_name = 'train_seed{}{}'.format(self._cur_seed, self._logger_create_time)
            elif len(config.checkpoint_names) < config.nb_tasks:
                log_file_name = 'resume_seed{}{}'.format(self._cur_seed, self._logger_create_time)
            elif len(config.checkpoint_names) == config.nb_tasks:
                log_file_name = 'test_seed{}{}'.format(self._cur_seed, self._logger_create_time)

            check_makedirs(config.logdir)
            file_handler = logging.FileHandler(filename=os.path.join(config.logdir, '{}.log'.format(log_file_name)), mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        self._logger.propagate = False
        self._logger.info('logger in GPID {} PID {} is created !'.format(gpid, pid))
        self._logger.info('Tmux session name: {}'.format(os.popen("tmux display-message -p '#S'").read().strip()))

        self._logger_type = None
        self._tblog = None

    def init_visual_log(self, config):
        # prepare visualize log
        self._logger_type = config.logger_type if config.logger_type is not None else ''
        if 'tensorboard' in self._logger_type and self._tblog == None:
            self._tblog = SummaryWriter(os.path.join(config.logdir, 'tb'))
            self.info('Applying tensorboard as visual log')
        if 'wandb' in self._logger_type:
            os.environ['WANDB_DIR']=config.logdir
            wandb.init(project=config.project, config=config.get_parameters_dict())
            # wandb.run.name = 'seed{}_{}_{}_{}_{}'.format(config.seed, config.method, config.dataset,
            #         config.backbone, config.note)
            self.info('Applying wandb as visual log')
        if self._logger_type == '':
            self.info('Applying nothing as visual log')
        elif 'tensorboard' not in self._logger_type and 'wandb' not in self._logger_type:
            raise ValueError('Unknown logger_type: {}'.format(self._logger_type))
    
    def info(self, msg):
        self._logger.info(msg=msg)
    
    def error(self, msg, **kwargs):
        self._logger.error(msg=msg, **kwargs)
    
    def debug(self, msg):
        self._logger.debug(msg=msg)
    
    def warning(self, msg):
        self._logger.warning(msg=msg)
    
    def visual_log(self, phase:str, msg_dict:dict, step:int):
        '''
        phase = 'train', 'valid', 'test'
        '''
        if 'wandb' in self._logger_type:
            wandb.log(msg_dict, step=step)
        if 'tensorboard' in self._logger_type:
            for key, value in msg_dict.items():
                self._tblog.add_scalar('seed{}_{}/{}'.format(self._cur_seed, phase, key), value, step)
    
    def release_handlers(self):
        if len(self._logger.handlers) > 0:
            for handler in list(self._logger.handlers):
                    self._logger.removeHandler(handler)
        if 'tensorboard' in self._logger_type:
            self._tblog.close()
        if 'wandb' in self._logger_type:
            pass