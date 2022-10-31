from utils.config import Config
from utils.logger import MyLogger
import copy
import os
import torch
from utils.data_manager import DataManager
import methods


os.environ['WANDB_API_KEY']='faf0e04a242bbf756c5f2894f620681e0a7dc358'
os.environ['WANDB_MODE']='offline'

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    seed_list = copy.deepcopy(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES']=config.device
    # logger = set_logger(config)
    logger = MyLogger(config)

    try:
        for seed in seed_list:
            temp_config = copy.deepcopy(config)
            temp_config.seed = seed
            set_random(seed)
            data_manager = DataManager(logger, temp_config.dataset, temp_config.img_size, temp_config.split_dataset,
                    temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment)
            temp_config.update({'total_class_num':data_manager.total_classes, 'nb_tasks':data_manager.nb_tasks,
                        'increment_steps':data_manager.increment_steps, 'img_size':data_manager.img_size})
            temp_config.print_config(logger)

            logger.init_visual_log(temp_config)

            trainer = methods.get_trainer(logger, temp_config)

            while trainer.cur_taskID < data_manager.nb_tasks - 1:
                trainer.prepare_task_data(data_manager)
                trainer.prepare_model()
                trainer.incremental_train()
                trainer.eval_task()
                trainer.after_task()

    except Exception as e:
        logger.error(e, exc_info=True, stack_info=True)
