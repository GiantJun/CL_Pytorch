from utils.config import Config
from utils.logger import MyLogger
import copy
import os
import torch
import yaml
from utils.data_manager import DataManager
import methods
import wandb
from multiprocessing import Process

os.environ['WANDB_API_KEY']='faf0e04a242bbf756c5f2894f620681e0a7dc358'
# os.environ['WANDB_MODE']='offline'

class SweepWorker(Process):
    def __init__(self, threadID, sweepID, device, exp_count):
        super(SweepWorker, self).__init__()
        self._threadID = threadID
        self._sweepID = sweepID
        self._device = device
        self._exp_count = exp_count
    
    def run(self):
        global global_logger
        global_logger.info('Thread {} Started using GPU-{}! '.format(self._threadID, self._device))
        os.environ['CUDA_VISIBLE_DEVICES'] = self._device
        wandb.agent(self._sweepID, function=single_run, count=self._exp_count)
        global_logger.info('Thread {} Finished with GPU-{} released'.format(self._threadID, self._device))


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def single_run():
    try:
        global global_config
        temp_config = copy.deepcopy(global_config)
        temp_logger = MyLogger(temp_config)
        set_random(temp_config.seed)
        with wandb.init():
            wandb_config = wandb.config
            temp_config.update(wandb_config)
            data_manager = DataManager(temp_logger, temp_config.dataset, temp_config.img_size, temp_config.split_dataset,
                    temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment)
            temp_config.update({'total_class_num':data_manager.total_classes, 'nb_tasks':data_manager.nb_tasks,
                        'increment_steps':data_manager.increment_steps})
            temp_config.print_config(temp_logger)

            temp_logger.init_visual_log(temp_config)
            
            trainer = methods.get_trainer(temp_logger, temp_config)

            while trainer.cur_taskID < data_manager.nb_tasks - 1:
                trainer.prepare_task_data(data_manager)
                trainer.prepare_model()
                trainer.incremental_train()
                trainer.eval_task()
                trainer.after_task()
    except Exception as e:
        temp_logger.error(e, exc_info=True, stack_info=True)

if __name__ == '__main__':
    with open('./search_config.yaml') as data_file:
        param_search_config = yaml.load(data_file, Loader=yaml.FullLoader)
    # init program configs
    is_parallel = param_search_config.pop('is_parallel', False)
    process_per_GPU = param_search_config.pop('process_per_GPU', 0)
    all_device = param_search_config.pop('all_device', '0')
    exp_count = param_search_config.pop('exp_count', 1)

    global_config = Config(config_mode='seach_param')
    global_config.seed = global_config.seed[0]
    global_config.logger_type = 'wandb'
    os.environ['WANDB_DIR']=global_config.logdir
    global_logger = MyLogger(global_config) # for main process

    sweep_id = wandb.sweep(param_search_config, project='IL_Framework_search')
    if is_parallel:
        try:            
            process_list = []
            process_id = 0
            exp_count_per_process = exp_count // (len(all_device.split(',')) * process_per_GPU)
            for device_id in all_device.split(','):
                for i in range(process_per_GPU):
                    process_list.append(SweepWorker(process_id, sweep_id, device_id, exp_count_per_process))
                    process_id += 1
                
            # 把子线程设置为守护线程,主线程结束则子线程结束
            # time.sleep(random.randint(0,10))
            for process in process_list:
                process.start()
            
            for process in process_list:
                process.join()

            global_logger.info('main process finished !')
        except KeyboardInterrupt:
            for process in process_list:
                process.terminate()
            global_logger.error('program stoped with keyborad interrupt !')
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=all_device
        wandb.agent(sweep_id, single_run, count=exp_count)
