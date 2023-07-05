import importlib
from os.path import join, exists
from datasets.idata import iData
from utils.logger import MyLogger

def get_idata(logger:MyLogger, dataset_name:str, img_size) -> iData:
    name = dataset_name.lower()

    if not exists(join('datasets', name+'.py')):
        raise ValueError('Dataset Python File {} do not exist!'.format(
                join('datasets', name+'.py')))
    
    dataset = None
    dataset_filename = 'datasets.' + name
    modellib = importlib.import_module(dataset_filename)
    for cls_name, cls in modellib.__dict__.items():
        if cls_name.lower() == name:
            dataset = cls

    if dataset is None:
        raise ValueError('Dataset object {} do not exist!'.format(name))
    
    logger.info('Dataset object {} is created!'.format(name))
    return dataset(img_size)