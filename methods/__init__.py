import importlib
from os.path import join, exists, dirname, basename
from methods.base import BaseLearner

def get_trainer(logger, config) -> BaseLearner:
    method_name = config.method.lower()

    if not exists(join('methods', config.method_type, method_name+'.py')):
        raise ValueError('Method Python File {} do not exist!'.format(
                join('methods', config.method_type, method_name+'.py')))
    
    model = None
    model_filename = 'methods.' + config.method_type + '.' + method_name
    modellib = importlib.import_module(model_filename)
    for cls_name, cls in modellib.__dict__.items():
        if cls_name.lower() == method_name:
            model = cls

    if model is None:
        raise ValueError('Method class {} do not exist!'.format(method_name))
    
    logger.info('Trainer {} created!'.format(method_name))
    return model(logger, config)