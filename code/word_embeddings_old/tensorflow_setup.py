'''
Conatins functions for tensorflow setting environment 
such as dynamic memory usage
'''
import tensorflow as tf 

def enable_dynamic_gpu_memory():
    '''
    Enables dynamic memory allocation when using GPU with tensorflow
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print('Enabled dynamic gpu memory')
            

def tensorflow_shutup():
    """
    Mutes many of the tensorflow warnings. Don't run tensorflow_shutup if you want warnings.

    Source:
    https://stackoverflow.com/a/54950981
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass
    print('Ran tensorflow_shutup')
    

def init(shutup=False):
    '''
    Run selected proceedures when initializing
    '''
    enable_dynamic_gpu_memory()
    if shutup: tensorflow_shutup()
    