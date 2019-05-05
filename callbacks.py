import math
import tensorflow as tf
# import tensorflow.keras.backend as K
import numpy as np

def step_decay(epoch):
    initial_lrate = 1.0
    drop = 0.998
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)) * (1.0 - momentum_schedule(epoch))
    return lrate

def momentum_schedule(epoch):
    initial_momentum = 0.5
    final_momentum   = 0.99
    epoch_max_value  = 20.0
    if epoch<epoch_max_value:
        return ((final_momentum - initial_momentum)/epoch_max_value) * epoch + initial_momentum
    else:
        return final_momentum

class MomentumScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer,'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute')
        momentum = float(tf.keras.backend.get_value(self.model.optimizer.momentum))
        try:
            momentum = self.schedule(epoch,momentum)
        except TypeError:
            momentum = self.schedule(epoch)
        if not isinstance(momentum, (float,np.float32,np.float64)):
            raise ValueError('The output of the "schedule" function should be float')
        tf.keras.backend.set_value(self.model.optimizer.momentum,momentum)
        if self.verbose > 0:
            print('\nEpoch {}: MomentumScheduler setting momentum to {}'.format(epoch+1,momentum))
    
    def on_epoch_end(self,epoch,logs=None):
        logs = logs or {}
        logs['momentum'] = tf.keras.backend.get_value(self.model.optimizer.momentum)