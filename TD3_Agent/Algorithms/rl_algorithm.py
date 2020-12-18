import sys
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/Algorithms')
sys.path.append('/home/wangyw/cq/code/manipulator/TD3_Agent/pytorch_core')
import abc
import numpy as np
import torch
from pytorch_core.pytorch_util import np_to_pytorch_batch
abs_path = '/'.join(str.split(__file__, '/')[:-2]) + '/checkpoints/policy_'


class RLAlgorithm(object):
    def __init__(self, sampler, num_total_epochs=1000, epoch_length=200,
                 num_train_repeat=1):
        self.sampler = sampler
        self._num_epochs, self._epoch_length = num_total_epochs, epoch_length
        self._n_train_repeat = num_train_repeat
        import os
        os.makedirs(abs_path + 'td3_', exist_ok=True)

    def _train(self, env, policy, pooling, load=None, save_param_name='td3_'):
        self.sampler.initialize(env, policy, pooling)

        for epoch in range(self._num_epochs):
            for t in range(self._epoch_length):
                self.sampler.sample()
                if not self.sampler.batch_ready():
                    continue

                for i in range(self._n_train_repeat):
                    self._do_training(t+epoch*self._epoch_length,
                                      np_to_pytorch_batch(self.sampler.random_batch()))

            if self.sampler.num_current_total_steps % 50 == 0:
                torch.save(policy.state_dict(), abs_path + save_param_name +
                           str(self.sampler.num_current_total_steps)+'.pt')
                self._do_evaluate()

    @abc.abstractmethod
    def _do_training(self, num_iter, batch_data):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_evaluate(self):
        raise NotImplementedError















