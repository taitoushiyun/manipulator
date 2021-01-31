import abc
import numpy as np
import torch
import SAC_Agent.pytorch_core.pytorch_util as ptu
abs_path = '/'.join(str.split(__file__, '/')[:-2]) + '/checkpoint_parameters/policy_'
abs_path1 = 'policy_'


class RLAlgorithm(object):
    def __init__(self, sampler,
                 num_total_epochs=1000,
                 epoch_length=200,
                 num_train_repeat=1):

        self.sampler = sampler
        self._num_epochs = num_total_epochs
        self._epoch_length = epoch_length
        self._n_train_repeat = num_train_repeat

    def _train(self, env, policy, pool, load=None, save_param_name='twin_'):
        self.sampler.initialize(env, policy, pool)
        for epoch in range(self._num_epochs):

            print('--------------------------- Epoch %i ---------------------------' % (epoch + 1))
            for t in range(self._epoch_length):
                self.sampler.sample()
                if not self.sampler.batch_ready():
                    continue

                for i in range(self._n_train_repeat):
                    self._do_training(num_iter=t + epoch * self._epoch_length,
                                      batch_data=np_to_pytorch_batch(self.sampler.random_batch()))

            if epoch % 5 == 0 and epoch >= 400:
                torch.save(policy.state_dict(), abs_path1 + save_param_name + str(epoch) + '.pt')
                self._do_evaluate()

    @abc.abstractmethod
    def _do_training(self, num_iter, batch_data):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_evaluate(self):
        raise NotImplementedError


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v






