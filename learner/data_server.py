import importmagic
from rpc_package.data_pb2_grpc import ResetServiceServicer
from rpc_package.data_pb2_grpc import SamplerInitParamServiceServicer
from rpc_package.data_pb2_grpc import Sampler2LearnerModelServiceServicer
from rpc_package.data_pb2_grpc import SamplerDataServiceServicer
from rpc_package.data_pb2_grpc import SamplerResultServiceServicer

from rpc_package.data_pb2 import CommonReply
from rpc_package.data_pb2 import ResetReply
from rpc_package.data_pb2 import SamplerInitParamReply
from rpc_package.data_pb2 import Sampler2LearnerModelReply

from logger_utility import logger


class ResetServer(ResetServiceServicer):
    """
    league resets initial parameters for both sampler workers and eval workers
    """
    def __init__(self, args, worker_id, lock):
        self.worker_id = worker_id
        self.lock = lock

    def get_reset_param(self, request, context):
        with self.lock:
            worker_id = self.worker_id.value
            self.worker_id.value += 1
        logger.info(f'ResetServer create worker {worker_id}')
        return ResetReply(worker_id=worker_id)

class SamplerInitParamServer(SamplerInitParamServiceServicer):
    """
    sampler get initial parameters from learner
    """
    def __init__(self, args, init_time_stamp, player_id, player_id_ready):
        self.args = args
        self.init_time_stamp = init_time_stamp
        self.player_id = player_id
        self.player_id_ready = player_id_ready

    def get_init_param(self, request, context):
        return SamplerInitParamReply(
            num_joints=self.args.num_joints,
            obs_dim=self.args.obs_dim,
            act_dim=self.args.act_dim,
            actor_hidden=self.args.actor_hidden,
            critic_hidden=self.args.critic_hidden,
            batch_size=self.args.batch_size,
            gamma=self.args.gamma,
            lammbda=self.args.lammbda,
            max_episode_steps=self.args.max_episode_steps
        )

class Sampler2LearnerModelServer(Sampler2LearnerModelServiceServicer):
    """
    sampler get model parameters from learner using gRPC
    """
    def __init__(self, model_buffer, time_stamp):
        self.model_buffer = model_buffer
        self.time_stamp = time_stamp

    def get_model(self, request, context):
        idx = request.actor_index
        rtime = request.time_stamp
        if rtime != self.time_stamp.value:
            modelstate = Sampler2LearnerModelReply(data=self.model_buffer[0],
                                                   time_stamp=self.time_stamp.value)
        else:
            modelstate = Sampler2LearnerModelReply(time_stamp=self.time_stamp.value)
        if idx == 0:
            logger.debug('Sampler2LearnerModelServer activated')
        return modelstate

class SamplerDataServer(SamplerDataServiceServicer):
    """
    sampler send collected data to learner
    """
    def __init__(self, pool):
        self.pool = pool

    def send_data(self, request, context):
        try:
            self.pool.put_nowait(request.compressed_data)
        except Exception:
            logger.info('data discarded')
            _ = self.pool.get()
        return CommonReply(signal=1)


class SamplerResultServer(SamplerResultServiceServicer):
    """
    sampler send results to learner
    """
    def __init__(self, results, rewards, result_lock, league):
        self.results = results
        self.rewards = rewards
        self.result_lock = result_lock
        self.league = league

    def send_result(self, request, context):
        player_id = request.red_index
        opponent_id = request.blue_index
        result_info = []
        for reward in request.rewards:
            if len(self.rewards[player_id]) >= 3000:
                self.rewards[player_id].pop(0)
            self.rewards[player_id].append(reward)
            result_info.append(reward)

        result = []
        for r in request.results:
            if len(self.results[player_id]) >= 3000:
                self.results[player_id].pop(0)
            self.results[player_id].append(r)
            result.append(r)
        if len(result):
            with self.result_lock:
                self.league.update_win_rate(player_id, opponent_id, result)
            logger.info(f'SamplerResultServer ({player_id}, {opponent_id}): {result}, {result_info}')
        return CommonReply(signal=2)