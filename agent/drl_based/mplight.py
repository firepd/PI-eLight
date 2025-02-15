from agent import FRAP


class MPLight(FRAP):
    """
    Chen, Chacha, et al. "Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.

    Note: the implementation of MPLight is based on FRAP, and its differences with FRAP are the choice of input
        features and reward function, which are defined in ``config.json'' and implemented in ``env/TSC_Env.py''.
        Besides, MPLight shares parameters among all intersections.
    """
    def __init__(self, config, env, idx):
        super(MPLight, self).__init__(config, env, idx)

        if idx > 0:
            # assert self.phase_2_passable_lanelink.tolist() == env.n_agent[0].phase_2_passable_lanelink.tolist(), \
            #     'MPLight shares parameters among intersections, therefore the topological structure of ' \
            #     'all intersections should be the same'
            share_layer(self.network_local, env.n_agent[0].network_local)
            share_layer(self.network_target, env.n_agent[0].network_target)
            self.network_optim = env.n_agent[0].network_optim


def share_layer(target, source):  # 让它在异质的区域也能work
    target.phase_embedding = source.phase_embedding
    target.num_vehicle_embedding = source.num_vehicle_embedding
    target.lanelink_embedding = source.lanelink_embedding
    target.relation_embedding = source.relation_embedding
    target.conv_cube = source.conv_cube
    target.conv_relation = source.conv_relation
    target.tail_layer = source.tail_layer