from env import TSCEnv
from utilities.utils import set_seed, get_config, set_thread, set_logger, release_logger, make_dir, copy_model_params
from utilities.UniTSA_runner import run_an_episode
import numpy as np
import multiprocessing
import argparse
from agent.universal_light.PPO import PPOAgent


parser = argparse.ArgumentParser(description='Training UniTSA')
parser.add_argument('--dataset', type=str, default='Hangzhou2', help='[Hangzhou1, Hangzhou2, Hangzhou3, Manhattan, Atlanta, Jinan, LosAngeles]')
parser.add_argument('--generalization', type=bool, default=False)
parser.add_argument('--target', type=str, default='Atlanta', help='[Hangzhou1, Hangzhou2, Hangzhou3, Manhattan, Jinan,]')
args = parser.parse_args()
DEBUG = False
data_name = args.dataset
eval_generalization = args.generalization
transfer_target = args.target
move_num = {'Atlanta': 16, 'Hangzhou1': 8, 'Hangzhou2': 8, 'Manhattan': 12, 'Jinan': 12, 'LosAngeles': 15}
max_move_num = move_num[data_name] if not eval_generalization else max(move_num[data_name], move_num[transfer_target])
cur_agent = 'UniTSA'


def run_an_experiment(inter_name, flow_idx, seed):
    dic_num_step = {'Atlanta': 900, 'Hangzhou1': 3600, 'Hangzhou2': 3600, 'Hangzhou3': 3600, 'Jinan': 3600, 'LosAngeles': 1800}
    config = get_config()
    config.update({
        'inter_name': inter_name,
        'seed': seed,
        'flow_idx': flow_idx,
        'save_result': not DEBUG,
        'dir': 'data/{}/'.format(inter_name),
        'flowFile': 'flow_{}.json'.format(flow_idx),
        'cur_agent': cur_agent,
        'render': False,
        'num_step': dic_num_step[inter_name] if inter_name in dic_num_step.keys() else 3600,
    })
    set_seed(config['seed'])
    set_thread()
    set_logger(config)

    env = TSCEnv(config)
    env.n_agent = []
    for idx in range(env.n):
        agent = PPOAgent(config, env, idx, max_move_num)
        env.n_agent.append(agent)

    train_episode = 64 if data_name in ['Jinan', 'Manhattan'] else 128
    for config['current_episode_idx'] in range(1, train_episode + 1):  # 开始训练， config['current_episode_idx']从0到64
        info = run_an_episode(env, config, True, max_move_num)
        if config['current_episode_idx'] % 10 != 0:
            continue
        config['logger'].info(
            '[{} On Training] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
                cur_agent, inter_name, flow_idx, config['current_episode_idx'],
                info['world_2_average_travel_time'][0],
                info['world_2_average_queue_length'][0],
                info['world_2_average_delay'][0],
                info['world_2_average_throughput'][0],
            ))

    ################################
    # Evaluation
    ################################
    if not eval_generalization:
        info = run_an_episode(env, config, False, max_move_num)
        config['logger'].info('[{} On Evaluation] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
            config['cur_agent'], inter_name, flow_idx, config['current_episode_idx'],
            info['world_2_average_travel_time'][0],
            info['world_2_average_queue_length'][0],
            info['world_2_average_delay'][0],
            info['world_2_average_throughput'][0],
        ))

    else:
        config.update({
            'inter_name': transfer_target,
            'flow_idx': flow_idx,  # 0~10
            'save_result': False,
            'dir': 'data/{}/'.format(transfer_target),
            'flowFile': 'flow_{}.json'.format(flow_idx),
            'num_step': dic_num_step[transfer_target] if transfer_target in dic_num_step.keys() else 3600,
        })

        trained_agent = env.n_agent[0]  # 迁移通常是杭州迁到其他地区, 杭州只有一个信号灯
        target_env = TSCEnv(config)
        target_env.n_agent = []
        for idx in range(target_env.n):
            agent = PPOAgent(config, target_env, idx, max_move_num)
            agent.actor = trained_agent.actor
            target_env.n_agent.append(agent)

        info = run_an_episode(target_env, config, False, max_move_num)
        config['logger'].info('[{} On Generalization] Inter: {}; Flow: {}; Episode: {}; ATT: {:.2f}; AQL: {:.2f}; AD: {:.2f}; Throughput: {:.2f}'.format(
            cur_agent, transfer_target, flow_idx, config['current_episode_idx'],
            info['world_2_average_travel_time'][0],
            info['world_2_average_queue_length'][0],
            info['world_2_average_delay'][0],
            info['world_2_average_throughput'][0],
        ))

    release_logger(config)
    return info['world_2_average_travel_time'][0], \
           info['world_2_average_queue_length'][0], \
           info['world_2_average_delay'][0], \
           info['world_2_average_throughput'][0]


if __name__ == '__main__':
    if not DEBUG:
        make_dir('log/{}/{}/'.format(data_name, cur_agent))
    parallel = True
    total_run = 10

    metrics = {
        'travel_time': [None for _ in range(total_run)],
        'queue_length': [None for _ in range(total_run)],
        'delay': [None for _ in range(total_run)],
        'throughput': [None for _ in range(total_run)]
    }
    seed_list = [992832, 284765, 905873, 776383, 198876, 192223, 223341, 182228, 885746, 992817]

    if parallel:
        with multiprocessing.Pool(processes=10) as pool:
            n_return_value = pool.starmap(run_an_experiment, [(data_name, f_idx, seed_list[f_idx]) for f_idx in range(10)])
            for f_idx, return_value in enumerate(n_return_value):
                metrics['travel_time'][f_idx] = return_value[0]
                metrics['queue_length'][f_idx] = return_value[1]
                metrics['delay'][f_idx] = return_value[2]
                metrics['throughput'][f_idx] = return_value[3]
    else:
        for f_idx in range(0, total_run):
            return_value = run_an_experiment(inter_name=data_name, flow_idx=f_idx, seed=seed_list[f_idx])
            metrics['travel_time'][f_idx] = return_value[0]
            metrics['queue_length'][f_idx] = return_value[1]
            metrics['delay'][f_idx] = return_value[2]
            metrics['throughput'][f_idx] = return_value[3]

    print('att: {:.2f}±{:.2f}; aql: {:.2f}±{:.2f}; ad: {:.2f}±{:.2f}; ath: {:.2f}±{:.2f}'.format(
        np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
        np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
        np.mean(metrics['delay']), np.std(metrics['delay']),
        np.mean(metrics['throughput']), np.std(metrics['throughput'])
    ))

    if not DEBUG:
        with open('log/{}/{}/summary.txt'.format(data_name, cur_agent), 'a') as fout:
            fout.write('tt: {:.2f}±{:.2f}; length: {:.2f}±{:.2f}; delay: {:.2f}±{:.2f}; through: {:.2f}±{:.2f}'.format(
                np.mean(metrics['travel_time']), np.std(metrics['travel_time']),
                np.mean(metrics['queue_length']), np.std(metrics['queue_length']),
                np.mean(metrics['delay']), np.std(metrics['delay']),
                np.mean(metrics['throughput']), np.std(metrics['throughput'])
            ))
