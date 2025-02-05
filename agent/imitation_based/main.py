from viper.core.rl import *
from viper.pong.pong import *
from viper.pong.dqn import *
from viper.core.dt import *
from viper.util.log import *


def learn_dt():
    # Parameters
    model_path = '../../data/model-atari-pong-1/saved'
    max_depth = 12
    n_test_rollouts = 50
    save_dirname = '../tmp/pong'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = True

    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)
    student = DTPolicy(max_depth)

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, get_pong_symbolic)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, get_pong_symbolic, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

