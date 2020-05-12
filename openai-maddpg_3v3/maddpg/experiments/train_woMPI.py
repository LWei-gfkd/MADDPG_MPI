import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy
import csv
from PIL import Image
import multiprocessing as mul
import threading

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from maddpg.trainer.replay_buffer import ReplayBuffer

slim = tf.contrib.slim
replay_buffer = ReplayBuffer(1e6)
# queue = mul.Queue(5)
lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="competition_3v3", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--start_training_len", type=int, default=5e4, help="number of start training steps")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--multiprocess", action="store_true", default=False, help="training with multiprocess")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="testv2", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./results/test_3v3",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save_gif", action="store_true", default=False)
    parser.add_argument("--evaluation", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def display_frames_as_gif(frames, file):
    frames[0].save(file, save_all=True, loop=True, append_images=frames[1:], duration=100)


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
    return env


def get_agents(env, num_adversaries, obs_shape_n, arglist, name="agent"):
    agents = []
    model = mlp_model
    agent = MADDPGAgentTrainer
    for i in range(num_adversaries):
        agents.append(agent(
            "%s_%d" % (name, i), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    ####green nodes take ddpg
    for i in range(num_adversaries, env.n):
        agents.append(agent(
            "%s_%d" % (name, i), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    return agents


def adversary_leave_screen(env):
    adversary_leave_num = 0
    for agent in env.agents:
        if agent.adversary:
            for p in range(env.world.dim_p):
                x = abs(agent.state.p_pos[p])
                if x > 1.0:
                    adversary_leave_num = adversary_leave_num + 1
                    break
    if adversary_leave_num >= env.num_adversaries:
        return True
    else:
        return False


def adversary_all_die(env):
    allDie = True
    for agent in env.agents:
        if agent.adversary:
            if not agent.death:
                allDie = False
                break
    return allDie


def green_leave_screen(env):
    green_leave_num = 0
    for agent in env.agents:
        if not agent.adversary:
            for p in range(env.world.dim_p):
                x = abs(agent.state.p_pos[p])
                if x > 1.0:
                    green_leave_num = green_leave_num + 1
                    break
    if green_leave_num >= env.n - env.num_adversaries:
        return True
    else:
        return False


def evaluation(arglist):
    print("Evaluation start!")
    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agents networks
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        ####changed by yuan li
        num_adversaries = copy.deepcopy(env.num_adversaries)
        arglist.num_adversaries = copy.deepcopy(num_adversaries)

        learners = get_agents(env, num_adversaries, obs_shape_n, arglist)
        test_num = 100
        obs_n = env.reset()
        episode_step = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        frames = []
        red_win, red_leave, green_win, green_leave = 0, 0, 0, 0

        # load model
        if arglist.load_dir == "":
            print("No checkpoint loaded! Evaluation ended!")
            return -1
        else:
            print('Loading model...')
            U.load_state(arglist.load_dir)

        for train_step in range(test_num):
            while True:
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(learners, obs_n)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                if arglist.display:
                    env.render()
                    time.sleep(0.01)
                if arglist.save_gif:
                    array = env.render(mode='rgb_array')[0]
                    im = Image.fromarray(array)
                    frames.append(im)
                # changed by liyuan
                done = any(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                if green_leave_screen(env) or adversary_all_die(env) or adversary_leave_screen(env):
                    terminal = True

                if adversary_all_die(env):
                    green_win += 1
                if green_leave_screen(env):
                    green_leave += 1
                if adversary_leave_screen(env):
                    red_leave += 1

                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                ###liyuan: compute the arverage win rate
                if done:
                    red_win = red_win + 1

                if done or terminal:
                    if arglist.save_gif:
                        display_frames_as_gif(frames, "./gif/episode_{}.gif".format(train_step))
                        frames = []
                    print("Episode {}, red win:{}, green win {}, red all leave {}, green all leave {}, steps: {}".
                          format(train_step, red_win, green_win, red_leave, green_leave, episode_step))
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    break
        print("evaluation: ", red_win, green_win, red_leave, green_leave)
        return red_win


def initialize_variables(env):
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    return episode_rewards, agent_rewards, final_ep_rewards, final_ep_ag_rewards, agent_info

def copy_variables(src_agents, dst_agents, sess):
    for i, dst_agent in enumerate(dst_agents):
        dict = src_agents[i].get_variable_dict(sess)
        dst_agent.load_weights(dict, sess)

def train(arglist, PID=None, lock=None):
    # global replay_buffer
    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agents networks
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        ####changed by yuan li
        num_adversaries = copy.deepcopy(env.num_adversaries)
        arglist.num_adversaries = copy.deepcopy(num_adversaries)

        learners = get_agents(env, num_adversaries, obs_shape_n, arglist)
        var_list_n = []
        for learner in learners:
            var_list_n.extend(learner.get_variable_list())
        actors = get_agents(env, num_adversaries, obs_shape_n, arglist, "actor")
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()
        saver = tf.train.Saver(var_list=var_list_n, max_to_keep=20)
        # Load previous results, if necessary
        if arglist.load_dir != "":
            U.load_state(arglist.load_dir, saver)

        copy_variables(learners, actors, sess)
        ####initiallize all variables
        episode_rewards, agent_rewards, final_ep_rewards, final_ep_ag_rewards, agent_info = initialize_variables(env)
        obs_n = env.reset()
        episode_step, train_step = 0, 0
        t_start = time.time()
        print('Starting iterations...')

        invalid_train, red_win, red_leave, green_win, green_leave = 0, 0, 0, 0, 0

        while True:
            loop_start_time = time.time()

            if len(episode_rewards) > arglist.num_episodes:
                break
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(actors, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            # changed by liyuan
            done = any(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            ###liyuan: compute the arverage win rate
            if green_leave_screen(env) or adversary_all_die(env) or adversary_leave_screen(env):
                terminal = True

            if adversary_all_die(env):
                green_win += 1
            if green_leave_screen(env):
                invalid_train += 1
                green_leave += 1
            if adversary_leave_screen(env):
                red_leave += 1

            if episode_step >= arglist.max_episode_len:
                for i, agent in enumerate(env.agents):
                    if agent.adversary:
                        rew_n[i] -= 50

            if adversary_all_die(env):
                for i, agent in enumerate(env.agents):
                    if agent.adversary:
                        rew_n[i] -= 100

            if done:
                red_win = red_win + 1
                for i, agent in enumerate(env.agents):
                    if agent.adversary:
                        rew_n[i] += 200
                        rew_n[i] += (arglist.max_episode_len - episode_step) / arglist.max_episode_len

            # collect experience
            replay_buffer.add(obs_n, action_n, rew_n, new_obs_n, done_n)
            obs_n = new_obs_n
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1
            # training
            if train_step % 100 == 0 and len(replay_buffer) >= arglist.batch_size * arglist.max_episode_len:
                # only update every 100 steps and the length of replay_buffer is large enough
                replay_sample_index = replay_buffer.make_index(arglist.batch_size)
                obs_n_a, act_n_a, rew_n_a, obs_next_n_a, done_n_a = replay_buffer.sample_index(replay_sample_index)
                sample = (obs_n_a, act_n_a, rew_n_a, obs_next_n_a, done_n_a)
                for i, agent in enumerate(learners):
                    agent.update(learners, sample)
                copy_variables(learners, actors, sess)

            # save model, display training output
            if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
                if red_win >= 0.8 * arglist.save_rate:
                    temp_dir = arglist.save_dir + "_" + str(len(episode_rewards)) + "_" + str(red_win) + "_{}".format(
                        PID)
                    U.save_state(temp_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print(
                        "PID {}, steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            PID, train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                            round(time.time() - t_start, 3)))
                    print("red win: {}, green win: {}, red all leave: {}, green all leave: {}".format(
                        red_win, green_win, red_leave, green_leave))
                    mydata = []
                    mydata.append(str(len(episode_rewards)))
                    mydata.append(str(np.mean(episode_rewards[-arglist.save_rate:])))
                    mydata.append(str(np.mean(agent_rewards[0][-arglist.save_rate:])))
                    mydata.append(str(np.mean(agent_rewards[1][-arglist.save_rate:])))
                    mydata.append(str(np.mean(agent_rewards[2][-arglist.save_rate:])))
                    mydata.append(str(red_win))
                    mydata.append(str(round(time.time() - t_start, 3)))
                    out = open('1mydata_{}.csv'.format(PID), 'a', newline='')
                    csv_write = csv.writer(out, dialect='excel')
                    csv_write.writerow(mydata)

                if len(episode_rewards) > 3000:
                    U.save_state(arglist.save_dir, saver=saver)

                invalid_train, red_win, red_leave, green_win, green_leave = 0, 0, 0, 0, 0
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            loop_end_time = time.time()
            print("loop:{},  time:{}".format(train_step, (loop_end_time - loop_start_time)))


def main():
    arglist = parse_args()
    arglist.scenario = 'competition_3v3'
    #arglist.save_dir = 'E:\openai-maddpg_3v3\maddpg\experiments/test'
    #arglist.save_dir = '/home/nudt-pdl/lw/openai-maddpg_3v3/maddpg/experiments/test'
    arglist.load_dir = './test'
    if arglist.evaluation:
        evaluation(arglist)
        return
    train(arglist)


if __name__ == '__main__':
    main()
