import argparse

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
    parser.add_argument("--priority_beta", type=float, default=0.4, help="the priority replay buffer factor")
    parser.add_argument("--priority_epsilon", type=float, default=1e-5,
                        help="Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--N-steps", type=int, default=1, help="number of N-steps")
    parser.add_argument("--start-training-len", type=int, default=5e4, help="number of start training steps")
    parser.add_argument("--buffer-size", type=int, default=1e6, help="size of replay buffer")
    parser.add_argument("--buffer-update-step", type=int, default=200, help="size of replay buffer")
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
    parser.add_argument("--save-gif", action="store_true", default=False)
    parser.add_argument("--evaluation", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--gif-dir", type=str, default="./gifs/",
                        help="directory where plot data is saved")
    return parser.parse_args()

