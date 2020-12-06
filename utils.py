import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AC Pytorch Implementation')

    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment to train with')

    parser.add_argument('--solved', type=list, default=[195, 100],
                        help='Average reward of X over Y episodes to be considered solved')

    parser.add_argument('--model', type=str, default='ActorCritic',
                        help='Model type')

    parser.add_argument('--agent', type=str, default='A2CAgent',
                        help='Agent type')

    parser.add_argument('--trainer', type=str, default='A2CTrainer',
                        help='Trainer type')

    parser.add_argument('--num_episodes', type=int, default=4000,
                        help='Number of episodes to run')

    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max number of steps to run within an episode')

    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')

    parser.add_argument('--hidden', type=list, default=[128],
                        help='Size of shared hidden layers in the model')

    parser.add_argument('--buffer_size', type=int, default=300,
                        help='Size of replay buffer')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of training batch')

    parser.add_argument('--lr', type=float, default=.001,
                        help='Learning rate')

    parser.add_argument('--gamma', type=float, default=.99,
                        help='Reward discount value')

    parser.add_argument('--beta', type=float, default=0.01,
                        help='Entropy value')

    parser.add_argument('--epsilon', type=list, default=[0.001, 0.6],
                        help='Randomness value [max, min]')

    parser.add_argument('--tensorboard', type=str, default='./runs/A2C_CartPole',
                        help='Where to save tensorboard files to')

    parser.add_argument('--render', type=bool, default=False,
                        help='Whether or not to render the env')

    args = parser.parse_args()

    return args
