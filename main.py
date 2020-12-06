import os
import gym
from utils import parse_arguments
from registry import ModelRegistry, TrainerRegistry, AgentRegistry
import models, agents, trainers

def main():
    args = parse_arguments()
    print('Command Line Args:', args)

    # If tensorboard path already exists, remove dir
    if os.path.exists(args.tensorboard):
        [os.remove(os.path.join(args.tensorboard, f)) for f in os.listdir(args.tensorboard)]
        os.rmdir(args.tensorboard)

    env = gym.make(args.env)
    model = ModelRegistry.get(args.model)(env.observation_space, env.action_space, args)
    agent = AgentRegistry.get(args.agent)(model, args)
    trainer = TrainerRegistry.get(args.trainer)(env, agent, args)
    trainer.train()

    env.close()

if __name__ == "__main__":
    main()