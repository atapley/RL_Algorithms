from torch.utils.tensorboard import SummaryWriter
from registry import TrainerRegistry
from bases import TrainerBase
from buffers import Buffer, ExpReplayBuffer
import torch as T


@TrainerRegistry.register('DDQNTrainer')
class DDQNTrainer(TrainerBase):
    def __init__(self, env, agent, args):
        super(DDQNTrainer, self).__init__()
        self.env = env
        self.agent = agent
        self.num_episodes = args.num_episodes
        self.replay_buffer = ExpReplayBuffer(args.buffer_size, args.batch_size, args.device)
        self.max_steps = args.max_steps

        assert len(args.solved) == 2, 'args.solved has to have length of exactly 2!'
        self.solved_r = args.solved[0]
        self.solved_ep = args.solved[1]
        self.render = args.render

        self.writer = SummaryWriter(args.tensorboard)

    def train(self):
        reward_history = []
        # For each update

        if self.render:
            self.env.render()

        for episode in range(self.num_episodes):
            episode_reward = 0
            episode_losses = []
            state = self.env.reset()

            for i in range(self.max_steps):
                action = self.agent.get_action(episode, state)

                state_, reward, done, _ = self.env.step(action)
                episode_reward += reward

                transition = [state, action, reward, state_, done]

                self.replay_buffer.insert_transition(transition)

                if self.replay_buffer.is_full():
                    states, actions, rewards,states_, dones = self.replay_buffer.sample_buffers()
                    loss, = self.agent.learn(states, actions, rewards, states_, dones)
                    episode_losses.append(loss)

                if episode % 100 == 0:
                    self.agent.update_target()

                if done or i == self.max_steps - 1:
                    print('Episode: ', episode, 'Reward: %i' % episode_reward)
                    reward_history.append(episode_reward)
                    self.writer.add_scalar("reward", episode_reward, episode)
                    self.writer.flush()

                    if len(reward_history) > self.solved_ep:
                        reward_history.pop(0)
                        if (sum(reward_history) / len(reward_history)) >= self.solved_r:
                            print('Env has been solved at episode ' + str(episode) + '!')
                            self.writer.close()
                            exit()
                    break
                else:
                    state = state_

            if len(episode_losses) > 0:
                self.writer.add_scalar("loss/loss", sum(episode_losses) / len(episode_losses), episode)
                self.writer.flush()

        self.writer.close()


@TrainerRegistry.register('A2CTrainer')
class A2CTrainer(TrainerBase):
    def __init__(self, env, agent, args):
        super(A2CTrainer, self).__init__()
        self.env = env
        self.agent = agent
        self.num_episodes = args.num_episodes
        self.buffer = Buffer(args.buffer_size, args.device)

        assert len(args.solved) == 2, 'args.solved has to have length of exactly 2!'
        self.solved_r = args.solved[0]
        self.solved_ep = args.solved[1]
        self.render = args.render

        self.writer = SummaryWriter(args.tensorboard)

    def train(self):
        reward_history = []
        # For each update

        if self.render:
            self.env.render()

        for episode in range(self.num_episodes):
            episode_reward = 0
            self.buffer.reset_buffers()
            state = self.env.reset()

            for i in range(self.buffer.buffer_size):
                action, log_probs, entropy = self.agent.get_action(state)

                state_, reward, done, _ = self.env.step(action)
                episode_reward += reward

                self.buffer.insert_transition(state, log_probs, entropy, reward, done)

                if done or i == self.buffer.buffer_size - 1:
                    print('Episode: ', episode, 'Reward: %i' % episode_reward)
                    reward_history.append(episode_reward)
                    self.writer.add_scalar("reward", episode_reward, episode)
                    self.writer.flush()

                    if len(reward_history) > self.solved_ep:
                        reward_history.pop(0)
                        if (sum(reward_history) / len(reward_history)) >= self.solved_r:
                            print('Env has been solved at episode ' + str(episode) + '!')
                            self.writer.close()
                            exit()
                    self.buffer.states = T.cat((self.buffer.states, T.unsqueeze(T.from_numpy(state), dim=0).float()))
                    break
                else:
                    state = state_

            if self.buffer.dones[-1]:
                R = T.as_tensor([0])
            else:
                _, R = self.agent.model(state)
                R = R.detach()

            states, log_probs, entropys, rewards, dones = self.buffer.get_buffers()
            returns = self.agent.calculate_returns(rewards, dones, R)
            actor_loss, critic_loss = self.agent.learn(states, log_probs, returns, entropys)

            self.writer.add_scalar("loss/actor", actor_loss, episode)
            self.writer.add_scalar("loss/critic", critic_loss, episode)
            self.writer.flush()

        self.writer.close()

