from torch.utils.tensorboard import SummaryWriter
from registry import TrainerRegistry
from bases import TrainerBase
from buffers import Buffer
import torch as T


@TrainerRegistry.register('DDQNTrainer')
class DDQNTrainer(TrainerBase):
    def __init__(self, env, agent, args):
        super(DDQNTrainer, self).__init__()
        self.env = env
        self.agent = agent
        self.num_episodes = args.num_episodes
        self.buffer = Buffer(args.buffer_size, args.batch_size)
        self.max_steps = args.max_steps

        assert len(args.solved) == 2, 'args.solved has to have length of exactly 2!'
        self.solved_r = args.solved[0]
        self.solved_ep = args.solved[1]
        self.render = args.render

        self.writer = SummaryWriter(args.tensorboard)

    def train(self):
        reward_history = []
        self.buffer.reset()
        # For each update

        if self.render:
            self.env.render()

        # For each episode
        for episode in range(self.num_episodes):
            episode_reward = 0
            episode_losses = []
            state = self.env.reset()

            # Until max_steps is reached
            for i in range(self.max_steps):

                # Get action
                action = self.agent.get_action(episode, state)

                # Take step based on action
                state_, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store transition within the buffer for batched learning
                transition = [state, action, reward, state_, done]

                self.buffer.insert_transition(transition)

                # If the buffer is full, start learning using a random sample of transitions from the buffer
                if self.buffer.is_full():
                    batch = self.buffer.sample_buffer()
                    loss = self.agent.learn(batch)
                    episode_losses.append(loss)

                # Update the target network every 100 episodes
                if episode % 100 == 0:
                    self.agent.update_target()

                # If the episode has finished (max_steps reached or from env)
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
        self.buffer = Buffer(args.buffer_size, args.batch_size)

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

        # For each episode
        for episode in range(self.num_episodes):
            episode_reward = 0
            self.buffer.reset()
            state = self.env.reset()

            # While there is room in the buffer
            for i in range(self.buffer.buffer_size):
                # Get action
                action, log_probs, entropy = self.agent.get_action(state)

                # Take step
                state_, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store transition in buffer for TD learning
                transition = [state, log_probs, entropy, reward, done]
                self.buffer.insert_transition(transition)

                # If the episode is finished (max_steps reached or from env)
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
                    break
                else:
                    state = state_

            # Get the estimated next step. If episode ended, then next value is 0, otherwise get from critic
            if self.buffer.buffer[-1][-1]:
                R = T.as_tensor([0])
            else:
                _, R = self.agent.model(state)
                R = R.detach()

            # Get transitions from buffer to train with
            transitions = self.buffer.get_buffer()
            states, log_probs, entropys, rewards, dones = self.agent.convert_to_tensors(transitions)

            # Calculate the discounted rewards
            returns = self.agent.calculate_returns(rewards, dones, R)
            actor_loss, critic_loss = self.agent.learn(states, log_probs, returns, entropys)

            self.writer.add_scalar("loss/actor", actor_loss, episode)
            self.writer.add_scalar("loss/critic", critic_loss, episode)
            self.writer.flush()

        self.writer.close()

