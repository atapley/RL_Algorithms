from registry import AgentRegistry
from bases import AgentBase
import torch as T
from torch.distributions import Categorical
import copy
import numpy as np
import torch.nn.functional as F
from math import exp


@AgentRegistry.register('DDQNAgent')
class DDQNAgent(AgentBase):
    def __init__(self, model, args):
        super(DDQNAgent, self).__init__()
        self.model1 = model
        self.model2 = copy.deepcopy(model)
        self.gamma = args.gamma
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.max_episodes = args.num_episodes
        self.device = args.device

    def get_action(self, e, obs):
        # Introduce randomness to encourage exploration in the early episodes
        randomness = self.epsilon[0] + ((self.epsilon[1] - self.epsilon[0]) * exp(-.008 * e))
        if np.random.random() > randomness:
            obs = T.from_numpy(obs).float()
            action_probs = self.model1(obs)
            action = T.argmax(action_probs).detach().item()
        else:
            # Randomly choose action to take
            action = self.model1.action_space.sample()

        return action

    def learn(self, batch):
        states, actions, rewards, states_, dones = self.convert_to_tensors(batch)
        self.model1.optimizer.zero_grad()

        # Get the actual chosen q values
        q1 = self.model1(states).gather(1, actions)

        # Get the predicted next states q values
        q1_ = self.model1(states_)
        q2_ = self.model2(states_)

        # Take the minimum from both models based off of Cliiped DDQN implementation
        q_ = T.min(T.max(q1_, -1).values, T.max(q2_, -1).values)

        # Calculate the discounted rewards based off of clipped q values
        target_q = rewards + (1 - dones) * self.gamma * q_

        loss = F.mse_loss(input=q1, target=target_q.detach())

        loss.backward()

        # Make sure the gradients are not too large
        T.nn.utils.clip_grad_norm_(self.model1.parameters(), 0.5)
        self.model1.optimizer.step()

        return loss

    # Update the target model to match the current model
    def update_target(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def convert_to_tensors(self, batch):
        states = []
        actions = []
        rewards = []
        states_ = []
        dones = []

        for t in batch:
            states.append(t[0])
            actions.append(t[1])
            rewards.append(t[2])
            states_.append(t[3])
            dones.append(t[4])

        return T.as_tensor(states, dtype=T.float).to(self.device), \
               T.unsqueeze(T.as_tensor(actions).to(self.device), 1), \
               T.unsqueeze(T.as_tensor(rewards).to(self.device), 1), \
               T.unsqueeze(T.as_tensor(states_, dtype=T.float).to(self.device), 1), \
               T.unsqueeze(T.as_tensor(dones, dtype=T.int).to(self.device), 1)


@AgentRegistry.register('A2CAgent')
class A2CAgent(AgentBase):
    def __init__(self, model, args):
        super(A2CAgent, self).__init__()
        self.model = model
        self.gamma = args.gamma
        self.beta = args.beta
        self.device = args.device

    # Sample the action from the actor's current policy
    def get_action(self, obs):
        obs = T.from_numpy(obs).float()
        action_probs, _ = self.model(obs)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        entropy = -(action_probs * log_probs).sum()

        return action.item(), log_probs, entropy

    def calculate_returns(self, rewards, dones, R):
        returns = [0]*(len(rewards) + 1)
        returns[-1] = R

        for i in reversed(range(len(rewards))):
            returns[i] = rewards[i] + self.gamma * returns[i + 1] * (1 - dones[i])

        return T.as_tensor(returns[:-1]).float()

    def learn(self, states, log_probs, returns, entropys):
        self.model.optimizer.zero_grad()

        _, values = self.model(states)
        values = T.squeeze(values)

        # Calculate an advantage (how good of a choice our action was)
        advantages = returns - values.detach()
        actor_loss = T.mean(((T.as_tensor(-1 * log_probs)) * advantages) + self.beta * entropys)
        critic_loss = T.mean(returns - values)
        loss = actor_loss + critic_loss
        loss.backward()

        # Make sure the gradients we use are not too large
        T.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.model.optimizer.step()

        return actor_loss, critic_loss

    def convert_to_tensors(self, transitions):
        self.states = T.empty(0).to(self.device)
        self.log_probs = T.empty(0).to(self.device)
        self.entropy = T.empty(0).to(self.device)
        self.rewards = T.empty(0).to(self.device)
        self.dones = T.empty(0).to(self.device)

        for t in transitions:
            self.states = T.cat((self.states, T.unsqueeze(T.from_numpy(t[0]), dim=0).float())).to(self.device)
            self.log_probs = T.cat((self.log_probs, T.unsqueeze(t[1],dim=0))).to(self.device)
            self.entropy = T.cat((self.entropy, T.unsqueeze(t[2],dim=0))).to(self.device)
            self.rewards = T.cat((self.rewards, T.unsqueeze(T.as_tensor(t[3]),dim=0))).to(self.device)
            self.dones = T.cat((self.dones, T.unsqueeze(T.as_tensor(t[4]),dim=0))).to(self.device)

        return self.states, \
               self.log_probs, \
               self.entropy, \
               self.rewards, \
               self.dones
