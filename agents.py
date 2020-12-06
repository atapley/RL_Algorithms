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

    def get_action(self, e, obs):

        randomness = self.epsilon[0] + ((self.epsilon[1] - self.epsilon[0]) * exp(-.008 * e))
        if np.random.random() > randomness:
            obs = T.from_numpy(obs).float()
            action_probs = self.model1(obs)
            action = T.argmax(action_probs).detach().item()
        else:
            action = self.model1.action_space.sample()

        return action

    def learn(self, states, actions, rewards, states_,  dones):
        self.model1.optimizer.zero_grad()

        q1 = self.model1(states).gather(1, actions)

        q1_ = self.model1(states_)
        q2_ = self.model2(states_)

        q_ = T.min(T.max(q1_, -1).values, T.max(q2_, -1).values)

        target_q = rewards + (1 - dones) * self.gamma * q_

        loss = F.mse_loss(input=q1, target=target_q.detach())

        loss.backward()

        T.nn.utils.clip_grad_norm_(self.model1.parameters(), 0.5)
        self.model1.optimizer.step()

        return loss

    def update_target(self):
        self.model2.load_state_dict(self.model1.state_dict())


@AgentRegistry.register('A2CAgent')
class A2CAgent(AgentBase):
    def __init__(self, model, args):
        super(A2CAgent, self).__init__()
        self.model = model
        self.gamma = args.gamma
        self.beta = args.beta

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

        _, values = self.model(states[:-1])
        values = T.squeeze(values)

        advantages = returns - values.detach()
        actor_loss = T.mean(((T.as_tensor(-1 * log_probs)) * advantages) + self.beta * entropys)
        critic_loss = T.mean(returns - values)
        loss = actor_loss + critic_loss
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.model.optimizer.step()

        return actor_loss, critic_loss

