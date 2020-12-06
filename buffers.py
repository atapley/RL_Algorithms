import torch as T
import random


class ExpReplayBuffer():
    def __init__(self, buffer_size, batch_size, device):
        super(ExpReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.buffer = []

    def is_full(self):
        return len(self.buffer) == self.buffer_size

    def insert_transition(self, transition):
        if self.is_full():
            self.buffer.pop(0)
            self.buffer.append(transition)
        else:
            self.buffer.append(transition)

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

    def sample_buffers(self):
        batch = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, states_, dones = self.convert_to_tensors(batch)

        return states, actions, rewards, states_, dones


class Buffer():
    def __init__(self, buffer_size, device):
        super(Buffer, self).__init__()
        self.buffer_size = buffer_size
        self.device = device

    def reset_buffers(self):
        self.states = T.empty(0).to(self.device)
        self.log_probs = T.empty(0).to(self.device)
        self.entropy = T.empty(0).to(self.device)
        self.rewards = T.empty(0).to(self.device)
        self.dones = T.empty(0).to(self.device)

    def insert_transition(self, state, log_probs, entropy, reward, done):
        self.states = T.cat((self.states, T.unsqueeze(T.from_numpy(state),dim=0).float())).to(self.device)
        self.log_probs = T.cat((self.log_probs, T.unsqueeze(log_probs,dim=0))).to(self.device)
        self.entropy = T.cat((self.entropy, T.unsqueeze(entropy,dim=0))).to(self.device)
        self.rewards = T.cat((self.rewards, T.unsqueeze(T.as_tensor(reward),dim=0))).to(self.device)
        self.dones = T.cat((self.dones, T.unsqueeze(T.as_tensor(done),dim=0))).to(self.device)

    def get_buffers(self):
        return self.states,\
               self.log_probs,\
               self.entropy,\
               self.rewards,\
               self.dones

