from abc import ABCMeta, abstractmethod
from torch import nn


class ModelBase(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(ModelBase, self).__init__()
        pass

    @abstractmethod
    def forward(self):
        pass

class TrainerBase(metaclass=ABCMeta):
    def __init__(self):
        super(TrainerBase, self).__init__()
        pass

    @abstractmethod
    def train(self):
        pass

class AgentBase(metaclass=ABCMeta):
    def __init__(self):
        super(AgentBase, self).__init__()
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def learn(self):
        pass
