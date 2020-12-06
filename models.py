from torch import nn
from torch.nn import functional as F
from registry import ModelRegistry
from bases import ModelBase
from torch import optim


# Typical Linear (Dense) block for training. Size of hidden layers can be controlled within args.hidden
# or when called.
class LinearBlock(nn.Module):

    def __init__(self, state_space, hidden):
        super(LinearBlock, self).__init__()
        self.block = nn.ModuleList()

        assert len(hidden) > 0, 'Linear block needs at least one layer!'

        for e in range(len(hidden)):
            if e == 0:
                self.block.append(nn.Linear(state_space, hidden[e]))
            else:
                self.block.append(nn.Linear(hidden[e - 1], hidden[e]))


@ModelRegistry.register('DDQN')
class DDQN(ModelBase):
    def __init__(self, state_space, action_space, args):
        super(DDQN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = args.hidden
        self.hidden.append(self.action_space.n)
        self.device = args.device

        self.layers = LinearBlock(self.state_space.shape[0], self.hidden).block

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = self.layers[-1](x)

        return out


@ModelRegistry.register('ActorCritic')
class ActorCritic(ModelBase):
    def __init__(self, state_space, action_space, args):
        super(ActorCritic, self).__init__()
        self.state_space = state_space.shape[0]
        self.action_space = action_space.n
        self.ac_hidden = args.hidden
        self.device = args.device

        self.shared = LinearBlock(self.state_space, self.ac_hidden).block

        self.actor_out = nn.Linear(self.ac_hidden[-1], self.action_space)
        self.critic_out = nn.Linear(self.ac_hidden[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)

        for layer in self.shared:
            x = F.relu(layer(x))

        actor_out = F.softmax(self.actor_out(x), dim=-1)
        critic_out = self.critic_out(x)

        return actor_out, critic_out
