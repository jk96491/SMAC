import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NoiseRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NoiseRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.noise_embedding_dim)
        self.noise_fc2 = nn.Linear(args.noise_embedding_dim, args.noise_embedding_dim)
        self.noise_fc3 = nn.Linear(args.noise_embedding_dim, args.n_actions)

        self.hyper = True
        self.hyper_noise_fc1 = nn.Linear(args.noise_dim + args.n_agents, args.rnn_hidden_dim * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, noise):
        agent_ids = th.eye(self.args.n_agents, device=inputs.device).repeat(noise.shape[0], 1)
        noise_repeated = noise.repeat(1, self.args.n_agents).reshape(agent_ids.shape[0], -1)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        noise_input = th.cat([noise_repeated, agent_ids], dim=-1)

        if self.hyper:
            W = self.hyper_noise_fc1(noise_input).reshape(-1, self.args.n_actions, self.args.rnn_hidden_dim)
            wq = th.bmm(W, h.unsqueeze(2))
        else:
            z = F.tanh(self.noise_fc1(noise_input))
            z = F.tanh(self.noise_fc2(z))
            wz = self.noise_fc3(z)

            wq = q * wz

        return wq, h
