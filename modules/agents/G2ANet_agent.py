import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class G2ANet(nn.Module):
    def __init__(self, input_shape, args):
        super(G2ANet, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.attention_dim = 32
        self.rnn_hidden_dim = 64

        # Encoding
        self.encoding = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.h = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # Hard
        self.hard_bi_GRU = nn.GRU(self.rnn_hidden_dim * 2, self.rnn_hidden_dim, bidirectional=True)
        self.hard_encoding = nn.Linear(self.rnn_hidden_dim * 2, 2)

        # Soft
        self.q = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.rnn_hidden_dim, self.attention_dim)

        # Decoding
        self.decoding = nn.Linear(self.rnn_hidden_dim + self.attention_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros((1, self.n_agents, self.rnn_hidden_dim))

    def forward(self, obs, hidden_state):
        size = obs.shape[0]

        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        h_out = self.h(obs_encoding, h_in)

        if self.args.hard:
            h = h_out.reshape(-1, self.args.n_agents,
                              self.args.rnn_hidden_dim)
            input_hard = []

            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
                h_hard_i = []
                for j in range(self.args.n_agents):
                    if j != i:
                        h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))

                h_hard_i = torch.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)

            input_hard = torch.stack(input_hard, dim=-2)

            input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)

            h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))

            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1,batch_size * n_agents,rnn_hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
            h_hard = h_hard.reshape(-1,
                                    self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
            # print(hard_weights)
            hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))

        # Soft Attention
        q = self.q(h_out).reshape(-1, self.args.n_agents,
                                  self.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_agents,
                                  self.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents,
                                          self.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        x = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]

            k_i = torch.stack(k_i, dim=0)
            k_i = k_i.permute(1, 2, 0)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            score = torch.matmul(q_i, k_i)

            scaled_score = score / np.sqrt( self.attention_dim)

            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)

            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        x = torch.stack(x, dim=1).reshape(-1,  self.attention_dim)  # (batch_size * n_agents, args.attention_dim)
        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        return output, h_out