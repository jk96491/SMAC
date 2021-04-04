import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.GAT.models import GAT


class QMixer_gat(nn.Module):
    def __init__(self, args):
        super(QMixer_gat, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.node_count = self.n_agents * 2
        self.feature_dim = int(self.state_dim / self.node_count)
        self.gat_output = self.feature_dim * 2
        self.layer_input = self.state_dim * 2
        self.action_input_size = self.n_actions * self.n_agents

        self.gat_layer = GAT(self.feature_dim, 4, self.gat_output, 0.6, 0.2, 4)
        self.fc = nn.Sequential(nn.Linear(self.layer_input + self.action_input_size, 128),
                                nn.ReLU())
        self.fc_state = nn.Linear(128, self.state_dim)
        self.fc_reward = nn.Linear(128, 1)


        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.layer_input, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.layer_input, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.layer_input, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.layer_input, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.layer_input, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.layer_input, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        cur_gat_states = []
        for i in range(states.size(1)):
            cur_gat_state = self.gat_forward(states[:, i])
            cur_gat_states.append(cur_gat_state)

        states = th.stack(cur_gat_states, dim=1)

        bs = agent_qs.size(0)
        states = states.reshape(-1, self.layer_input)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def gat_forward(self, state):
        batch_size = state.shape[0]
        gat_states = []
        adj = th.ones( self.n_agents * 2, self.n_agents * 2).to(self.args.device)

        for i in range(batch_size):
            cur_state = state[i].view(self.n_agents * 2, self.feature_dim)
            gat_state = self.gat_layer(cur_state, adj)
            gat_states.append(gat_state.view(-1))

        gat_states = th.stack(gat_states, dim=0)
        return gat_states

    def train_gat(self, state, action):
        gat_states = self.gat_forward(state)
        action = action.view(self.args.batch_size, self.action_input_size)
        concat_input = th.cat([gat_states, action], dim=1)
        x = self.fc(concat_input)

        expected_state = self.fc_state(x)
        expected_reward = self.fc_reward(x)

        return expected_state, expected_reward


    def get_params(self):
        return list(self.hyper_w_1.parameters()) + list(self.hyper_w_final.parameters()) + list(self.hyper_b_1.parameters()) + list(self.V.parameters())

    def get_gat_paams(self):
        return list(self.gat_layer.parameters()) + list(self.fc.parameters()) + list(self.fc_state.parameters()) + list(self.fc_reward.parameters())
