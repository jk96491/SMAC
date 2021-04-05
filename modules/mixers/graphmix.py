import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..GNNs.gnn import GNN

class GraphMixer(nn.Module):
    def __init__(self, args):
        super(GraphMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = args.obs_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed
        
        # mixing GNN
        combine_type = 'gin'
        self.mixing_GNN = GNN(num_input_features=1, hidden_layers=[self.embed_dim],
                              state_dim=self.state_dim, hypernet_embed=hypernet_embed,
                              weights_operation='abs',
                              combine_type=combine_type)
        
        # attention mechanism
        self.enc_obs = True
        obs_dim = self.rnn_hidden_dim
        if self.enc_obs:
            self.obs_enc_dim = 16
            self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, self.obs_enc_dim),
                                             nn.ReLU())
            self.obs_dim_effective = self.obs_enc_dim
        else:
            self.obs_encoder = nn.Sequential()
            self.obs_dim_effective = obs_dim
            
        self.W_attn_query = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)
        self.W_attn_key = nn.Linear(self.obs_dim_effective, self.obs_dim_effective, bias=False)

        # output bias
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
            
    def forward(self, agent_qs, states,
                agent_obs=None,
                team_rewards=None,
                hidden_states=None):
        
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents, 1)
        
        # find the agents which are alive
        alive_agents = 1. * (th.sum(agent_obs, dim=3) > 0).view(-1, self.n_agents)
        # create a mask for isolating nodes which are dead by taking the outer product of the above tensor with itself
        alive_agents_temp1 = alive_agents.unsqueeze(2)
        alive_agents_temp2 = alive_agents.unsqueeze(1)

        alive_agents_tensor = th.zeros_like(alive_agents_temp1, dtype=th.float32)
        alive_agents_tensor[alive_agents_temp1 == True] = 1

        alive_agents_tensor2 = th.zeros_like(alive_agents_temp2, dtype=th.float32)
        alive_agents_tensor2[alive_agents_temp2 == True] = 1

        alive_agents_mask = th.bmm(alive_agents_tensor, alive_agents_tensor2)

        # encode hidden states
        encoded_hidden_states = self.obs_encoder(hidden_states)
        encoded_hidden_states = encoded_hidden_states.contiguous().view(-1, self.n_agents, self.obs_dim_effective)

        # adjacency based on the attention mechanism
        attn_query = self.W_attn_query(encoded_hidden_states)
        attn_key = self.W_attn_key(encoded_hidden_states)
        attn = th.matmul(attn_query, th.transpose(attn_key, 1, 2)) / np.sqrt(self.obs_dim_effective)

        # make the attention with softmax very small for dead agents so they get zero attention
        attn = nn.Softmax(dim=2)(attn + (-1e10 * (1 - alive_agents_mask)))
        batch_adj = attn * alive_agents_mask # completely isolate the dead agents in the graph
            
        GNN_inputs = agent_qs
        local_reward_fractions, y = self.mixing_GNN(GNN_inputs, batch_adj, states, self.n_agents)
        
        # state-dependent bias
        v = self.V(states).view(-1, 1, 1)
        q_tot = (y + v).view(bs, -1, 1)
        
        # effective local rewards
        if team_rewards is None:
            local_rewards = None
        else:
            local_rewards = local_reward_fractions.view(bs, -1, self.n_agents) * team_rewards.repeat(1, 1, self.n_agents)
            
        return q_tot, local_rewards, alive_agents.view(bs, -1, self.n_agents)
        