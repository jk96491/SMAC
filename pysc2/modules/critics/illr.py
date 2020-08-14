import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LIIRCritic(nn.Module):
    def __init__(self, scheme, args):
        super(LIIRCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_r_in = nn.Linear(128, self.n_actions)
        self.fc3_v_mix = nn.Linear(128, 1)

        self.fc4 = nn.Linear(128 * self.n_agents, 1)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)  # (bs, eplen, nagts, fea_len)
        x_1 = F.relu(self.fc1(inputs))  # (bs, eplen, nagts, 128)
        x_1 = F.relu(self.fc2(x_1))  # (bs, eplen, nagts, 128)
        v_mix = self.fc3_v_mix(x_1)  # (bs, eplen, nagts)

        max_t = batch.max_seq_length if t is None else 1

        x1 = x_1.reshape(self.args.batch_size, max_t, -1)  # (bs, eplen, nagts*128)
        v_ex = self.fc4(x1)  # (bs, eplen, 1)
        x = (self.fc3_r_in(x_1))  # (bs, eplen, nagts, nactions)
        x1 = x / 10.
        x2 = F.tanh(x1)
        r_in = x2 * 10.

        return r_in, v_mix, v_ex

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        #         last actions
        if t == 0:
            inputs.append(
                th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(
                batch["actions_onehot"][:, slice(t - 1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                                  dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        only needs state shape
        """
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape
