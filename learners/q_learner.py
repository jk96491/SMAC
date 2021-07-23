import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.graphmix import GraphMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "graphmix":
                self.mixer = GraphMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 관련된 데이터를 가져온다.
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Agent 개별의 Q값을 산출함
        mac_out = []
        hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        mac_out = th.stack(mac_out, dim=1)  # 시간순에 따라 Concat
        hidden_states = th.stack(hidden_states, dim=1)

        # Agent가 선택한 행동의 Q값을 뽑는다.
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Agent Target Network 의 개별 Q값을 산출함
        target_mac_out = []
        target_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        # target network는 next_state 기준 이므로 t=1부터 저장
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = th.stack(target_hidden_states[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)

            # Agent의 Q값 저장
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999

            # Agent의 next_state 기준의 최대 Q값의 Action 저장
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

            # 이 action 에 대한 Target Network Q값을 저장한다.
            # 실제 target_mac_out의 최댓값과 다를 수 있음
            # 실제로 Agent가 선택한 행동의 대한 Q값을 가져오는 셈
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.args.mixer == 'graphmix':

            # Mix
            chosen_action_qvals_peragent = chosen_action_qvals.clone()
            target_max_qvals_peragent = target_max_qvals.detach()

            Q_total, local_rewards, alive_agents_mask = self.mixer(chosen_action_qvals,
                                                                               batch["state"][:, :-1],
                                                                               agent_obs=batch["obs"][:, :-1],
                                                                               team_rewards=rewards,
                                                                               hidden_states=hidden_states[:, :-1]
                                                                               )

            target_Q_total = self.target_mixer(target_max_qvals,
                                                 batch["state"][:, 1:],
                                                 agent_obs=batch["obs"][:, 1:],
                                                 hidden_states=target_hidden_states
                                                 )[0]

            ## Global loss
            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_Q_total

            # Td-error
            td_error = (Q_total - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            global_loss = (masked_td_error ** 2).sum() / mask.sum()

            ## Local losses
            # Calculate 1-step Q-Learning targets
            local_targets = local_rewards + self.args.gamma * (1 - terminated).repeat(1, 1, self.args.n_agents) \
                            * target_max_qvals_peragent

            # Td-error
            local_td_error = (chosen_action_qvals_peragent - local_targets)
            local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask.float()

            # 0-out the targets that came from padded data
            local_masked_td_error = local_td_error * local_mask

            # Normal L2 loss, take mean over actual data
            local_loss = (local_masked_td_error ** 2).sum() / mask.sum()

            # total loss
            lambda_local = self.args.lambda_local
            loss = global_loss + lambda_local * local_loss

        else:
            # Mix
            if self.mixer is not None:
                # Agent가 선택한 Q값과 state 정보를 넘겨 Q_total, target_Q_total 산출
                Q_total = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_Q_total = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
            else:
                Q_total = chosen_action_qvals
                target_Q_total = target_max_qvals

            # 1 step Q 러닝 수행
            targets = rewards + self.args.gamma * (1 - terminated) * target_Q_total

            # Td-error
            td_error = (Q_total - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)

            if self.args.mixer == 'graphmix':
                self.logger.log_stat("global_loss", global_loss.item(), t_env)
                self.logger.log_stat("local_loss", local_loss.item(), t_env)

            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (Q_total * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))