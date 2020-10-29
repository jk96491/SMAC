import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.illr import LIIRCritic
from utils.rl_utils import build_td_lambda_targets_v2
import torch as th
from torch.optim import RMSprop

vf_coef = 1.0


class LIIRLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = LIIRCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.policy_new = copy.deepcopy(self.mac)
        self.policy_old = copy.deepcopy(self.mac)

        if self.args.use_cuda:
            # following two lines should be used when use GPU
            self.policy_old.agent = self.policy_old.agent.to("cuda")
            self.policy_new.agent = self.policy_new.agent.to("cuda")
        else:
            # following lines should be used when use CPU,
            self.policy_old.agent = self.policy_old.agent.to("cpu")
            self.policy_new.agent = self.policy_new.agent.to("cpu")

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.fc1.parameters()) + list(self.critic.fc2.parameters()) + list(
            self.critic.fc3_v_mix.parameters())
        self.intrinsic_params = list(self.critic.fc3_r_in.parameters()) + list(self.critic.fc4.parameters())  # to do
        self.params = self.agent_params + self.critic_params + self.intrinsic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)
        self.intrinsic_optimiser = RMSprop(params=self.intrinsic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)  # should distinguish them
        self.update = 0
        self.count = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask_long = mask.repeat(1, 1, self.n_agents).view(-1, 1)
        mask = mask.view(-1, 1)

        avail_actions1 = avail_actions.reshape(-1, self.n_agents, self.n_actions)  # [maskxx,:]
        mask_alive = 1.0 - avail_actions1[:, :, 0]
        mask_alive = mask_alive.float()

        q_vals, critic_train_stats, target_mix, target_ex, v_ex, r_in = self._train_critic(batch, rewards, terminated,
                                                                                           actions, avail_actions,
                                                                                           critic_mask, bs, max_t)

        actions = actions[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        q_vals = q_vals.reshape(-1, 1)
        pi = mac_out.view(-1, self.n_actions)
        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask_long.squeeze(-1) == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (target_mix.reshape(-1, 1) - q_vals).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_pi_taken = log_pi_taken.reshape(-1, self.n_agents)
        log_pi_taken = log_pi_taken * mask_alive
        log_pi_taken = log_pi_taken.reshape(-1, 1)

        liir_loss = - ((advantages * log_pi_taken) * mask_long).sum() / mask_long.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        liir_loss.backward()
        grad_norm_policy = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # _________Intrinsic loss optimizer --------------------
        # ____value loss
        v_ex_loss = (((v_ex - target_ex.detach()) ** 2).view(-1, 1) * mask).sum() / mask.sum()

        # _____pg1____
        mac_out_old = []
        self.policy_old.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs_tmp = self.policy_old.forward(batch, t=t, test_mode=True)
            mac_out_old.append(agent_outs_tmp)
        mac_out_old = th.stack(mac_out_old, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out_old[avail_actions == 0] = 0
        mac_out_old = mac_out_old / mac_out.sum(dim=-1, keepdim=True)
        mac_out_old[avail_actions == 0] = 0
        pi_old = mac_out_old.view(-1, self.n_actions)

        # Calculate policy grad with mask
        pi_taken_old = th.gather(pi_old, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken_old[mask_long.squeeze(-1) == 0] = 1.0
        log_pi_taken_old = th.log(pi_taken_old)

        log_pi_taken_old = log_pi_taken_old.reshape(-1, self.n_agents)
        log_pi_taken_old = log_pi_taken_old * mask_alive

        # ______pg2___new pi theta
        self._update_policy()  # update policy_new to new params

        mac_out_new = []
        self.policy_new.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs_tmp = self.policy_new.forward(batch, t=t, test_mode=True)
            mac_out_new.append(agent_outs_tmp)
        mac_out_new = th.stack(mac_out_new, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out_new[avail_actions == 0] = 0
        mac_out_new = mac_out_new / mac_out.sum(dim=-1, keepdim=True)
        mac_out_new[avail_actions == 0] = 0

        pi_new = mac_out_new.view(-1, self.n_actions)
        # Calculate policy grad with mask
        pi_taken_new = th.gather(pi_new, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken_new[mask_long.squeeze(-1) == 0] = 1.0
        log_pi_taken_new = th.log(pi_taken_new)

        log_pi_taken_new = log_pi_taken_new.reshape(-1, self.n_agents)
        log_pi_taken_new = log_pi_taken_new * mask_alive
        neglogpac_new = - log_pi_taken_new.sum(-1)

        pi2 = log_pi_taken.reshape(-1, self.n_agents).sum(-1).clone()
        ratio_new = th.exp(- pi2 - neglogpac_new)

        adv_ex = (target_ex - v_ex.detach()).detach()
        adv_ex = (adv_ex - adv_ex.mean()) / (adv_ex.std() + 1e-8)

        # _______ gadient for pg 1 and 2---
        mask_tnagt = critic_mask.repeat(1, 1, self.n_agents)

        pg_loss1 = (log_pi_taken_old.view(-1, 1) * mask_long).sum() / mask_long.sum()
        pg_loss2 = ((adv_ex.view(-1) * ratio_new) * mask.squeeze(-1)).sum() / mask.sum()
        self.policy_old.agent.zero_grad()
        pg_loss1_grad = th.autograd.grad(pg_loss1, self.policy_old.parameters())

        self.policy_new.agent.zero_grad()
        pg_loss2_grad = th.autograd.grad(pg_loss2, self.policy_new.parameters())

        grad_total = 0
        for grad1, grad2 in zip(pg_loss1_grad, pg_loss2_grad):
            grad_total += (grad1 * grad2).sum()

        target_mix = target_mix.reshape(-1, max_t - 1, self.n_agents)
        pg_ex_loss = ((grad_total.detach() * target_mix) * mask_tnagt).sum() / mask_tnagt.sum()

        intrinsic_loss = pg_ex_loss + vf_coef * v_ex_loss
        self.intrinsic_optimiser.zero_grad()
        intrinsic_loss.backward()

        self.intrinsic_optimiser.step()

        self._update_policy_piold()

        # ______config tensorboard
        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "value_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask_long).sum().item() / mask_long.sum().item(),
                                 t_env)
            self.logger.log_stat("liir_loss", liir_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm_policy, t_env)
            self.logger.log_stat("pi_max",
                                 (pi.max(dim=1)[0] * mask_long.squeeze(-1)).sum().item() / mask_long.sum().item(),
                                 t_env)

            reward1 = rewards.reshape(-1, 1)
            self.logger.log_stat('rewards_mean', (reward1 * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        r_in, target_vals, target_val_ex = self.target_critic(batch)

        r_in, _, target_val_ex_opt = self.critic(batch)
        r_in_taken = th.gather(r_in, dim=3, index=actions)
        r_in = r_in_taken.squeeze(-1)

        target_vals = target_vals.squeeze(-1)

        targets_mix, targets_ex = build_td_lambda_targets_v2(rewards, terminated, mask, target_vals, self.n_agents,
                                                          self.args.gamma, self.args.td_lambda, r_in, target_val_ex)

        vals_mix = th.zeros_like(target_vals)[:, :-1]
        vals_ex = target_val_ex_opt[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "value_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            _, q_t, _ = self.critic(batch, t)  # 8,1,3,1,
            vals_mix[:, t] = q_t.view(bs, self.n_agents)
            targets_t = targets_mix[:, t]

            td_error = (q_t.view(bs, self.n_agents) - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["value_mean"].append((q_t.view(bs, self.n_agents) * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return vals_mix, running_log, targets_mix, targets_ex, vals_ex, r_in

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _update_policy(self):
        self.policy_new.load_state(self.mac)

    def _update_policy_piold(self):
        self.policy_old.load_state(self.mac)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))