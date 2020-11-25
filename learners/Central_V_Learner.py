from modules.critics.CentralV import CentralV_Critic
import torch
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch


class CentralV_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.n_actions = args.n_actions
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.mac = mac
        self.logger = logger
        self.args = args

        self.critic_training_steps = 0
        self.last_target_update_step = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = CentralV_Critic(self.state_shape, args)
        self.target_critic = CentralV_Critic(self.state_shape, args)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.rnn_parameters = list(self.mac.parameters())
        self.critic_parameters = list(self.critic.parameters())

        self.critic_optimizer = RMSprop(self.critic_parameters, lr=args.critic_lr)
        self.agent_optimiser = RMSprop(self.rnn_parameters, lr=args.lr)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents)

        td_error, critic_train_stats = self._train_critic(batch, rewards,
                                                        mask)

        actions = actions[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        pi_taken = torch.gather(mac_out, dim=3, index=actions).squeeze(3)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        centralV_loss = - ((td_error.detach() * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        centralV_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("coma_loss", centralV_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, mask):
        r, terminated = batch['reward'][:, :-1], batch['terminated'][:, :-1].type(torch.FloatTensor)
        v_evals, v_targets = [], []
        for t in reversed(range(rewards.size(1))):
            state = batch["state"][:, t]
            nextState = batch["state"][:, t + 1]
            val = self.critic(state)
            target_val = self.target_critic(nextState)

            v_evals.append(val)
            v_targets.append(target_val)

        v_evals = torch.stack(v_evals, dim=1)  # (episode_num, max_episode_len, 1)
        v_targets = torch.stack(v_targets, dim=1)

        re_mask = mask.repeat(1, 1, self.n_agents)
        targets = r + self.args.gamma * v_targets * (1 - terminated)
        td_error = targets.detach() - v_evals
        masked_td_error = re_mask * td_error

        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": []
        }

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)

        self.critic_training_steps += 1

        return td_error, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), "{}/critic.th".format(path))
        torch.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        torch.save(self.critic_optimizer.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(torch.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            torch.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimizer.load_state_dict(
            torch.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
