import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_td_lambda_targets_v2(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda, r_in, v_ex):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    # td_lambda = 1
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))

    ret_ex = v_ex.new_zeros(*v_ex.shape)

    ret_ex[:, -1] = v_ex[:, -1] * (1 - th.sum(terminated, dim=1))

    theta = 0.01
    #     theta = 1.
    rewards_broad = rewards.repeat(1, 1, r_in.shape[2])

    if len(r_in.shape) > 3:
        r_in = r_in.squeeze(-1)

    rewards_mix = rewards_broad + theta * r_in[:, :-1, :]

    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret_ex[:, t] = td_lambda * gamma * ret_ex[:, t + 1] + mask[:, t] * (
                    rewards[:, t] + (1 - td_lambda) * gamma * v_ex[:, t + 1] * (1 - terminated[:, t]))

        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                    rewards_mix[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))

    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1], ret_ex[:, :-1]
