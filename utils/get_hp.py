from envs.starcraft2.maps import get_map_params


def get_HP(args, obs):

    map_params = get_map_params(args.env_args["map_name"])
    unit_type_bits = map_params["unit_type_bits"]
    _agent_race = map_params["a_race"]

    own_feats = unit_type_bits
    shield_bits_ally = 1 if _agent_race == "P" else 0

    own_feats += 1 + shield_bits_ally

    return obs[:, : , obs.shape[2] - own_feats]