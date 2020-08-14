from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env

from absl import app
import random

class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgent, self).__init__()

        self.attack_coordinate = None

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and  obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and  obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def get_unit_by_type(self, obs, unit_type):
        return  [unit for unit in obs.observation.feature_units
                       if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        if obs.first():
            player_x, player_y = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()

            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinate = (47, 47)
            else:
                self.attack_coordinate = (12, 16)

        Zergling_units = self.get_unit_by_type(obs, units.Zerg.Zergling)

        if len(Zergling_units) > 0:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap('now', self.attack_coordinate)
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army('select')

        SpawningPool_pools_units = self.get_unit_by_type(obs, units.Zerg.SpawningPool)

        if len(SpawningPool_pools_units) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    x = random.randint(0, 63)
                    y = random.randint(0, 63)
                    return actions.FUNCTIONS.Build_SpawningPool_screen('now', (x, y))

            drone_units = self.get_unit_by_type(obs, units.Zerg.Drone)

            if len(drone_units) > 0:
                drone_units = random.choice(drone_units)
                return actions.FUNCTIONS.select_point('select_all_type', (drone_units.x, drone_units.y))

        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            if actions.FUNCTIONS.Train_Zergling_quick.id in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Zergling_quick('now')

        larva_units = self.get_unit_by_type(obs, units.Zerg.Larva)

        if len(larva_units) > 0:
            larva_units = random.choice(larva_units)
            return actions.FUNCTIONS.select_point('select_all_type', (larva_units.x, larva_units.y))

        return actions.FUNCTIONS.no_op()


def main(unused_arg):
    agent = ZergAgent()
    map = 'AbyssalReef'

    try:
        while True:
            with sc2_env.SC2Env(map_name=map, players=[sc2_env.Agent(sc2_env.Race.zerg),
                                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=1,
                                game_steps_per_episode=0, visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)