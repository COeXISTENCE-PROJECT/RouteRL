import functools
import pandas as pd

from gymnasium.spaces import Box, Discrete
from prettytable import PrettyTable

from keychain import Keychain as kc
from services import Simulator
from utilities import create_agent_objects
from utilities import make_dir


class TrafficEnvironment:

    def __init__(self, environment_params, simulation_params, agent_params, render_mode=None):
        self.simulator = Simulator(simulation_params)
        self.agents = create_agent_objects(agent_params, self.calculate_free_flow_times())

        self.possible_agents = [agent.id for agent in self.agents] 

        self.observation_spaces = {
            agent: Box(low=0, high=1, shape=(1,), dtype=float) for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: Discrete(3) for agent in self.possible_agents
        }

        self.render_mode = render_mode

        print("[SUCCESS] Environment initiated!")


    def start(self):
        self.simulator.start_sumo()
        state = None
        return state

    def stop(self):
        self.simulator.stop_sumo()
        state = None
        return state

    def reset(self):
        self.simulator.reset_sumo()

        observations = {
            a: Box(low=0, high=1, shape=(1,), dtype=float).sample() for a in self.possible_agents
        }

        infos = {a: {}  for a in self.possible_agents}


        return observations, infos


    def calculate_free_flow_times(self):
        free_flow_times = self.simulator.calculate_free_flow_times()
        self.print_free_flow_times(free_flow_times)
        self.save_free_flow_times_csv(free_flow_times)
        return free_flow_times


    def step(self, joint_action):        

        sumo_df = self.simulator.run_simulation_iteration(joint_action)
        joint_reward = self.calculate_rewards(sumo_df)

        sample_observation = {
            a: (Box(low=0, high=1, shape=(1,), dtype=float).sample()) for a in self.possible_agents
        }

        terminated = {
            terminated: True for terminated in self.possible_agents
        }

        truncated = {
            truncated: 1 for truncated in self.possible_agents
        }

        info = {a: {} for a in self.agents} 

        if any(terminated.values()) or all(truncated.values()):
            self.agents = []

        return sample_observation, joint_reward, terminated, truncated, info


    def calculate_rewards(self, sumo_df):
        # Calculate reward from cost (skipped)
        # Turn cost column to reward, drop everything but id and reward
        reward_df = sumo_df.rename(columns={kc.TRAVEL_TIME : kc.REWARD})
        reward_df = reward_df[[kc.AGENT_ID, kc.REWARD]]
        return reward_df


    def get_last_sim_duration(self):
        return self.simulator.get_last_sim_duration()


    def print_free_flow_times(self, free_flow_times):
        table = PrettyTable()
        table.field_names = ["Origin", "Destination", "Index", "FF Time"]

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                table.add_row([od[0], od[1], idx, "%.3f"%time])
            table.add_row(["----", "----", "----", "----"])

        print("------ Free flow travel times ------")
        print(table)


    def save_free_flow_times_csv(self, free_flow_times):
        cols = [kc.ORIGINS, kc.DESTINATIONS, kc.PATH_INDEX, kc.FREE_FLOW_TIME]
        free_flow_pd = pd.DataFrame(columns=cols)

        for od, times in free_flow_times.items():
            for idx, time in enumerate(times):
                free_flow_pd.loc[len(free_flow_pd.index)] = [od[0], od[1], idx, time]
        save_to = make_dir(kc.RECORDS_FOLDER, kc.FREE_FLOW_TIMES_CSV_FILE_NAME)
        free_flow_pd.to_csv(save_to, index = False)
        print(f"[SUCCESS] Free-flow travel times calculated and saved to: {save_to}")


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(1,), dtype=float)


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)