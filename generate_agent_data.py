import numpy as np
import pandas as pd
import random

from keychain import Keychain as kc
from utilities import get_params


def generate_agents_data(num_agents, ratio_mutating, agent_attributes, simulation_timesteps, num_origins, num_destinations):

    """
    Generates agents data
    Constructs a dataframe, where each row is an agent and columns are attributes
    """

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents

    # Generating agent data
    for id in range(num_agents):
        
        # Decide on agent type 
        kind_token, agent_type = random.random(), kc.TYPE_HUMAN
        if kind_token < ratio_mutating:
            agent_type = kc.TYPE_MACHINE

        # Randomly assign origin & destination
        origin, destination = random.randrange(num_origins), random.randrange(num_destinations)

        # Randomly assign start time (normal dist)
        mean_timestep = simulation_timesteps / 2
        std_dev_timestep = simulation_timesteps / 6
        start_time = int(np.random.normal(mean_timestep, std_dev_timestep))
        start_time = max(0, min(simulation_timesteps, start_time))

        # Registering to the dataframe
        agent_features = [id, origin, destination, start_time, agent_type]
        agent_dict = {attribute : feature for attribute, feature in zip(agent_attributes, agent_features)}
        agents_df.loc[id] = agent_dict

    return agents_df



def save_agents(agents_df, save_to):
    agents_df.to_csv(save_to, index = False)



if __name__ == "__main__":
    params = get_params(kc.PARAMS_PATH)
    params = params[kc.AGENT_GEN]

    num_agents = params[kc.NUM_AGENTS]
    ratio_mutating = params[kc.RATIO_MUTATING]
    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])

    agents_data_path = kc.AGENTS_DATA_PATH

    agents_df = generate_agents_data(num_agents, ratio_mutating, agent_attributes, simulation_timesteps, num_origins, num_destinations)
    save_agents(agents_df, agents_data_path)
    print(f'[SUCCESS] Generated agent data and saved to: {agents_data_path}')