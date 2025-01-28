import numpy as np
import pandas as pd
import random

from .keychain import Keychain as kc


def generate_agents_data(params):

    """
    Generates agents data
    Constructs a dataframe, where each row is an agent and columns are attributes
    """
    
    num_agents = params[kc.NUM_AGENTS]
    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])
    agents_data_path = params[kc.AGENTS_DATA_PATH]

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents

    # Generating agent data
    for id in range(num_agents):
        
        # Decide on agent type 
        agent_type = kc.TYPE_HUMAN

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
        
        agents_df.to_csv(agents_data_path, index = False)

    return agents_df