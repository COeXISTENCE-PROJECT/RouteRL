import logging
import numpy as np
import pandas as pd
import random

from RouteRL.environment import HumanAgent
from RouteRL.environment import MachineAgent
from ..keychain import Keychain as kc


logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def generate_agents(params, free_flow_times, generate_data, seed=23423):

    """
    Generates agent objects
    """

    set_seed(seed)
    if generate_data:
        agents_data_df = generate_agent_data(params, seed)
    else:
        agents_data_df = pd.read_csv(params[kc.AGENTS_CSV_PATH])

    # Getting parameters
    action_space_size = params[kc.ACTION_SPACE_SIZE]
    agents = list()     # Where we will store & return agents
    
    # Generating agent objects from generated agent data
    for _, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]

        if row_dict[kc.AGENT_KIND] == kc.TYPE_MACHINE:
            agent_params = params[kc.HUMAN_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            mutate_to = MachineAgent(id, start_time, origin, destination, params[kc.MACHINE_PARAMETERS], action_space_size)
            agents.append(mutate_to)
        elif row_dict[kc.AGENT_KIND] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            agents.append(HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge))
        else:
            raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_KIND])
    return agents



def generate_agent_data(params, seed=23423):

    """
    Generates agent data
    Constructs a dataframe, where each row is an agent and columns are attributes
    """
    
    num_agents = params[kc.NUM_AGENTS]
    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])
    rng = set_seed(seed)

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents
    
    mean_timestep = simulation_timesteps / 2
    std_dev_timestep = simulation_timesteps / 6

    # Generating agent data
    for id in range(num_agents):
        
        # Decide on agent type 
        agent_type = kc.TYPE_HUMAN

        # Randomly assign origin & destination
        origin, destination = random.randrange(num_origins), random.randrange(num_destinations)

        # Randomly assign start time (normal dist)
        start_time = int(rng.normal(mean_timestep, std_dev_timestep))
        start_time = max(0, min(simulation_timesteps, start_time))

        # Registering to the dataframe
        agent_features = [id, origin, destination, start_time, agent_type]
        agent_dict = {attribute : feature for attribute, feature in zip(agent_attributes, agent_features)}
        agents_df.loc[id] = agent_dict
        
    agents_df.to_csv(params[kc.AGENTS_CSV_PATH], index=False)
    return agents_df


def set_seed(seed: int) -> None:
    """ Set the seed for random number generation. """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng
