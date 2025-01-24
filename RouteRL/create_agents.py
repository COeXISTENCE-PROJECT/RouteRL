import logging
import os
import pandas as pd

#from RouteRL.environment import HumanAgent
from RouteRL.environment import MachineAgent
from .keychain import Keychain as kc

from pathlib import Path

# Set the relative path to go up two levels and then to Simulator_human_behaviour
simulator_path = (Path(__file__).parent.parent / "../Simulator_human_behaviour").resolve()

# Convert it to a string if needed
simulator_path_str = str(simulator_path)

# Add it to sys.path
import sys
sys.path.append(simulator_path_str)

# Debugging path
print(f"Simulator path resolved to: {simulator_path_str}")

# Now import from the module
from agent import HumanAgent


logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def create_agent_objects(params, free_flow_times, kwargs):

    """
    Generates agent objects
    """

    # Getting parameters

    action_space_size = params[kc.ACTION_SPACE_SIZE]
    agents_data_path = params[kc.CREATE_AGENTS_DATA_PATH]
    if os.path.isfile(agents_data_path):
        logging.info("[CONFIRMED] Agents data file is ready.")
    else:
        raise FileNotFoundError("Agents data file is not ready. Please generate agents data first.")
    
    agents_data_df = pd.read_csv(agents_data_path)
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
            #new_agent = HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, mutate_to)
            agents.append(mutate_to)
        elif row_dict[kc.AGENT_KIND] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            agents.append(HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, action_space_size, **kwargs))
        else:
            raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_KIND])

    logging.info(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
    return agents
