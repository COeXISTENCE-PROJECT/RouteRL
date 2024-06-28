import os
import pandas as pd

from components import HumanAgent
from components import MachineAgent
from keychain import Keychain as kc


def create_agent_objects(params, free_flow_times):

    """
    Generates agent objects
    """

    # Getting parameters
    action_space_size = params[kc.ACTION_SPACE_SIZE]

    agents_data_path = kc.AGENTS_DATA_PATH
    if os.path.isfile(agents_data_path):
        print("[CONFIRMED] Agents data file is ready.")
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
            new_agent = HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, mutate_to)
            agents.append(new_agent)
        elif row_dict[kc.AGENT_KIND] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            agents.append(HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge))
        else:
            raise ValueError('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_KIND])

    print(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
    return agents
