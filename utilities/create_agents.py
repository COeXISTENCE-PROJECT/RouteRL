import pandas as pd
import random

from prettytable import PrettyTable

from agent import MachineAgent, HumanAgent
from keychain import Keychain as kc


def create_agent_objects(params, free_flow_times):

    """
    Generates agent objects
    """

    # Getting parameters
    agents_data_path = params[kc.AGENTS_DATA_PATH]
    num_agents = params[kc.NUM_AGENTS]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    num_origins = len(params[kc.ORIGINS])
    num_destinations = len(params[kc.DESTINATIONS])

    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    action_space_size = params[kc.ACTION_SPACE_SIZE]
    
    # Generating agent data
    agents_data_df = generate_agents_data(num_agents, agent_attributes, simulation_timesteps, num_origins, num_destinations, agents_data_path)
    agents = list() # Where we will store & return agents
    
    # Generating agent objects from generated agent data
    for _, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]

        if row_dict[kc.AGENT_TYPE] == kc.TYPE_MACHINE: ##### Changed
            agent_params = params[kc.HUMAN_AGENT_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            mutate_to = MachineAgent(id, start_time, origin, destination, params[kc.MACHINE_AGENT_PARAMETERS], action_space_size)
            new_agent = HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge, mutate_to)
            agents.append(new_agent)
        elif row_dict[kc.AGENT_TYPE] == kc.TYPE_HUMAN:
            agent_params = params[kc.HUMAN_AGENT_PARAMETERS]
            initial_knowledge = free_flow_times[(origin, destination)]
            agents.append(HumanAgent(id, start_time, origin, destination, agent_params, initial_knowledge))
        else:
            print('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_TYPE])

    print(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
    #print_agents(agents, agent_attributes, print_every=50)
    return agents



def generate_agents_data(num_agents, agent_attributes, simulation_timesteps, num_origins, num_destinations, save_to = None):

    """
    Generates agents data
    Constructs a dataframe, where each row is an agent and columns are attributes
    Saves it to specified named csv file
    """

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents

    for id in range(num_agents):
        # Generating agent data
        agent_type = kc.TYPE_MACHINE if random.randint(0,10) > 8 else kc.TYPE_HUMAN ###### 80% of the agents are humans
        origin, destination = random.randrange(num_origins), random.randrange(num_destinations)
        start_time = random.randrange(simulation_timesteps)

        # Registering to the dataframe
        agent_features = [id, origin, destination, start_time, agent_type]
        agent_dict = {agent_attributes[i] : agent_features[i] for i in range(len(agent_features))}
        agents_df.loc[id] = agent_dict

    agents_df.to_csv(save_to, index = False)
    print('[SUCCESS] Generated agent data and saved to: ' + save_to)
    return agents_df



def print_agents(agents, agent_attributes, print_every=1):
    table = PrettyTable()
    table.field_names = agent_attributes

    for a in agents:
        if not (a.id % print_every):
            table.add_row([a.id, a.origin, a.destination, a.start_time, a.__class__.__name__])

    if print_every > 1: print("------ Showing every %dth agent ------" % (print_every))
    print(table)