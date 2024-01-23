import pandas as pd
from prettytable import PrettyTable

from agent import MachineAgent, HumanAgent
from keychain import Keychain as kc


def create_agent_objects(params):

    """
    Generates agent objects
    """

    # Getting parameters
    agents_data_path = params[kc.AGENTS_DATA_PATH]
    simulation_timesteps = params[kc.SIMULATION_TIMESTEPS]
    agent_start_intervals = params[kc.AGENT_START_INTERVALS]
    learning_params = params[kc.AGENT_LEARNING_PARAMETERS]
    agent_attributes = params[kc.AGENT_ATTRIBUTES]
    
    # Generating agent data
    agents_data_df = generate_agents_data(agent_attributes, simulation_timesteps, agent_start_intervals, agents_data_path)
    agents = list() # Where we will store & return agents
    
    # Generating agent objects from generated agent data
    for _, row in agents_data_df.iterrows():
        row_dict = row.to_dict()

        id, start_time = row_dict[kc.AGENT_ID], row_dict[kc.AGENT_START_TIME]
        origin, destination = row_dict[kc.AGENT_ORIGIN], row_dict[kc.AGENT_DESTINATION]

        if row_dict[kc.AGENT_TYPE] == kc.TYPE_MACHINE:
            agents.append(MachineAgent(id, start_time, origin, destination, learning_params))
        elif row_dict[kc.AGENT_TYPE] == kc.TYPE_HUMAN:
            agents.append(HumanAgent(id, start_time, origin, destination, learning_params))
        else:
            print('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict[kc.AGENT_TYPE])

    print(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
    print_agents(agents, agent_attributes, print_every=50)
    return agents



def generate_agents_data(agent_attributes, simulation_timesteps, agent_start_intervals, save_to = None):

    """
    Generates agents data
    Constructs a dataframe, where each row is an agent and columns are attributes
    Saves it to specified named csv file
    """

    agents_df = pd.DataFrame(columns=agent_attributes)  # Where we store our agents
    id_counter = 0

    for t in range(simulation_timesteps):
        if not t % agent_start_intervals:
            agent_features = [id_counter, 0, 0, t, kc.TYPE_MACHINE]
            agent = {agent_attributes[i] : agent_features[i] for i in range(len(agent_features))}   # Agent that goes to 0 from 0

            agents_df.loc[id_counter] = agent
            id_counter += 1

            agent_features = [id_counter, 1, 1, t, kc.TYPE_MACHINE]
            agent = {agent_attributes[i] : agent_features[i] for i in range(len(agent_features))}   # Agent that goes to 1 from 1

            agents_df.loc[id_counter] = agent
            id_counter += 1

    agents_df.to_csv(save_to, index = False)
    print('[SUCCESS] Generated agent data and saved to: ' + save_to)

    return agents_df


def print_agents(agents, agent_attributes, print_every=1):
    table = PrettyTable()
    table.field_names = agent_attributes

    for a in agents:
        if not (a.id % print_every):
            table.add_row([a.id, a.origin, a.destination, a.start_time, a.__class__.__name__])

    if print_every > 1: print("--- Showing every %dth agent ---" % (print_every))
    print(table)