import pandas as pd
import random

from agent import MachineAgent, HumanAgent


def create_agent_objects(params):

    """
    Generates agent objects
    """

    # Getting parameters
    agents_data_path = params["agents_data_path"]
    agent_attributes = params["agent_attributes"]
    simulation_timesteps = params["simulation_timesteps"]
    agent_start_intervals = params["agent_start_intervals"]
    action_space_size = params["action_space_size"]
    learning_params = params["agent_learning_parameters"]
    
    # Generating agent data
    agents_data_df = generate_agents_data(agent_attributes, simulation_timesteps, agent_start_intervals, agents_data_path)
    agents = list() # Where we will store & return agents
    
    # Generating agent objects from generated agent data
    for idx, row in agents_data_df.iterrows():
        row_dict = row.to_dict()
        id, start_time, origin, destination = row_dict['id'], row_dict['start_time'], row_dict['origin'], row_dict['destination']
        if row_dict['agent_type'] == 'm':
            agents.append(MachineAgent(id, start_time, origin, destination, action_space_size, learning_params))
        elif row_dict['agent_type'] == 'h':
            agents.append(HumanAgent(id, start_time, origin, destination, action_space_size))
        else:
            print('[AGENT TYPE INVALID] Unrecognized agent type: ' + row_dict['agent_type'])

    print(f'[SUCCESS] Created agent objects (%d)' % (len(agents)))
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
            agent_features = [id_counter, 0, 0, t, 'm']
            agent = {agent_attributes[i] : agent_features[i] for i in range(len(agent_features))}   # Agent that goes to 0 from 0

            agents_df.loc[id_counter] = agent
            id_counter += 1

            agent_features = [id_counter, 1, 1, t, 'm']
            agent = {agent_attributes[i] : agent_features[i] for i in range(len(agent_features))}   # Agent that goes to 1 from 1

            agents_df.loc[id_counter] = agent
            id_counter += 1

    agents_df.to_csv(save_to, index = False)
    print('[SUCCESS] Generated agent data and saved to: ' + save_to)

    return agents_df
