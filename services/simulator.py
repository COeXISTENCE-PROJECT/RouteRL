import networkx as nx
import pandas as pd
import traci
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from keychain import Keychain as kc
from services import SumoController


class Simulator:

    """
    The interface between traffic simulator (SUMO, HihgEnv, Flow) and the environment
    """

    def __init__(self, params):

        self.sumo_controller = SumoController(params)
        self.routes_xml_save_path = params[kc.ROUTES_XML_SAVE_PATH] # RK: why do we want to double variables - this is exactly the same and static?
        self.number_of_paths = params[kc.NUMBER_OF_PATHS] #RK: same here
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS] #RK: same here
        self.beta = params[kc.BETA] #why this is here? this is main of simulator

        self.origins = {i : origin for i, origin in enumerate(params[kc.ORIGINS])}  # RK: I think it should be more generic, and operate on sth like nx.get_nearest_node(lon, lat) - otherwise it is very error prone.
        self.destinations = {i : dest for i, dest in enumerate(params[kc.DESTINATIONS])}

        self.last_simulation_duration = 0

        print("[SUCCESS] Simulator is ready to simulate!") # RK: replace with logger.log()


    def start_sumo(self):
        self.sumo_controller.sumo_start() # RK: why do we need this extra level of calls? why not directly to sumo_controller?

    def stop_sumo(self):
        self.sumo_controller.sumo_stop()

    def reset_sumo(self):
        self.sumo_controller.sumo_reset()

    
    def get_last_sim_duration(self): #RK: this can be nicely decorated with @ to make a property :)
        return self.last_simulation_duration
    
    def joint_action_to_sorted_stack(self, joint_action):
        # Sort the joint_action dataframe by start times (descending order for stack)
        sorted_joint_action = joint_action.sort_values(kc.AGENT_START_TIME, ascending=False)

        # Make a sumo_action column in sorted_joint_action dataframe
        sumonize_action = lambda row: f'{row[kc.AGENT_ORIGIN]}_{row[kc.AGENT_DESTINATION]}_{row[kc.ACTION]}'
        sorted_joint_action[kc.SUMO_ACTION] = sorted_joint_action.apply(sumonize_action, axis=1)

        # Create a stack of agents and their sumo actions
        stack_bottom_placeholder = {kc.AGENT_START_TIME : -1} #redundant line - this variable is used just once, make it direct.
        agents_stack = [stack_bottom_placeholder]

        for _, row in sorted_joint_action.iterrows(): # RK: typically iterrows is very slow, you may use .apply() or see what is fastest now.
            stack_row = {kc.AGENT_ID : f"{row[kc.AGENT_ID]}", kc.AGENT_START_TIME : row[kc.AGENT_START_TIME], kc.SUMO_ACTION : row[kc.SUMO_ACTION]}
            agents_stack.append(stack_row)

        return agents_stack
    

    def run_simulation_iteration(self, joint_action): #RK: Why iteration? Let's think about naming it: episode? It will be confusing later.
        arrivals = {kc.AGENT_ID : list(), kc.ARRIVAL_TIME: list()}  # Where we save arrivals #RK: what data type is this?
        agents_stack = self.joint_action_to_sorted_stack(joint_action)  # Where we keep agents and their actions
        should_continue = True

        # Simulation loop
        while should_continue:
            timestep = int(traci.simulation.getTime()) # RK: do we need this? or maybe add simulationStep everytime>

            # Add vehicles to the simulation
            while agents_stack[-1][kc.AGENT_START_TIME] == timestep:
                row = agents_stack.pop()
                traci.vehicle.add(row[kc.AGENT_ID], row[kc.SUMO_ACTION])

            # Collect vehicles that have reached their destination
            arrived_now = traci.simulation.getArrivedIDList()   # returns a list of arrived vehicle ids
            arrived_now = [int(value) for value in arrived_now]   # Convert values to int

            for id in arrived_now:
                arrivals[kc.AGENT_ID].append(id)
                arrivals[kc.ARRIVAL_TIME].append(timestep) #RK: are you sure you keep track of indexes? it seems easy to be mixed up
            
            # Did all vehicles arrive?
            should_continue = len(arrivals[kc.AGENT_ID]) < len(joint_action)
            # Advance the simulation
            traci.simulationStep()
        
        # Needed for plots
        self.last_simulation_duration = timestep
        # Calculate travel times
        travel_times_df = self.prepare_travel_times_df(arrivals, joint_action)
        # RK: here add: get_state
        return travel_times_df
        

    def prepare_travel_times_df(self, arrivals, joint_action): #RK: this shall be a more generic reward. including distance, travel time, emmisions, etc. it can also become part of the state and observation.
        # Initiate the travel_time_df
        travel_times_df = pd.DataFrame(arrivals)

        # Retrieve the start times of the agents from the joint_action dataframe
        start_times_df = joint_action[[kc.AGENT_ID, kc.AGENT_START_TIME]]

        # Merge the travel_time_df with the start_times_df for travel time calculation
        travel_times_df = pd.merge(left=start_times_df, right=travel_times_df, on=kc.AGENT_ID, how='left')

        # Calculate travel time
        calculate_travel_duration = lambda row: ((row[kc.ARRIVAL_TIME] - row[kc.AGENT_START_TIME]) / 60.0)  # RK: be very careful with dividing by 60! I'd avoid this, or be very precise when do you divide - only once!
        travel_times_df[kc.TRAVEL_TIME] = travel_times_df.apply(calculate_travel_duration, axis=1)

        # Retain only the necessary columns
        travel_times_df = travel_times_df[[kc.AGENT_ID, kc.TRAVEL_TIME]] # RK: maybe it can be pd.Series()? it is just indexed column.
        return travel_times_df

    
    def read_xml_file(self, file_path, element_name, attribute_name, attribute_name_2): # RK: add docstring.
        #RK: this shall be part of utils, not a cass method. it does not even use 'self'
        with open(file_path, 'r') as f:
            data = f.read()
        Bs_data_con = BeautifulSoup(data, "xml")
        
        connections = Bs_data_con.find_all(element_name)

        empty=[]
        for x in range(len(connections)):
            empty.append(str(connections[x]))

        from_=[]
        to_=[]
        for x in range(len(empty)):
            root = ET.fromstring(empty[x])
            from_.append(root.attrib.get(attribute_name))
            to_.append(root.attrib.get(attribute_name_2))

        from_db=pd.DataFrame(from_)
        to_db=pd.DataFrame(to_)
        return from_db, to_db