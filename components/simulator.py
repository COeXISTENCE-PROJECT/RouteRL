import os
import pandas as pd
import random
import traci

from abc import ABC, abstractmethod

from keychain import Keychain as kc
from utilities import confirm_env_variable


class BaseSimulator(ABC):

    """
    The interface between the simulation software and the environment
    """

    def __init__(self,):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def simulate_episode(self):
        pass



class SumoSimulator(BaseSimulator):

    def __init__(self, params):
        super().__init__()

        self.sumo_config_path = kc.SUMO_CONFIG_PATH
        self.paths_csv_path = kc.PATHS_CSV_SAVE_PATH
        self.routes_xml_path = kc.ROUTES_XML_SAVE_PATH

        self.sumo_type = params[kc.SUMO_TYPE]
        self.env_var = params[kc.ENV_VAR]
        self.number_of_paths = params[kc.NUMBER_OF_PATHS]
        self.simulation_length = params[kc.SIMULATION_TIMESTEPS]

        self.sumo_id = f"{random.randint(0, 1000)}"
        self.last_simulation_duration = 0

        self._check_paths_ready()
        confirm_env_variable(self.env_var, append="tools")

        print("[SUCCESS] Simulator is ready to simulate!")

    #####################

    ##### CONFIG CHECK #####

    def _check_paths_ready(self):
        if os.path.isfile(self.paths_csv_path) and os.path.isfile(self.routes_xml_path):
            print("[CONFIRMED] Paths file is ready.")
        else:
            raise FileNotFoundError("Paths file is not ready. Please generate paths first.")
        
    #####################

    ##### PROPERTIES #####

    @property
    def last_simulation_duration(self):
        return self._last_simulation_duration
    
    @last_simulation_duration.setter
    def last_simulation_duration(self, duration):
        self._last_simulation_duration = duration

    #####################

    ##### SUMO CONTROL #####

    def start(self):
        sumo_cmd = [self.sumo_type, "-c", self.sumo_config_path]
        traci.start(sumo_cmd, label=self.sumo_id)

    def stop(self):
        traci.switch(self.sumo_id)
        traci.close()

    def reset(self):
        traci.switch(self.sumo_id)
        traci.load(['-c', self.sumo_config_path])

    #####################

    ##### SIMULATION #####

    def simulate_episode(self, joint_action):
        arrivals = {kc.AGENT_ID : list(), kc.ARRIVAL_TIME: list()}  # Where we save arrivals
        agents_stack = self._joint_action_to_sorted_stack(joint_action)  # Where we keep agents and their actions
        should_continue = True
        traci.switch(self.sumo_id)  # Attention: Ensure it remains the same in concurrent execution

        # Simulation loop
        while should_continue:
            timestep = int(traci.simulation.getTime())
            
            # Add vehicles to the simulation
            while agents_stack[-1][kc.AGENT_START_TIME] == timestep:
                row = agents_stack.pop()
                traci.switch(self.sumo_id)
                traci.vehicle.add(row[kc.AGENT_ID], row[kc.SUMO_ACTION])

            # Collect vehicles that have reached their destination
            arrived_now = traci.simulation.getArrivedIDList()   # returns a list of arrived vehicle ids
            arrived_now = [int(value) for value in arrived_now]   # Convert values to int

            for id in arrived_now:
                arrivals[kc.AGENT_ID].append(id)
                arrivals[kc.ARRIVAL_TIME].append(timestep)
            
            # Did all vehicles arrive?
            should_continue = len(arrivals[kc.AGENT_ID]) < len(joint_action)
            # Advance the simulation
            traci.simulationStep()
        
        # Needed for plots
        self.last_simulation_duration = timestep
        # Calculate travel times
        travel_times_df = self._prepare_travel_times_df(arrivals, joint_action)
        return travel_times_df
        

    def _prepare_travel_times_df(self, arrivals, joint_action):
        # Initiate the travel_time_df
        travel_times_df = pd.DataFrame(arrivals)

        # Retrieve the start times of the agents from the joint_action dataframe
        start_times_df = joint_action[[kc.AGENT_ID, kc.AGENT_START_TIME]]

        # Merge the travel_time_df with the start_times_df for travel time calculation
        travel_times_df = pd.merge(left=start_times_df, right=travel_times_df, on=kc.AGENT_ID, how='left')

        # Calculate travel time
        calculate_travel_duration = lambda row: ((row[kc.ARRIVAL_TIME] - row[kc.AGENT_START_TIME]) / 60.0)
        travel_times_df[kc.TRAVEL_TIME] = travel_times_df.apply(calculate_travel_duration, axis=1)

        # Retain only the necessary columns
        travel_times_df = travel_times_df[[kc.AGENT_ID, kc.TRAVEL_TIME]]
        return travel_times_df
    

    def _joint_action_to_sorted_stack(self, joint_action):
        # Sort the joint_action dataframe by start times (descending order for stack)
        sorted_joint_action = joint_action.sort_values(kc.AGENT_START_TIME, ascending=False)

        # Make a sumo_action column in sorted_joint_action dataframe
        sumonize_action = lambda row: f'{row[kc.AGENT_ORIGIN]}_{row[kc.AGENT_DESTINATION]}_{row[kc.ACTION]}'
        sorted_joint_action[kc.SUMO_ACTION] = sorted_joint_action.apply(sumonize_action, axis=1)

        # Create a stack of agents and their sumo actions
        agents_stack = [{kc.AGENT_START_TIME : -1}]     # with bottom placeholder

        for _, row in sorted_joint_action.iterrows():
            stack_row = {kc.AGENT_ID : f"{row[kc.AGENT_ID]}", kc.AGENT_START_TIME : row[kc.AGENT_START_TIME], kc.SUMO_ACTION : row[kc.SUMO_ACTION]}
            agents_stack.append(stack_row)

        return agents_stack
    
    #####################
    
    def get_free_flow_times(self):
        paths_df = pd.read_csv(self.paths_csv_path)
        origins = paths_df[kc.ORIGIN].unique()
        destinations = paths_df[kc.DESTINATION].unique()
        ff_dict = {(o, d): list() for o in origins for d in destinations}
        for _, row in paths_df.iterrows():
            ff_dict[(row[kc.ORIGIN], row[kc.DESTINATION])].append(row[kc.FREE_FLOW_TIME])
        return ff_dict