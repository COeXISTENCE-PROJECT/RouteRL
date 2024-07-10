import logging
import numpy as np
import random

from abc import ABC, abstractmethod
from keychain import Keychain as kc

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class Phases(ABC):
    """Abstract base class for phases."""

    def __init__(self, phases_params):
        """Initialize a phase"""
        self.phases_params = phases_params

    @abstractmethod
    def __call__(self):
        pass


class HumanLearning_Mutation(Phases):

    def __init__(self, phases_params):
        """Initialize phases function."""
        super().__init__(phases_params)
        print("inside init function\n\n")

    def __call__(self):
        # Define what should happen when the instance is called
        print("HumanLearning_Mutation instance is called")


    def mutation(self):
        start_times = [human.start_time for human in self.human_agents]
        percentile_25 = np.percentile(start_times, 25)

        # Filter the human agents whose start_time is higher than the 25th percentile
        filtered_human_agents = [human for human in self.human_agents if human.start_time > percentile_25]

        number_of_machines = self.agent_params[kc.NUM_MACHINE_AGENTS]

        ### Need to mutate to humans that have start time after the 25% of the rest of the vehicles
        random_humans_deleted = []

        for i in range(0, number_of_machines):
            random_human = random.choice(filtered_human_agents)
            print("Human that will be mutated is: ", random_human.id)

            self.human_agents.remove(random_human)
            filtered_human_agents.remove(random_human)

            random_humans_deleted.append(random_human)
            self.machine_agents.append(MachineAgent(random_human.id,
                                                    random_human.start_time,
                                                    random_human.origin, 
                                                    random_human.destination, 
                                                    self.agent_params[kc.MACHINE_PARAMETERS], 
                                                    self.simulation_params[kc.NUMBER_OF_PATHS]))
            print("The new machine agent is: ", random_human.id)
            self.possible_agents.append(str(random_human.id))
            print("self.possible agents is: ", self.possible_agents, "\n\n")

        self.n_agents = len(self.possible_agents)
        self.machines = True
        self.humans_learning = False

        logging.info("Now there are %s human agents.\n", len(self.human_agents))

        self._initialize_machine_agents(mutation=True)    

    def human_learning(self):
        pass