
import numpy as np
from routerl import Keychain as kc
from routerl import MachineAgent
from routerl import TrafficEnvironment


class MyExtendedMutation(TrafficEnvironment):

    def mutation(self) -> None:
            """
            Perform mutation by converting selected human agents into machine agents.

            This method identifies a subset of human agents that start after the 25th percentile of
            start times of other vehicles, removes a specified number of these agents, and replaces them with machine agents.

            Raises:
                ValueError: If there are insufficient human agents available for mutation.
            """


            # Mutate to a human that starts after the 25% of the rest of the vehicles
            start_times = [human.start_time for human in self.human_agents]
            percentile_25 = np.percentile(start_times, 25)

            filtered_human_agents = [human for human in self.human_agents if human.start_time > percentile_25]

            every_two_humans = []
            number_of_machines_to_be_added = self.agent_params[kc.NEW_MACHINES_AFTER_MUTATION]

            for i in range(len(self.human_agents)):
                if len(every_two_humans) >= number_of_machines_to_be_added:
                    break

                if i % 2 != 0 and i > 2: ## So that there are 3 human agents before all the AVs
                    every_two_humans.append(self.human_agents[i])

            random_humans_deleted = []

            if len(filtered_human_agents) < number_of_machines_to_be_added:
                raise ValueError(
                    f"Insufficient human agents for mutation. Required: {number_of_machines_to_be_added}, "
                    f"Available: {len(filtered_human_agents)}.\n"
                    f"Decrease the number of machines to be added after the mutation.\n"
                )

            for human in every_two_humans:
                self.human_agents.remove(human)

                random_humans_deleted.append(human)
                self.machine_agents.append(MachineAgent(human.id,
                                                        human.start_time,
                                                        human.origin, 
                                                        human.destination, 
                                                        self.agent_params[kc.MACHINE_PARAMETERS], 
                                                        self.simulation_params[kc.NUMBER_OF_PATHS]))
                self.possible_agents.append(str(human.id))


            """for i in range(0, number_of_machines_to_be_added):
                random_human = random.choice(filtered_human_agents)

                self.human_agents.remove(random_human)
                filtered_human_agents.remove(random_human)

                random_humans_deleted.append(random_human)
                self.machine_agents.append(MachineAgent(random_human.id,
                                                        random_human.start_time,
                                                        random_human.origin, 
                                                        random_human.destination, 
                                                        self.agent_params[kc.MACHINE_PARAMETERS], 
                                                        self.simulation_params[kc.NUMBER_OF_PATHS]))
                self.possible_agents.append(str(random_human.id))"""


            self.n_agents = len(self.possible_agents)
            self.all_agents = self.machine_agents + self.human_agents
            self.machines = True
            self.human_learning = False
            
            self._initialize_machine_agents()
