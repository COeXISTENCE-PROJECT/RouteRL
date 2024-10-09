import itertools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from RouteRL.keychain import Keychain as kc
from RouteRL.environment.environment import TrafficEnvironment
from RouteRL.utilities import get_params

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##################### Environment Creation #####################
params = get_params("params.json")

env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PLOTTER])

env.start()

##################### Human Learning #####################
num_episodes = 200

for episode in range(num_episodes):
    env.step()

##################### Mutation #####################
env.mutation()

print("env.human_agents", env.human_agents)
print("env.machine_agents", env.machine_agents)
print("\n\n\n")

env.reset()

actions = [0, 1]

for combination in itertools.product(actions, repeat=len(env.possible_agents)):
    print(combination)

    for action in combination:
        env.step(action)

    env.reset()

env.stop()