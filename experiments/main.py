import sys
import os
from tqdm import tqdm
from keychain import Keychain as kc

original_sys_path = sys.path.copy()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from RouteRL.environment.environment import TrafficEnvironment
from RouteRL.services import plotter

from RouteRL.create_agents import create_agent_objects
from RouteRL.utilities import check_device
from RouteRL.utilities import get_params
from RouteRL.utilities import set_seeds

from pettingzoo.test import api_test

sys.path = original_sys_path


def main(params):
    env = TrafficEnvironment(params[kc.RUNNER], params[kc.ENVIRONMENT], params[kc.SIMULATOR], params[kc.AGENT_GEN], params[kc.AGENTS], params[kc.PHASE])

    env.start()
    env.reset()

    num_episodes = 10

    for agent in env.agent_iter():
         print("agent", agent, "\n\n")
         break

    for episode in range(num_episodes):
        print("episode is: ", episode, "\n\n")
        env.reset()

        
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample()

        env.step(action)

    """env.mutation()
    env.reset()

    for episode in range(num_episodes):
        print("episode is: ", episode, "\n\n")
        env.reset()

        while(1):
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                print("truncations break\n\n")
                break
            else:
                # This is where you would insert your policy
                action = env.action_space(agent).sample()

            env.step(action)


    env.stop()"""

    env.stop()

    plotter(params[kc.PLOTTER])


if __name__ == "__main__":
    check_device()
    set_seeds()
    params = get_params(kc.PARAMS_PATH)
    main(params)